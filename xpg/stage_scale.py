"""SCALE stage: observation selection, ice rings, symmetry choice, scaling and error model, profile / summation
combination, zero-dose and shared-error checks, merged and unmerged outputs, report."""
from __future__ import annotations

import json
import os
from pathlib import Path

import gemmi
import numpy as np

from .geometry import rotation_matrices_np
from .ice import ice_spot_bands
from .io.legacy import write_legacy_hkl, write_xparm
from .io.mtz import write_unmerged_mtz
from .io.xpgfiles import write_geometry, write_spots
from .lattice import lattice_candidates
from .merging import cc_half_cutoff, format_stats_table, merging_statistics
from .models import Crystal, Experiment
from .models import d_spacing_from_exp
from .outputs import _isa_txt, write_merged_outputs
from .refine import refine
from .scaling import Scaler
from .stage_filters import FILTER_CYCLES
from .stage_lattices import LATTICE_TRIAGE
from .symmetry import score_space_groups, score_symmetry
from .symops import apply_rot, to_asu


# tNCS weak class with another spot shape (_combine_prf_sum): "scale" = its profile intensities scaled by the class
# I_sum / I_prf ratio, "sum" = summation intensities, "off" (XPG_TNCS_MODE). at 1.13-1.26 it was within noise or
# slightly worse, hence the threshold TNCS_MIN_RATIO.
TNCS_MODE = os.environ.get("XPG_TNCS_MODE", "scale")
TNCS_MIN_RATIO = 1.4

def _stats(s):
    """Merging statistics of a shell as plain numbers (JSON)."""
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in s.__dict__.items()}


class ScaleStage:
    """Pipeline mixin (see pipeline.Pipeline)."""

    def _write_bayes_merged(self, hasu, Is, ss, cell, sg, dcut, wavelength, nu=4.0):
        """XPG_BAYES_MERGE=1: xpg_bayes_merged.mtz -- robust Bayesian merge of the scaled observations (Student-t
        likelihood, Wilson prior with empirical-Bayes shell means; xpg/bayes_merge.py): posterior I, sig(I), F, sig(F)
        per unique reflection (Friedel mates merged), to the CC1/2 = 0.3 limit."""
        from .bayes_merge import bayes_merge
        from .io.mtz import write_merged_mtz
        from .merging import unique_ids
        uid, uk = unique_ids(hasu, None, anomalous=False)
        first = np.unique(uid, return_index=True)[1]
        hkl = np.asarray(hasu[first], int)
        ops = sg.operations()
        centric = np.asarray(ops.centric_flag_array(hkl), bool)
        eps = np.asarray(ops.epsilon_factor_without_centering_array(hkl), float)
        uc = gemmi.UnitCell(*cell)
        d = np.asarray(uc.calculate_d_array(hkl), float)
        res = bayes_merge(uid, Is, ss, centric, eps, 1.0 / d ** 2, nu=nu)
        m = dict(hkl=hkl, I=res["I"], sig=res["sigI"], F=res["F"], SIGF=res["sigF"])
        sel = (res["n_obs"] > 0) & (d >= dcut if dcut else True)
        mi = {k: v[sel] for k, v in m.items()}
        write_merged_mtz(self.wd / "xpg_bayes_merged.mtz", sg, cell, wavelength, mi,
                         [("IMEAN", "J", "I"), ("SIGIMEAN", "Q", "sig"), ("F", "F", "F"), ("SIGF", "Q", "SIGF")],
                         history=[f"XPG Bayesian merge (Student-t nu {nu:g}, Wilson prior), cut-off {dcut}"])
        self.log(f"  SCALE: Bayesian merge (Student-t nu {nu:g} likelihood, Wilson prior): {sel.sum()} unique "
                 f"reflections -> xpg_bayes_merged.mtz")

    # ------------------------------------------------------------------
    def _select_observations(self, refl, report=False):
        # MINPK covers partials at the sweep ends too: unrecorded frames reduce 'peak'
        ok = (refl["n_over"] == 0) & (refl["peak"] >= self.p.minpk)
        ok &= np.isfinite(refl["I"]) & (refl["sig"] > 0)
        # overloaded reflections fitted by the forward model from their unsaturated pixels (XPG_INTEGRATE_MODE=fm):
        # last resort only, for unique reflections without any other usable observation. Their error is set by the
        # profile model, not by counting statistics (simulation: -3% bias, scatter 6x their sigma even with one
        # overloaded pixel; sigma carries a 5% model error, see _forward_model_integrate)
        if "fm_overload_recovered" in refl:
            rec = np.asarray(refl["fm_overload_recovered"], bool) & np.isfinite(refl["I"]) & (refl["sig"] > 0) & ~ok
            if rec.any():
                from .merging import unique_ids
                hasu, _, _ = to_asu(refl["hkl"], self.st.space_group or gemmi.SpaceGroup("P 1"))
                uid, _ = unique_ids(hasu)
                have = np.zeros(uid.max() + 1, bool)
                have[uid[ok]] = True
                ok |= rec & ~have[uid]
        # overlapping pairs whose profiles nearly coincide (ill-conditioned joint fit) are unusable
        if "rho" in refl:
            ok &= ~(np.nan_to_num(refl["rho"]) > 0.8)
        # EXCLUDE_RESOLUTION_RANGE and detected ice rings (the background under a ring is not the smooth
        # local background assumed by integration: such reflections are biased)
        rings = self.st.results.get("ice_rings_excluded")
        if rings is None:
            rings = self._validate_ice_rings(refl, ok)
        ranges = list(self.p.exclude_resolution) + list(rings)
        excl = np.zeros_like(ok)
        if ranges:
            d = d_spacing_from_exp(refl["hkl"], self.st.exp)
            for lo, hi in ranges:
                excl |= (d >= min(lo, hi)) & (d <= max(lo, hi))
            ok &= ~excl
        if report:
            self._lost_report(refl, ok, excl)
            if ok.sum() < 50:
                # snapshots / raster scans picked up as sweeps ended in an IndexError
                raise RuntimeError(f"SCALE: only {int(ok.sum())} usable observations of {len(ok)}: nothing to scale "
                                   "(not a rotation sweep -- snapshots, raster scan -- or a wrong data range?)")
        return ok

    def _lost_report(self, refl, ok, excl):
        """Which predicted reflections are not used and why (Stage 3: data not used), incl. unique reflections
        lost entirely (no observation kept)."""
        sg = self.st.space_group or gemmi.SpaceGroup("P 1")
        hasu, _, plus = to_asu(refl["hkl"], sg)
        from .merging import unique_ids
        uid, _ = unique_ids(hasu, plus, anomalous=False)
        kept_u = np.zeros(uid.max() + 1, bool)
        kept_u[uid[ok]] = True
        rec = np.asarray(refl["fm_overload_recovered"], bool) if "fm_overload_recovered" in refl else np.zeros(len(ok), bool)
        rec = rec & ok & (refl["n_over"] > 0)
        if rec.any():
            self.log(f"  SCALE: {int(rec.sum())} overloaded observations kept (fitted by the forward model from their "
                     f"unsaturated pixels; unique reflections without another usable observation)")
        reasons = [("overloaded pixels", (refl["n_over"] > 0) & ~rec),
                   (f"PEAK < MINPK ({self.p.minpk:g}%)", (refl["n_over"] == 0) & (refl["peak"] < self.p.minpk)),
                   ("not integrated (no valid pixels / failed fit)", ~(np.isfinite(refl["I"]) & (refl["sig"] > 0))),
                   ("near-coincident overlap (rho > 0.8)", np.nan_to_num(refl.get("rho", np.zeros(len(ok)))) > 0.8),
                   ("ice rings / EXCLUDE_RESOLUTION_RANGE", excl)]
        n = len(ok)
        self.log(f"  SCALE: predicted observations {n}, used {ok.sum()} ({100 * ok.mean():.1f}%); unique reflections "
                 f"{kept_u.size}, with no usable observation {int((~kept_u).sum())} ({100 * (~kept_u).mean():.1f}%)")
        for lab, m in reasons:
            if m.any():
                lost_u = np.setdiff1d(np.unique(uid[m]), np.unique(uid[ok])).size
                self.log(f"     {lab:46s} {int(m.sum()):8d} obs ({100 * m.mean():5.1f}%), {lost_u:6d} unique lost entirely")

    def _validate_ice_rings(self, refl, ok):
        """Keep a detected ice-ring band only if its reflections are measurably worse than those in the flanking
        resolution shells: scatter against symmetry mates (median |I - <I>| / sigma; ice spots, uneven ring
        background) or a biased intensity level (median I/sigma relative to the interpolated flanks). The
        background template already models ring-shaped background, so weak rings need no exclusion.
        XPG_ICE_VALIDATE=FALSE excludes every detected ring."""
        rings = list(self.st.results.get("ice_rings") or [])
        if not rings:
            self.st.results["ice_rings_excluded"] = []
            return []
        if not getattr(self.p, "ice_validate", True):
            self.st.results["ice_rings_excluded"] = rings
            return rings
        sg = self.st.space_group or gemmi.SpaceGroup("P 1")
        I, sig = refl["I"], refl["sig"]
        s = 1.0 / d_spacing_from_exp(refl["hkl"], self.st.exp)
        hasu, _, plus = to_asu(refl["hkl"], sg)
        from .merging import unique_ids
        uid, _ = unique_ids(hasu, plus, anomalous=False)
        okv = ok & np.isfinite(I) & (sig > 0)
        n = np.bincount(uid[okv], minlength=uid.max() + 1)
        m = np.bincount(uid[okv], I[okv], minlength=uid.max() + 1) / np.maximum(n, 1)
        dev = np.abs(I - m[uid]) / sig * np.sqrt(n[uid] / np.maximum(n[uid] - 1, 1))
        multi = okv & (n[uid] >= 2)
        keep_out = []

        ice_any = np.zeros(len(s), bool)
        for lo_, hi_ in rings:
            ice_any |= (s >= 1.0 / hi_ - 0.002) & (s <= 1.0 / lo_ + 0.002)

        def band_stats(s_lo, s_hi):
            """(scatter ratio, I/sigma level ratio) of a shell vs its flanking shells (flanks never on an ice ring),
            None if too few."""
            w = s_hi - s_lo
            band = multi & (s >= s_lo) & (s <= s_hi)
            fl = multi & ~ice_any & (((s >= s_lo - 4 * w - 0.004) & (s < s_lo - 0.002)) |
                                     ((s > s_hi + 0.002) & (s <= s_hi + 4 * w + 0.004)))
            if band.sum() < 30 or fl.sum() < 100:
                return None
            scat = np.median(dev[band]) / max(np.median(dev[fl]), 1e-6)
            lev_fl = np.median(I[fl] / sig[fl])
            lev = np.median(I[band] / sig[band]) / lev_fl if abs(lev_fl) > 0.2 else 1.0
            return scat, lev, int(band.sum())
        ice_s = [(1.0 / hi_, 1.0 / lo_) for lo_, hi_ in rings]
        on_ice = lambda a_, b_: any(not (b_ < s0_ - 0.004 or a_ > s1_ + 0.004) for s0_, s1_ in ice_s)
        for lo, hi in rings:
            s_lo, s_hi = 1.0 / hi, 1.0 / lo
            st_ = band_stats(s_lo, s_hi)
            if st_ is None:
                keep_out.append((lo, hi))
                continue
            scat, lev, nb_ = st_
            # empirical null: identical shells at nearby ice-free resolutions, each against its own flanks; captures
            # how much the statistics vary from shell to shell without ice (resolution trends, sampling, systematics)
            w = s_hi - s_lo
            ctrl = []
            for c0 in np.arange(s_lo - 0.06, s_hi + 0.06, w):
                if on_ice(c0, c0 + w):
                    continue
                cs = band_stats(c0, c0 + w)
                if cs is not None:
                    ctrl.append(cs[:2])
            if len(ctrl) >= 8:
                ctrl = np.array(ctrl)
                thr_s = np.quantile(ctrl[:, 0], 0.95)
                lv = np.abs(np.log(np.maximum(ctrl[:, 1], 1e-3)))
                thr_l = np.quantile(lv, 0.95)
                bad = (scat > max(thr_s, 1.0)) or (abs(np.log(max(lev, 1e-3))) > thr_l)
                note = f"control shells: scatter 95% {thr_s:.2f}, level 95% x{np.exp(thr_l):.2f} (n {len(ctrl)})"
            else:
                bad = scat > 1.25 or not (0.75 < lev < 1.33)
                note = "too few control shells: fixed limits"
            self.log(f"  ICE: ring {lo:.3f}-{hi:.3f} A: scatter vs flanks x{scat:.2f}, I/sigma level x{lev:.2f} ({nb_} obs; "
                     f"{note}) -> {'excluded' if bad else 'kept (not worse than ice-free shells)'}")
            if bad:
                keep_out.append((lo, hi))
        if self.st.space_group is not None:          # final decision once the symmetry (mates) is known
            self.st.results["ice_rings_excluded"] = keep_out
        return keep_out

    def job_scale(self, final=False):
        self.log("\n=== SCALE ===" if not final else "\n=== SCALE (final) ===")
        p = self.p
        self.st.results.pop("ice_rings_excluded", None)   # decided anew from this run's data
        refl = self.st.refl
        exp = self.st.exp
        g = exp.geom
        sel = self._select_observations(refl, report=True)
        r = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape else v) for k, v in refl.items()}
        s = 1.0 / d_spacing_from_exp(r["hkl"], exp)
        # ---- symmetry -------------------------------------------------
        if self.st.lattice is None:
            cands = lattice_candidates(exp.crystal.cell, max_obliq=p.lattice_tolerance)
            cand, sg = self._choose_symmetry(r, cands, s)
            self._apply_lattice(cand, sg, refl)       # also reindexes refl (and REIDX)
            r["hkl"] = refl["hkl"][sel]
            exp = self.st.exp
        sg = self.st.space_group
        cand = self.st.lattice
        # ---- REFERENCE_DATA_SET: resolve indexing ambiguity against a reference data set ----
        if p.reference_data_set and not self.st.results.get("reference_applied"):
            from .analysis import best_reindex, read_reference
            ref_path = p.reference_data_set if Path(p.reference_data_set).is_absolute() else \
                str(Path(p.workdir) / p.reference_data_set)
            rk, rI = read_reference(ref_path, sg)
            R, ccs = best_reindex(r["hkl"], r["I"], r["sig"], rk, rI, exp.crystal.cell, sg)
            self.log("  SCALE: REFERENCE_DATA_SET correlations per indexing operator: " +
                     ", ".join(f"{c:.3f} (n={n})" for c, n in ccs))
            if not np.array_equal(R, np.eye(3, dtype=int)):
                refl["hkl"] = apply_rot(refl["hkl"], R)
                r["hkl"] = apply_rot(r["hkl"], R)
                Ast = exp.crystal.Astar @ np.linalg.inv(R.T)
                exp = Experiment(exp.geom.copy(), Crystal.from_Astar(Ast, exp.crystal.system, space_group=sg.xhm()))
                self.st.exp = exp
                self.log(f"  SCALE: reindexed to match reference with h' = h R, R = {R.tolist()}")
            self.st.results["reference_applied"] = True
        # ---- post-refinement on integrated centroids (GXPARM) --------
        # strong reflections for post-refinement: MINIMUM_I/SIGMA (default 3), fully recorded
        strong = (r["I"] / r["sig"] > p.minimum_i_sigma) & np.isfinite(r["xo"]) & (r["peak"] > 90)
        self._prediction_error(r, strong, exp, final)
        self._cell_trajectory(r, strong, exp)
        if strong.sum() > 100:
            rr = refine(exp, r["hkl"][strong], r["xo"][strong], r["yo"][strong], r["zo"][strong], p.refine_correct,
                        sigma_xy=0.3, max_iter=25)
            self.log(f"  SCALE: post-refinement on {rr.n_used} strong reflections: rmsd {rr.rmsd_xy:.3f} px, "
                     f"{rr.rmsd_phi:.4f} deg; cell {np.array2string(rr.exp.crystal.cell, precision=3)}")
            exp = rr.exp
            self.st.exp = exp
            write_geometry(self.wd / "xpg_geometry.json", exp, sg.number, p.data_range, p.friedels_law)
            if p.export_legacy:
                write_xparm(self.wd / "GXPARM.XDS", exp, sg.number)
        # ---- scaling --------------------------------------------------
        hasu, isym, plus = to_asu(r["hkl"], sg)
        from .merging import unique_ids
        # FRIEDEL'S_LAW=FALSE: Bijvoet mates are different reflections for scaling, error model and
        # outlier rejection; otherwise the anomalous signal is mistaken for error
        # STRICT_ABSORPTION_CORRECTION=FALSE given explicitly: mates merged for scaling
        strict = (not p.friedels_law) if p.strict_absorption is None else bool(p.strict_absorption)
        uid, _ = unique_ids(hasu, plus, anomalous=strict)
        if "Isum" in r and p.combine_intensities:
            r["I"], r["sig"] = self._combine_prf_sum(r, uid)
        from .scaling import inconsistent_regions
        # judged per ~20 degree rotation block (rotation-dependent shadows), falling back to the whole sweep
        nblk = max(1, int(round(20.0 / max(abs(g.osc), 1e-6))))
        blkid = (np.floor(r["z"]).astype(int) - (self.p.data_range[0] - 1)) // nblk
        badreg, rows = inconsistent_regions(r["I"], r["sig"], uid, r["x"], r["y"], g.nx, g.ny, block=blkid)
        if __import__("os").environ.get("XPG_WEAK_REGIONS", "1").upper() in ("0", "F", "FALSE", "N", "NO", "OFF"):
            badreg = np.zeros_like(badreg)
        if badreg.any():
            self.log(f"  SCALE WARNING: {len(rows)} detector region(s) give systematically weak intensities "
                     f"(median ratio to equivalents < 0.5): {badreg.sum()} observations excluded. Unmasked shadow "
                     f"or wrong detector/segment geometry?")
            for ix, iy, n, rt, bk in rows[:12]:
                f0 = self.p.data_range[0] + bk * nblk
                self.log(f"     region x {ix * g.nx // 16}-{(ix + 1) * g.nx // 16}, y {iy * g.ny // 16}-{(iy + 1) * g.ny // 16}, "
                         f"frames {f0}-{f0 + nblk - 1}: {n} strong obs, ratio {rt:.2f}")
            if len(rows) > 12:
                self.log(f"     ... {len(rows) - 12} more")
            keep = ~badreg
            r = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in r.items()}
            s, hasu, isym, plus, uid = s[keep], hasu[keep], isym[keep], plus[keep], uid[keep]
            _, uid = np.unique(uid, return_inverse=True)
        phi = g.z_to_phi(r["z"])
        R = rotation_matrices_np(g.axis, -np.radians(phi))
        s1 = r["S"] / np.linalg.norm(r["S"], axis=1, keepdims=True)
        s1c = np.einsum("nij,nj->ni", R, s1) @ exp.crystal.U       # crystal frame: U^T R(-phi) s1
        s2 = s ** 2
        from .symops import reflection_props
        cen_o, eps_o, _ = reflection_props(hasu, sg)
        sc_phi0 = float(phi.min())
        sc = Scaler(r["I"], r["sig"], uid, phi, s2, s1c, r["x"], r["y"], g.nx, g.ny, corrections=p.corrections,
                    nsig=6.0 * p.wfac1 / 1.0, snrc=p.snrc, nbatch=p.nbatch, eps=eps_o, centric=cen_o)
        sc.run(log=self.log)
        if p.prune_image_groups:
            self._prune_image_groups(sc, r, uid)
        if sc.batch_error:
            fb = sc.batch_f
            hi = np.flatnonzero(fb > 1.3)
            self.log(f"  SCALE: per-batch error factors (5 deg batches) {fb.min():.2f}-{fb.max():.2f}; {hi.size} batch(es) "
                     f"down-weighted > 1.3"
                     + (f" (phi {sc_phi0 + 5 * hi.min():.0f}-{sc_phi0 + 5 * (hi.max() + 1):.0f} deg)" if hi.size else ""))
        if p.fixed_scale_range:
            f0, f1, fac = p.fixed_scale_range
            frame = np.floor(r["z"]).astype(int) + 1
            sc.fix_scale((frame >= f0) & (frame <= f1), fac)
            self.log(f"  SCALE: scale fixed to {fac:g} for frames {f0}-{f1} (DATA_RANGE_FIXED_SCALE_FACTOR)")
        Is, ss, act = sc.scaled()
        if p.zero_dose:
            Is, ss = self._zero_dose(Is, ss, act, uid, r["z"], s)
        try:
            from .io.corrimages import write_correction_images
            write_correction_images(self.wd, sc, g, exp.crystal.U, smax2=float(s2.max()))
        except Exception as e:  # pragma: no cover
            self.log(f"  SCALE: correction images not written ({e})")
        self.st.results["scaler_summary"] = sc.model_summary()
        self.st.results["isa"] = sc.isa()
        # ---- statistics ----------------------------------------------
        cell = exp.crystal.cell
        shells, overall, merged = merging_statistics(hasu[act], plus[act], Is[act], ss[act], cell, sg, nbins=10,
                                                     anomalous=True)
        self.log(f"\n  SCALE: space group {sg.xhm()} (#{sg.number}), cell {np.array2string(cell, precision=3)}")
        isa_x = sc.isa_two_param()
        self.st.results["isa_two_param"] = isa_x
        from .merging import isa_empirical, unique_ids as _uids
        uid_pm, _ = _uids(hasu[act], plus[act], anomalous=not p.friedels_law)
        isa_e = isa_empirical(uid_pm, Is[act])
        self.st.results["isa_empirical"] = isa_e
        # ISa (two-parameter error model, Diederichs 2010); the applied error model may have SdB > 0
        self.st.results["isa"] = isa_x
        self.log(f"  SCALE: ISa = {isa_x:.1f} (two-parameter error model a, b); error model SdFac {sc.sd[0]:.3f} SdB {sc.sd[1]:.3f} "
                 f"SdAdd {sc.sd[2]:.4f} ({_isa_txt(sc.isa())}); ISa (empirical, strong-reflection scatter) = {isa_e:.1f}")
        self.log("\n" + format_stats_table(shells, overall, anomalous=True) + "\n")
        fine, _, _ = merging_statistics(hasu[act], plus[act], Is[act], ss[act], cell, sg, nbins=30, anomalous=False)
        dcut = cc_half_cutoff(fine, 0.3)
        # errors shared by repeated measurements inflate CC1/2 (random half-datasets put repeats of a reflection in
        # both halves): then the limit comes from half-datasets that split whole P1 reflections (symmetry mates)
        self._shared_error_check(r["hkl"][act], Is[act], r["z"][act], cell, sg)
        worst = self.st.results.get("shared_error_worst")
        if worst is not None and worst < 0.6:
            h_ = np.asarray(r["hkl"][act], np.int64)
            neg = (h_[:, 0] < 0) | ((h_[:, 0] == 0) & (h_[:, 1] < 0)) | ((h_[:, 0] == 0) & (h_[:, 1] == 0) & (h_[:, 2] < 0))
            from .symops import hkl_key as _hk
            fine_m, _, _ = merging_statistics(hasu[act], plus[act], Is[act], ss[act], cell, sg, nbins=30, anomalous=False,
                                              split_key=_hk(np.where(neg[:, None], -h_, h_)))
            dcut_m = cc_half_cutoff(fine_m, 0.3)
            if dcut_m:
                self.log(f"  SCALE: CC1/2 from half-datasets of whole symmetry mates (immune to the shared errors): "
                         f"CC1/2 = 0.3 at {dcut_m:.2f} A (random half-datasets: {dcut if dcut else float('nan'):.2f} A)")
                fine, dcut = fine_m, dcut_m
        self.events.emit("metric", name="shells", title="Resolution shells", xlabel="1/d^2 (A^-2)",
                         x=[float(1 / (0.5 * (sh.dmax + sh.dmin)) ** 2) for sh in fine],
                         series={"CC1/2": [float(sh.cc_half) for sh in fine],
                                 "<I/sigma>": [float(sh.i_over_sigma) for sh in fine],
                                 "completeness (%)": [float(sh.completeness) * (100.0 if max((x.completeness for x in fine),
                                                      default=0) <= 1.0 else 1.0) for sh in fine]})
        # statistics inside the chosen limit (the summary reports these; the table above runs to the
        # detector edge)
        if dcut:
            sh_c, ov_c, _ = merging_statistics(hasu[act], plus[act], Is[act], ss[act], cell, sg, nbins=10, dmin=dcut,
                                               anomalous=True)
            self.st.results.update(dict(shells_cut=sh_c, overall_cut=ov_c))
        self.st.results.update(dict(shells=shells, overall=overall, cc_half_cutoff=dcut, cc_fine=fine,
                                    n_obs_merged=int(act.sum())))
        if dcut:
            self.log(f"  SCALE: isotropic resolution estimate (CC1/2 = 0.3 fit): {dcut:.2f} A")
            self.decide("resolution limit", f"{dcut:.2f} A (CC1/2 = 0.3 fit)")
        from .analysis import batch_statistics, twinning_analysis
        try:
            tw = twinning_analysis(merged["hkl"], merged["I"], merged["sig"], cell, sg,
                                   tncs=self.st.results.get("tncs_modulation"))
            self.log("\n  SCALE: Wilson and twinning statistics")
            for line in tw.lines():
                self.log("  " + line)
            self.st.results["twinning"] = tw
            # twinning seen by the L-test but not explained by any twin law of this space group: the apparent symmetry
            # may include twin operators
            from .analysis import hidden_twinning_check, l_test_alpha
            self._hidden_twin = None
            ht = hidden_twinning_check(r["hkl"][act], Is[act], ss[act], cell, sg, tw)
            best_law = max([0.5 * (t[2] + t[4]) for t in tw.twin_laws], default=0.0)
            ht = [t for t in ht if 0.5 * (t[2] + t[3]) >= best_law + 0.1]   # clearly beyond the laws of sg itself
            if ht:
                a_l = l_test_alpha(tw.l_mean)
                self.log(f"  TWINNING WARNING: L-test twin fraction {a_l:.2f} is not explained by the twin laws of "
                         f"{sg.xhm()}; elements of the apparent symmetry tested as twin laws in its subgroups:")
                for sub, op, a_h, a_b in ht[:6]:
                    self.log(f"     {sub.xhm():12s} twin law {op:14s}: alpha {a_h:.2f} (H-test), {a_b:.2f} (Britton)")
                # near-perfect twinning makes the subgroups indistinguishable: a twin law of alpha ~0.5 relates
                # intensities as exactly as symmetry -> every subgroup within 0.05 of the best is reported, written
                # and passed to MR (xpg mr) -- only the structure decides
                a0 = 0.5 * (ht[0][2] + ht[0][3])
                if a0 >= 0.2:
                    tied, seen = [], set()
                    for sub, op, a_h, a_b in ht:
                        if 0.5 * (a_h + a_b) >= a0 - 0.05 and sub.xhm() not in seen:
                            seen.add(sub.xhm())
                            tied.append((sub, op, a_h, a_b))
                    names = ", ".join(f"{t[0].xhm()} (twin law {t[1]}, alpha ~{0.5 * (t[2] + t[3]):.2f})" for t in tied)
                    self.log(f"  TWINNING WARNING: the true space group may be {names}"
                             f"{' -- not distinguishable from the intensities (near-perfect twinning); try each in MR' if len(tied) > 1 else ''}"
                             f"; use it for MR / refinement (twin refinement)")
                    self._hidden_twin = tied
                    self.st.results["hidden_twin"] = [dict(space_group=t[0].xhm(), law=t[1], alpha_h=t[2],
                                                           alpha_britton=t[3], alpha_l=a_l) for t in tied]
            from .analysis import tncs_patterson
            ratio, vec, dist = tncs_patterson(merged["hkl"], merged["I"], cell, sg)
            if np.isfinite(ratio):
                flag = "  <-- tNCS likely: weak reflection classes may mimic absences / bias twin tests" if ratio > 0.2 else ""
                self.log(f"    native Patterson: largest off-origin peak {100 * ratio:.1f}% of origin at "
                         f"({vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f}), {dist:.1f} A{flag}")
                self.st.results["tncs"] = dict(ratio=ratio, vector=vec.tolist(), distance=dist)
        except Exception as e:
            self.log(f"  SCALE: twinning analysis failed ({e})")
        rows = batch_statistics(r["z"][act], uid[act], Is[act], ss[act], sc.scales()[act],
                                block_frames=max(1, int(round(5.0 / abs(g.osc)))))
        self.log("\n  SCALE: statistics per 5-degree batch (radiation damage diagnostics)")
        self.log("     frames        Nobs   <I/sig>   Rmeas   scale")
        for a_, b_, n_, isg, rm, scl in rows:
            self.log(f"   {a_:5d}-{b_:5d} {n_:8d} {isg:9.1f} {100 * rm:6.1f}% {scl:7.3f}")
        self.st.results["batch_stats"] = rows
        if rows:
            self.events.emit("metric", name="batch_stats", title="Per-batch statistics", xlabel="frame",
                             x=[float(a_) for a_, *_r in rows],
                             series={"<I/sigma>": [float(x[3]) for x in rows], "Rmeas (%)": [100 * float(x[4]) for x in rows],
                                     "scale": [float(x[5]) for x in rows]})
        # ---- outputs ----------------------------------------------------
        self._main_hkl, self._main_I, self._main_sig = r["hkl"][act], Is[act], ss[act]
        # rejected misfits are written with negative sigma (legacy format); observations removed for other reasons (excluded frames,
        # shadowed detector regions) are not written not write them: runs of batches holding only
        # flagged observations break the initial scaling of other programs
        fr = np.floor(r["z"]).astype(int)
        live = np.bincount(fr - fr.min(), act.astype(int))[fr - fr.min()] > 0     # frame still has accepted data
        out = act | (sc.misfit & live)
        sig_out = np.where(act, ss, -ss)[out]
        if p.export_legacy:
            write_legacy_hkl(self.wd / "XDS_ASCII.HKL", exp, sg.number, r["hkl"][out], Is[out], sig_out, r["x"][out],
                            r["y"][out], r["z"][out], r["rlp"][out], r["peak"][out], r["corr"][out], friedel=p.friedels_law,
                            data_range=p.data_range,
                            sigma_m=self.st.model.sigma_m, sigma_d=self.st.model.sigma_d, template=p.name_template,
                            dmin=float(1 / s.max()), variance_model=(sc.sd[0] ** 2, sc.sd[2] ** 2))
        for sub_t in (getattr(self, "_hidden_twin", None) or []):
            if not sub_t[0].is_reference_setting():
                continue
            # same observations in the subgroup (same axes and indices): input for MR / twin refinement
            fn = f"xpg_unmerged_{sub_t[0].xhm().replace(' ', '')}.mtz"
            self._write_unmerged(self.wd / fn, exp, sub_t[0], r["hkl"][act], Is[act], ss[act], r["x"][act], r["y"][act],
                                 r["z"][act], r["rlp"][act], geometry=False)
            self.log(f"  TWINNING: data in {sub_t[0].xhm()} written to {fn}")
        self._write_unmerged(self.wd / "xpg_unmerged.mtz", exp, sg, r["hkl"][act], Is[act], ss[act], r["x"][act],
                             r["y"][act], r["z"][act], r["rlp"][act], geometry=True)
        self._write_merged(merged, cell, sg, dcut, g.wavelength, final)
        self._aniso_statistics(hasu[act], plus[act], Is[act], ss[act], cell, sg)
        if os.environ.get("XPG_BAYES_MERGE", "0") == "1":
            self._write_bayes_merged(hasu[act], Is[act], ss[act], cell, sg, dcut, g.wavelength)
        self.st.results["merged"] = merged
        if self.st.results.get("extra_refl") and (final or not self.second_pass):
            self._process_extra_lattices(merged, cell, sg)
        elif self.st.results.get("extra_refl") and self.second_pass and LATTICE_TRIAGE:
            self._triage_extra_lattices(merged, cell, sg)
        n_rf, t_rf = getattr(self, "_refit_stats", (0, 0.0))
        if n_rf:
            self.log(f"  SCALE: {n_rf} scale refits after frame filters, {t_rf:.1f}s ({FILTER_CYCLES} cycle(s) each)")
            self._refit_stats = (0, 0.0)
        self._write_report(final)

    def _shared_error_check(self, hkl, I, z, cell, sg, min_cc=0.3, min_ratio=0.6):
        """Symmetry mates vs repeated measurements of the same reflection (analysis.repeat_symmetry_cc): where the
        data still carry signal (CC_rep >= min_cc), true symmetry with independent errors keeps CC_sym / CC_rep
        near 1; a lower ratio means errors shared by the repeats, which also inflate CC1/2 and the resolution
        limit."""
        from .analysis import repeat_symmetry_cc
        if len(sg.operations().sym_ops) < 2:      # P1: no symmetry mates
            return
        rows = repeat_symmetry_cc(hkl, I, z, cell, sg)
        if not rows:
            return
        self.events.emit("metric", name="shared_error", title="Repeats vs symmetry mates (sweep halves)",
                         xlabel="1/d^2 (A^-2)", x=[float(0.5 / r_[0] ** 2 + 0.5 / r_[1] ** 2) for r_ in rows],
                         series={"CC repeats": [r_[2] for r_ in rows], "CC symmetry mates": [r_[3] for r_ in rows]})
        self.st.results["repeat_symmetry_cc"] = rows
        sig_rows = [r_ for r_ in rows if np.isfinite(r_[2]) and np.isfinite(r_[3]) and r_[2] >= min_cc]
        self.st.results["shared_error_worst"] = min((r_[3] / r_[2] for r_ in sig_rows), default=None)
        bad = [r_ for r_ in sig_rows if r_[3] < min_ratio * r_[2]]
        self.log("  SCALE: repeats vs symmetry mates (sweep halves, CC_sym / CC_rep): " +
                 ", ".join(f"{r_[1]:.2f} A {r_[3] / r_[2]:.2f}" for r_ in rows if np.isfinite(r_[2]) and r_[2] > 0))
        if bad:
            worst = min(bad, key=lambda r_: r_[3] / r_[2])
            self.log(f"  SCALE WARNING: symmetry mates agree less than repeated measurements of the same reflection "
                     f"({worst[0]:.2f}-{worst[1]:.2f} A: CC {worst[3]:.2f} vs {worst[2]:.2f}): errors shared by the "
                     f"repeats (overlapping spots in crowded / mosaic data, spot widths, background) inflate CC1/2 "
                     f"and the resolution limit; or the symmetry is too high")

    def _zero_dose(self, I, sig, act, uid, z, s, nbins=10, min_obs=4):
        """Zero-dose extrapolation per reflection (Diederichs 2006) with empirical-Bayes shrinkage instead of a
        significance threshold. For each unique reflection a weighted line I = a + b D (D: dose ~ frame number) gives
        the relative slope r = b / <I> with error s_r; the variance of true slopes per resolution bin is
        tau^2 = max(var(r) - <s_r^2>, 0); slopes are shrunk by tau^2 / (tau^2 + s_r^2), so noisy slopes contribute
        nothing and only consistent dose trends (specific radiation damage) are removed. Applied per observation:
        I' = I - r* <I> D (extrapolated to the start of the exposure)."""
        D = (z - z.min()) / max(z.max() - z.min(), 1.0)
        a_ = act & np.isfinite(I) & (sig > 0)
        n = uid.max() + 1
        w = np.where(a_, 1.0 / sig ** 2, 0.0)
        S = np.bincount(uid, w, n)
        Sx = np.bincount(uid, w * D, n)
        Sy = np.bincount(uid, w * I, n)
        Sxx = np.bincount(uid, w * D * D, n)
        Sxy = np.bincount(uid, w * D * I, n)
        cnt = np.bincount(uid, a_.astype(int), n)
        det = S * Sxx - Sx ** 2
        ok = (cnt >= min_obs) & (det > 1e-9 * np.maximum(S, 1e-30) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (S * Sxy - Sx * Sy) / det
            sb = np.sqrt(S / det)
            mean = Sy / S
        good = ok & np.isfinite(b) & np.isfinite(sb) & (mean > 3.0 / np.sqrt(np.maximum(S, 1e-30)))
        if good.sum() < 200:
            self.log("  SCALE: zero-dose extrapolation skipped (too few reflections with enough observations)")
            return I, sig
        r = np.where(good, b / mean, 0.0)
        sr = np.where(good, sb / np.abs(mean), np.inf)
        su = np.zeros(n)
        su[uid] = s
        edges = np.quantile(su[good], np.linspace(0, 1, nbins + 1))
        rb = np.clip(np.searchsorted(edges, su, side="right") - 1, 0, nbins - 1)
        shrink = np.zeros(n)
        taus = []
        for k in range(nbins):
            m = good & (rb == k)
            if m.sum() < 30:
                continue
            # robust variance of the slopes (MAD) minus their mean sampling variance
            v = (1.4826 * np.median(np.abs(r[m] - np.median(r[m])))) ** 2
            tau2 = max(v - np.median(sr[m] ** 2), 0.0)
            taus.append(np.sqrt(tau2))
            shrink[m] = tau2 / (tau2 + sr[m] ** 2)
        rs = r * shrink
        corr = rs[uid] * mean[uid] * D
        corr = np.where(a_ & good[uid], corr, 0.0)
        I2 = I - corr
        sig2 = np.sqrt(sig ** 2 + (np.where(good, shrink * sb, 0.0)[uid] * D) ** 2)
        eff = good & (shrink > 0.5)
        self.log(f"  SCALE: zero-dose extrapolation (empirical Bayes): true relative slope spread per resolution bin "
                 f"{min(taus) if taus else 0:.3f}-{max(taus) if taus else 0:.3f}; {int(eff.sum())} of {int(good.sum())} "
                 f"reflections corrected by > half their slope; median |correction| "
                 f"{100 * np.median(np.abs(corr[a_ & good[uid]] / np.maximum(np.abs(I[a_ & good[uid]]), 1e-9))):.1f}%")
        return I2, sig2

    def _combine_prf_sum(self, r, uid, block_frames=100):
        """Combination of profile-fitted and summation intensities,
        I = w I_prf + (1 - w) I_sum with w = 1 / (1 + I_sum / I_mid): profile fitting for weak reflections
        (lower noise), summation for strong ones (insensitive to profile-model errors). I_mid is chosen by
        the lowest Rmeas of symmetry mates within blocks of frames (scale-insensitive, before scaling)."""
        # summation misses the unrecorded part of partially recorded reflections (profile fitting extrapolates
        # it): scale by the recorded profile fraction (PEAK)
        # summation counts only the foreground: divide by the fraction of the reflection inside it (from the learnt
        # profile, so broad / split spots are not biased low); older integrations: recorded fraction (PEAK)
        rec = np.clip(r["peak"] / 100.0, 0.5, 1.0)

        def fg_frac(key):
            """recorded fraction (PEAK, per reflection) x foreground shape fraction (median over reflections of the
            same detector region, 100-frame block and |zeta| bin: a smooth property of the spot shape; the
            per-reflection estimate carries ~1-2% profile/interpolation noise)."""
            if key not in r or np.isfinite(r[key]).mean() < 0.9:
                return rec
            v = np.clip(np.nan_to_num(r[key], nan=1.0), 0.5, 1.0)
            g = self.st.exp.geom
            reg = (np.clip((r["x"] / g.nx * 3).astype(int), 0, 2) * 3 + np.clip((r["y"] / g.ny * 3).astype(int), 0, 2))
            zk = np.abs(r.get("zeta_k", np.ones(len(v))))
            zb = np.clip(np.searchsorted(np.quantile(zk, [0.2, 0.4, 0.6, 0.8]), zk), 0, 4)
            key_ = (reg * 5 + zb) * 100000 + (r["z"] // 100).astype(int)
            uk_, inv_ = np.unique(key_, return_inverse=True)
            med = np.array([np.median(v[inv_ == k]) for k in range(uk_.size)])
            return np.clip(rec * med[inv_], 0.5, 1.0)
        frac = fg_frac("p_fg")
        if "p_fg" in r:
            q = lambda k: np.nanpercentile(r[k], [5, 50, 95]) if k in r and np.isfinite(r[k]).any() else [np.nan] * 3
            a1 = q("p_fg")
            self.log(f"  SCALE: profile fraction inside the summation foreground {a1[0]:.3f}/{a1[1]:.3f}/{a1[2]:.3f} "
                     f"(5/50/95%)")
        Ip, sp, Is_, ss_ = r["I"], r["sig"], r["Isum"] / frac, r["sigsum"] / frac
        okc = np.isfinite(Is_) & (ss_ > 0)
        if "sum_ok" in r:                 # forward-model mode: no summation for reflections with overlap partners
            okc &= np.asarray(r["sum_ok"], bool)
        key = uid.astype(np.int64) * 10000 + (r["z"] // block_frames).astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        n = np.bincount(inv)

        def rmeas(I):
            m = np.bincount(inv, I) / n
            mm = n[inv] >= 2
            dev = np.abs(I - m[inv]) * np.sqrt(n[inv] / np.maximum(n[inv] - 1, 1))
            return dev[mm].sum() / max(np.abs(I[mm]).sum(), 1e-30)
        pos = np.abs(Is_[okc])
        cands = [np.inf] + list(np.quantile(pos, [0.5, 0.75, 0.9, 0.97, 0.99]))
        # consistency of the profile model: for strong, fully recorded, unobstructed reflections both estimates
        # measure the same counts, so a systematic difference means the profile model does not describe the spots
        # (high mosaicity, mis-centred profiles, split spots). Summation is model-free: then profile fitting alone is
        # not allowed for strong reflections
        strong = okc & (Is_ / np.maximum(ss_, 1e-9) > 20) & (r["peak"] > 95) & (Ip > 0) & (r["n_over"] == 0)
        if strong.sum() > 100:
            lr = float(np.median(np.log(np.maximum(Is_[strong], 1e-9) / Ip[strong])))
            self.log(f"  SCALE: strong fully recorded reflections: median I_sum/I_prf = {np.exp(lr):.3f}")
            if abs(np.expm1(lr)) > 0.1:          # ratio more than 10% from 1
                cands = cands[1:]
                self.log("  SCALE: profile model inconsistent with summation (> 10%): profile fitting alone "
                         "not used for strong reflections")
        best = None
        for imid in cands:
            w = np.where(okc, 1.0 / (1.0 + np.maximum(Is_, 0) / imid), 1.0)
            I = w * Ip + (1 - w) * np.where(okc, Is_, 0.0)
            rm = rmeas(I)
            if best is None or rm < best[0] - 1e-5:
                best = (rm, imid, w)
        rm0 = rmeas(Ip)
        rm, imid, w = best
        self.log(f"  SCALE: profile-fitted / summation combination: I_mid = "
                 f"{'inf (profile fitting only)' if not np.isfinite(imid) else f'{imid:.0f}'}; "
                 f"block Rmeas {100 * rm0:.2f}% (profile) -> {100 * rm:.2f}%")
        I = w * Ip + (1 - w) * np.where(okc, Is_, 0.0)
        sig = np.sqrt((w * sp) ** 2 + ((1 - w) * np.where(okc, ss_, 0.0)) ** 2)
        # The weak class is corrected (TNCS_MODE) when its strong reflections disagree with the profile model by more
        # than TNCS_MIN_RATIO relative to the other class (and > 5 sd).
        tn = self.st.results.get("tncs_modulation")
        if tn is not None and TNCS_MODE != "off":
            # the Patterson peak position carries ~0.003-0.01 of error, which turns into a wrong class at |h| ~ 100: a
            # translation within 0.015 of a fraction k / N (N <= 6) is that fraction
            t = np.asarray(tn[0], float)
            for N in (2, 3, 4, 5, 6):
                if np.max(np.abs(t - np.round(t * N) / N)) < 0.015:
                    t = np.round(t * N) / N
                    break
            low = np.cos(2 * np.pi * (r["hkl"] @ t)) < 0
            st_ = okc & (Is_ / np.maximum(ss_, 1e-9) > 10) & (Ip > 0) & (r["n_over"] == 0)
            a, b = st_ & low, st_ & ~low
            if a.sum() >= 200 and b.sum() >= 200:
                la, lb = np.log(Is_[a] / Ip[a]), np.log(Is_[b] / Ip[b])
                mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
                d = float(np.median(la) - np.median(lb))
                se = float(1.2533 * np.hypot(mad(la) / np.sqrt(a.size), mad(lb) / np.sqrt(b.size)))
                # a moderate class difference also comes from summation contaminated by the strong neighbours; a large
                # one is the profile bias
                use = d > np.log(TNCS_MIN_RATIO) and d > 5 * se
                self.log(f"  SCALE: tNCS classes (cos 2 pi h.t < 0 vs > 0), strong reflections: I_sum / I_prf "
                         f"{np.exp(np.median(la)):.3f} vs {np.exp(np.median(lb)):.3f} ({d / max(se, 1e-9):.1f} sd) -> "
                         + (("weak class from summation" if TNCS_MODE == "sum" else
                             f"weak-class profile intensities x {np.exp(d):.3f}") + " (its spots differ from the learnt "
                            "profiles)" if use else "profile / summation combination for both"))
                self.decide("tNCS weak-class intensities", (TNCS_MODE if use else "combination"),
                            margin=d - np.log(TNCS_MIN_RATIO),
                            unit=f"log I_sum/I_prf class difference minus log {TNCS_MIN_RATIO:g}",
                            low=abs(d - np.log(TNCS_MIN_RATIO)) < 0.05 or abs(d / max(se, 1e-9) - 5) < 2,
                            evidence=dict(ratio_weak=float(np.exp(np.median(la))),
                                          ratio_strong=float(np.exp(np.median(lb))), n=[int(a.sum()), int(b.sum())]))
                if use and TNCS_MODE == "sum":
                    sel = low & okc
                    I, sig = np.where(sel, Is_, I), np.where(sel, ss_, sig)
                elif use:
                    # the profile fit captures a fixed fraction of the broader spots: their profile intensities are
                    # scaled by the class ratio and combined with summation as before (profile-fit precision for the
                    # weak ones)
                    f = np.exp(d)
                    Ipl, spl = np.where(low, Ip * f, Ip), np.where(low, sp * f, sp)
                    I = w * Ipl + (1 - w) * np.where(okc, Is_, 0.0)
                    sig = np.sqrt((w * spl) ** 2 + ((1 - w) * np.where(okc, ss_, 0.0)) ** 2)
        return I, sig

    def _write_unmerged(self, path, exp, sg, hkl, I, sig, x, y, z, rlp, geometry=True):
        """Unmerged MTZ (standard columns) + its geometry JSON (<stem>_geometry.json; the
        main output uses xpg_geometry.json) so it is self-contained for re-scaling and merging."""
        g = exp.geom
        sgo = sg if isinstance(sg, gemmi.SpaceGroup) else gemmi.SpaceGroup(sg)
        hasu, isym_, _ = to_asu(hkl, sgo)
        batch = np.floor(z).astype(int) + 1
        write_unmerged_mtz(path, sgo, exp.crystal.cell, g.wavelength, hasu, isym_, batch, I, sig, x, y,
                           g.z_to_phi(z), rlp, batch_phi=lambda b: (float(g.z_to_phi(b - 1)), float(g.z_to_phi(b))))
        if geometry:
            write_geometry(Path(path).with_name(Path(path).stem.replace("_unmerged", "") + "_geometry.json"), exp,
                           sgo.number, self.p.data_range, self.p.friedels_law)

    def _write_spots(self):
        sp = self.st.spots
        write_spots(self.wd / "xpg_spots.txt", sp.x, sp.y, sp.z, sp.intensity)
        if self.p.export_legacy:
            sp.write_legacy_spots(self.wd / "SPOT.XDS")

    def _plausible_merge_limit(self, n_all):
        """Best CC1/2 = 0.3 limit the all-lattice merge can reach if the extra lattices were independent data of the
        main lattice's quality: k = n_all / n_main times the observations raise CC1/2 per shell to the
        Spearman-Brown value k CC / (1 + (k - 1) CC). A merge claiming more has correlated errors."""
        import dataclasses
        fine, n_main = self.st.results.get("cc_fine"), self.st.results.get("n_obs_merged")
        if not fine or not n_main or not n_all:
            return None
        k = max(n_all / n_main, 1.0)
        boost = [dataclasses.replace(sh, cc_half=k * sh.cc_half / (1 + (k - 1) * sh.cc_half))
                 if np.isfinite(sh.cc_half) and sh.cc_half > 0 else sh for sh in fine]
        return cc_half_cutoff(boost, 0.3)

    def _cell_trajectory(self, r, strong, exp, block_deg=30.0, min_z=4.0, min_index_shift=0.1, min_refl=300):
        """Unit cell along the sweep: orientation + cell refined per ~30-degree block on the centroids of strong
        integrated reflections (detector and beam fixed). A model-free marker of non-isomorphous states: smooth
        drift is radiation damage; a persistent STEP marks a change of state or a crystal slipping. The noise is
        the data set's own block-to-block scatter (robust sd of the steps between adjacent blocks): the formal
        refinement esds are 10-60x too small, and poor data scatter by 0.5-1% per block, clean data by 0.03-0.1%. A
        step is flagged when the mean cell of the 2 blocks after a boundary differs from the 2 before by >
        ``min_z`` noise sd AND shifts reflections at the high-resolution end of the strong reflections by >
        ``min_index_shift`` reciprocal-lattice units (|da| / d; effect floor)."""
        try:
            g = exp.geom
            ok = strong & (r["I"] / r["sig"] > 10) & np.isfinite(r["zo"])
            if "lattice" in r:
                ok &= r["lattice"] == 0
            if ok.sum() < 4 * min_refl:
                return
            z0, z1 = float(np.min(r["zo"][ok])), float(np.max(r["zo"][ok]))
            nfr = max(int(round(block_deg / abs(g.osc))), 1)
            rows = []
            for b0 in np.arange(np.floor(z0), z1, nfr):
                m = ok & (r["zo"] >= b0) & (r["zo"] < b0 + nfr)
                if m.sum() < min_refl:
                    continue
                rr = refine(exp, r["hkl"][m], r["xo"][m], r["yo"][m], r["zo"][m], ["ORIENTATION", "CELL"],
                            max_err_xy=4.0, max_err_phi=2.0, max_iter=15)
                rows.append((int(b0) + 1, int(b0 + nfr), *[float(v) for v in rr.exp.crystal.cell], int(rr.n_used)))
            if len(rows) < 3:
                return
            C = np.array([row[2:5] for row in rows])
            rel = C / np.median(C, axis=0) - 1.0
            self.st.results["cell_trajectory"] = rows
            self.events.emit("metric", name="cell_trajectory", title="Unit cell along the sweep (% from median)",
                             xlabel="frame", x=[0.5 * (row[0] + row[1]) for row in rows],
                             series={ax: [float(100 * v) for v in rel[:, i]] for i, ax in enumerate("abc")})
            # noise: robust sd of a single block's cell from the adjacent-block steps (robust to one real step)
            noise = np.maximum(1.4826 * np.median(np.abs(np.diff(C, axis=0)), axis=0) / np.sqrt(2), 1e-4 * C.mean(0))
            self.log("  SCALE: unit cell along the sweep (per %g-deg block, detector fixed): " % block_deg +
                     "; ".join(f"{row[0]}-{row[1]}: {row[2]:.2f} {row[3]:.2f} {row[4]:.2f}" for row in rows) +
                     "; block scatter " + "/".join(f"{100 * v:.2f}" for v in noise / C.mean(0)) + "%")
            if len(rows) < 4:
                return
            d_hi = float(np.percentile(1.0 / np.maximum(d_spacing_from_exp(r["hkl"][ok], exp), 1e-6), 95)) ** -1
            best = None
            for k in range(1, len(rows) - 2):            # boundary between blocks k and k+1, 2 blocks on each side
                dlt = C[k + 1:k + 3].mean(0) - C[k - 1:k + 1].mean(0)
                z = np.abs(dlt) / noise                   # sd of a difference of two 2-block means = noise
                j = int(np.argmax(z))
                if best is None or z[j] > best[0]:
                    best = (float(z[j]), k, j, float(dlt[j]))
            z, k, j, dlt = best
            shift = abs(dlt) / d_hi
            self.st.results["cell_step"] = dict(z=z, frames=(rows[k][1], rows[k + 1][0]), axis="abc"[j],
                                                change=dlt, index_shift=shift)
            if z > min_z and shift > min_index_shift:
                self.log(f"  SCALE WARNING: unit cell steps by {dlt:+.2f} A ({'abc'[j]}, {100 * dlt / C[:, j].mean():+.2f}%, "
                         f"{z:.1f} x the block scatter, {shift:.2f} index units at {d_hi:.2f} A) between frames "
                         f"{rows[k][1]} and {rows[k + 1][0]}: non-isomorphous states in the sweep (dehydration, "
                         f"damage, crystal slipping); total range {100 * (rel.max(axis=0) - rel.min(axis=0)).max():.2f}%")
            else:
                self.log(f"  SCALE: no significant cell step (largest {z:.1f} x the block scatter, {shift:.2f} index "
                         f"units at {d_hi:.2f} A, {'abc'[j]} at frame {rows[k][1]})")
        except Exception as e:  # pragma: no cover (diagnostic only)
            self.log(f"  SCALE: cell trajectory not computed ({type(e).__name__}: {e})")

    def _prediction_error(self, r, strong, exp, final):
        """How far the integration's predictions are from the observed centroids of strong reflections, in units of
        the spot width (detector, sigma_D footprint) and of the rocking width (phi, sigma_M / zeta). Small values
        mean a second integration pass with post-refined geometry cannot move the shoeboxes / profiles much."""
        try:
            sel = strong & (r["I"] / r["sig"] > 10)
            if sel.sum() < 100 or self.st.model is None:
                return
            g = exp.geom
            Lr = np.linalg.norm(g.pixel_to_lab(r["x"][sel], r["y"][sel]), axis=1)
            spx = np.sqrt((Lr * np.radians(self.st.model.sigma_d) / min(g.qx, g.qy)) ** 2 + 1 / 12)
            zk = np.abs(r["zeta_k"][sel]) if "zeta_k" in r else np.ones(sel.sum())
            sfr = np.sqrt((self.st.model.sigma_m / np.maximum(zk, 0.05) / abs(g.osc)) ** 2 + 1 / 12)
            exy = np.hypot(r["xo"][sel] - r["x"][sel], r["yo"][sel] - r["y"][sel]) / spx
            ez = np.abs(r["zo"][sel] - r["z"][sel]) / sfr
            med_xy, med_z = float(np.median(exy)), float(np.median(ez))
            self.st.results["prediction_error" + ("_final" if final else "_pass1")] = (med_xy, med_z)
            self.log(f"  SCALE: prediction error of strong reflections (median, centroid - prediction): "
                     f"{med_xy:.3f} spot sigma on the detector, {med_z:.3f} rocking sigma in phi")
        except Exception:  # pragma: no cover (diagnostic only)
            pass

    def _choose_symmetry(self, r, cands, s):
        p = self.p
        phi = self.st.exp.geom.z_to_phi(r["z"])
        if p.space_group_number > 0:
            want = gemmi.find_spacegroup_by_number(p.space_group_number)
            pg = want.point_group_hm()
            match = [c for c in cands if gemmi.SpaceGroup(c.space_group).point_group_hm() == pg
                     and c.bravais[1] == want.hm[0]]
            if p.unit_cell is not None and match:
                match.sort(key=lambda c: np.abs(c.cell[:3] - np.array(p.unit_cell[:3])).sum()
                           + np.abs(c.cell[3:] - np.array(p.unit_cell[3:])).sum())
            if match:
                self.log(f"  SCALE: using user-defined space group {want.xhm()}")
                return match[0], want
            self.log(f"  SCALE WARNING: SPACE_GROUP_NUMBER={p.space_group_number} incompatible with lattice; ignoring")
        # reflections on a sharp ice ring carry its excess counts in every repeat (the background annulus lies mostly
        # off the ring) and decorrelate true symmetry mates at those resolutions; the NCS test then reads the deficit
        # as pseudo-symmetry decaying with resolution. Such bands are left out of the symmetry test only.
        bands = ice_spot_bands(s, r["I"], r["sig"], phi, r["hkl"])
        use = np.ones(len(s), bool)
        for lo, hi in bands:
            use &= ~((1.0 / s >= lo) & (1.0 / s <= hi))
        if bands:
            self.log("  SYMMETRY: ice-ring bands left out of the symmetry test (excess of Wilson-implausible "
                     "reflections): " + ", ".join(f"{a:.3f}-{b:.3f}" for a, b in bands) + f" A ({int((~use).sum())} obs)")
            self.st.results["symmetry_ice_bands"] = [list(b) for b in bands]
        scores, opsc, cc_true = score_symmetry(r["hkl"][use], r["I"][use], r["sig"][use], s[use], phi[use], cands,
                                               isig_limit=p.isig_limit_symmetry, log=self.log)
        best = scores[0].cand
        chosen_score = scores[0]
        # power of the test: a pseudo-symmetry element (NCS) correlates at a fraction f ~ 0.8 of a true element;
        # telling the two apart needs 0.2 * CC_true >~ 3 x the systematic floor (0.03) of the CC width, i.e.
        # CC_true >~ 0.45. Below that the likelihood cannot separate symmetry from pseudo-symmetry, and the
        # losses are asymmetric (merging pseudo-equivalent reflections corrupts the data; too low a symmetry only
        # halves the multiplicity): the higher symmetry is then accepted only with probability >= 0.9
        order = lambda c: len(gemmi.SpaceGroup(c.space_group).operations().sym_ops)
        if cc_true < 0.45 and scores[0].prob < 0.9:
            # any lower-symmetry candidate qualifies (not only the 'accepted' ones: the acceptance cut is part of
            # the underpowered test); the most probable of them is used
            lower = [sc for sc in scores if order(sc.cand) < order(best)]
            low = max(lower, key=lambda sc: sc.prob) if lower else scores[0]
            if order(low.cand) < order(best):
                self.log(f"  SYMMETRY WARNING: test underpowered (expected CC for true symmetry {cc_true:.2f} < 0.45) and "
                         f"{best.bravais} {best.point_group} only {scores[0].prob:.2f} probable: using the lower symmetry "
                         f"{low.cand.bravais} {low.cand.point_group} (alternative {best.bravais} {best.point_group} "
                         f"reported; pseudo-symmetry cannot be excluded)")
                self.st.results["symmetry_alternative"] = f"{best.bravais} {best.point_group}"
                best = low.cand
                chosen_score = low
        # space group from absences: merge in the point group (conventional cell)
        hc = r["hkl"] @ best.P
        pg_sg = gemmi.SpaceGroup(best.space_group)
        hasu, _, plus = to_asu(hc, pg_sg)
        from .merging import unique_ids, weighted_merge
        uid, uk = unique_ids(hasu)
        Im, sm, _ = weighted_merge(r["I"], r["sig"], uid, uk.size)
        from .symops import key_hkl
        hk_u = key_hkl(uk)
        from .symops import d_spacing as _dsp, reflection_props as _rp
        _, eps_u, _ = _rp(hk_u, pg_sg)
        dd = _dsp(hk_u, best.cell)
        s2u = 1 / dd ** 2
        edges = np.quantile(s2u, np.linspace(0, 1, 21))
        cen_ = 0.5 * (edges[1:] + edges[:-1])
        mI = np.array([np.mean((Im / eps_u)[(s2u >= edges[i]) & (s2u <= edges[i + 1])]) for i in range(20)])
        mI = np.maximum(mI, 1e-6 * np.abs(mI).max() + 1e-9)
        sigma_of_d = lambda d: np.exp(np.interp(1 / d ** 2, cen_, np.log(mI)))
        # tNCS: modulation of expected intensities, 1 + s cos(2 pi h.t) (cf. Read et al. 2013 tNCS epsilon
        # factors); without it weak parity classes look like absences
        from .analysis import tncs_patterson
        t_ratio, t_vec, t_dist = tncs_patterson(hk_u, Im, best.cell, pg_sg)
        mod = None
        if np.isfinite(t_ratio) and t_ratio > 0.2:
            self.log(f"  SYMMETRY: tNCS peak {100 * t_ratio:.0f}% at {np.round(t_vec, 3).tolist()}: expected intensities "
                     f"modulated by 1 + {min(t_ratio, 0.95):.2f} cos(2 pi h.t) in the absence analysis")
            mod = (np.array(t_vec), min(t_ratio, 0.95))
            self.st.results["tncs_modulation"] = (t_vec.tolist(), float(min(t_ratio, 0.95)))
        sgs = score_space_groups(hk_u, Im, sm, best, chiral=self.chiral, log=self.log,
                                 obs=(hasu, r["I"], r["sig"]), cell=best.cell, sigma_of_d=sigma_of_d, tncs=mod)
        sg = sgs[0][0]
        self.st.results["space_group_alternatives"] = [g.xhm() for g, sc in sgs if sgs[0][1] - sc < 1.0]
        pseudo = [f"{o.order}-fold ({o.pseudo}, CC {o.cc:.2f})" for o in opsc.values() if getattr(o, "pseudo", "")
                  and np.isfinite(o.cc) and o.cc > 0.3]
        alts = [x for x in scores if x is not chosen_score and x.prob > 0.01][:3]
        self.decide("point group", f"{best.bravais} {best.point_group}", margin=chosen_score.prob, unit="posterior",
                    low=chosen_score.prob < 0.9, evidence=dict(expected_cc=float(cc_true), pseudo_symmetry=pseudo),
                    alternatives=[f"{x.cand.bravais} {x.cand.point_group} ({x.prob:.2f})" for x in alts])
        # for the decision audit (_write_audit_merges): each alternative's conventional reciprocal basis in the lab
        # frame, so the final data can be mapped to it whatever reindexing follows
        Ast0 = self.st.exp.crystal.Astar
        self.st.results["point_group_alternatives"] = [
            dict(label=f"{x.cand.bravais} {x.cand.point_group}", space_group=x.cand.space_group, prob=float(x.prob),
                 cell=np.asarray(x.cand.cell, float), Astar=Ast0 @ np.linalg.inv(np.asarray(x.cand.P, float).T))
            for x in alts]
        gap = float(sgs[0][1] - sgs[1][1]) if len(sgs) > 1 else None
        self.decide("space group", sg.xhm(), margin=gap, unit="log-likelihood", low=gap is not None and gap < 2.0,
                    alternatives=[f"{g.xhm()} ({sc - sgs[0][1]:+.1f})" for g, sc in sgs[1:4]])
        self.log(f"  SCALE: selected {best.bravais} point group {best.point_group} -> space group {sg.xhm()} (#{sg.number})")
        return best, sg

    def _write_merged(self, merged, cell, sg, dcut, wavelength, final):
        aniso = write_merged_outputs(merged, cell, sg, dcut, wavelength, self.wd / "xpg", log=self.log,
                                     anisotropy=self.anisotropy, free_r_reference=self.p.free_r_reference)
        tw = self.st.results.get("twinning")
        if tw is not None and tw.twin_laws:
            op, h, a, n, ab = max(tw.twin_laws, key=lambda t: t[2])
            if 0.05 < a < 0.4:
                from .analysis import detwin
                Id, sd, ok = detwin(merged["hkl"], merged["I"], merged["sig"], sg, op, a)
                md = dict(merged)
                md["I"], md["sig"] = np.where(ok, Id, merged["I"]), np.where(ok, sd, merged["sig"])
                for k in ("Iplus", "Iminus"):
                    md[k] = np.full(len(Id), np.nan)
                    md["sig" + k] = np.full(len(Id), np.nan)
                write_merged_outputs(md, cell, sg, dcut, wavelength, self.wd / "xpg_detwinned", log=self.log,
                                     anisotropy=False)
                self.log(f"  TWINNING: twin law {op}, alpha {a:.3f} (H-test) / {ab:.3f} (Britton): detwinned data "
                         f"written to xpg_detwinned_merged_iso.mtz (prefer twin refinement for alpha > 0.3)")
        if aniso:
            self._aniso_inside = aniso.pop("inside_hkl")
            self.st.results["aniso"] = aniso

    def _aniso_statistics(self, hasu, plus, I, sig, cell, sg):
        """Merging statistics of the anisotropic data, the reflections inside the cut-off surface (as in
        xpg_anisotropic.mtz): shells to the best limit with their spherical completeness, and the ellipsoidal
        completeness overall."""
        from .aniso import ellipsoidal_completeness
        from .symops import hkl_key
        an, inside = self.st.results.get("aniso"), getattr(self, "_aniso_inside", None)
        if not an or inside is None or len(inside) < 100:
            return
        sel = np.isin(hkl_key(hasu), hkl_key(inside))
        sh, ov, _ = merging_statistics(hasu[sel], plus[sel], I[sel], sig[sel], cell, sg, nbins=10,
                                       dmin=float(min(an["limits"])), anomalous=True)
        ec = ellipsoidal_completeness(inside, cell, sg, an["limits"], an["directions_hkl"])
        self.st.results.update(shells_aniso=sh, overall_aniso=ov, ellipsoidal_completeness=ec[2] if ec else None)
        self.log("\n  SCALE: anisotropic data (inside the cut-off surface, limits " +
                 " / ".join(f"{v:.2f}" for v in an["limits"]) + " A; xpg_anisotropic.mtz)" +
                 (f", ellipsoidal completeness {100 * ec[2]:.1f}%" if ec else "") + "\n" +
                 format_stats_table(sh, ov, anomalous=True) + "\n")

    def _write_report(self, final):
        st = self.st
        out = dict(format="xpg-summary/1", space_group=st.space_group.xhm() if st.space_group else None,
                   space_group_alternatives=st.results.get("space_group_alternatives"),
                   cell=st.exp.crystal.cell.tolist(), isa=st.results.get("isa"),
                   cc_half_cutoff=st.results.get("cc_half_cutoff"),
                   sigma_d=st.model.sigma_d, sigma_m=st.model.sigma_m,
                   aniso=st.results.get("aniso"), primary_mtz=st.results.get("primary_mtz", "first lattice"),
                   overall=_stats(st.results["overall"]))
        if st.results.get("overall_cut") is not None and st.results.get("shells_cut"):
            # the report's table: overall, inner and outer shell to the CC1/2 = 0.3 limit
            sh = st.results["shells_cut"]
            out["statistics"] = dict(overall=_stats(st.results["overall_cut"]), inner=_stats(sh[0]), outer=_stats(sh[-1]))
        if st.results.get("overall_aniso") is not None and st.results.get("shells_aniso"):
            sh = st.results["shells_aniso"]
            out["statistics_aniso"] = dict(overall=_stats(st.results["overall_aniso"]), inner=_stats(sh[0]),
                                           outer=_stats(sh[-1]),
                                           ellipsoidal_completeness=st.results.get("ellipsoidal_completeness"))
        # data pathologies for the reports and MR: tNCS, twinning (all twin laws tested), twinning hidden in the
        # apparent symmetry
        patho = {}
        if st.results.get("tncs"):
            tn = st.results["tncs"]
            patho["tncs"] = dict(tn, present=bool(np.isfinite(tn["ratio"]) and tn["ratio"] > 0.2))
        tw = st.results.get("twinning")
        if tw is not None:
            from .analysis import l_test_alpha
            patho["twinning"] = dict(l_mean=tw.l_mean, alpha_l=l_test_alpha(tw.l_mean), i2_over_i_sq=tw.i2_over_i_sq,
                                     laws=[dict(law=op, alpha_h=a_h, alpha_britton=a_b, pairs=n)
                                           for op, _, a_h, n, a_b in tw.twin_laws])
        if st.results.get("hidden_twin"):
            patho["hidden_twin"] = st.results["hidden_twin"]
        if patho:
            out["pathologies"] = patho
        (self.wd / "xpg_summary.json").write_text(json.dumps(out, indent=1, default=float))

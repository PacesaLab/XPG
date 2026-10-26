"""Extra lattices: pass-1 triage, joint merge with the main lattice and the merge verdict (which output is primary),
per-lattice corrections and report."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .geometry import rotation_matrices_np
from .io.legacy import write_legacy_hkl
from .lattice import lattice_candidates
from .models import Crystal, Experiment
from .models import d_spacing_from_exp
from .scaling import Scaler
from .symops import apply_rot, to_asu
LATTICE_TRIAGE = os.environ.get("XPG_LATTICE_TRIAGE", "1") != "0"
TRIAGE_TOL = 0.05
# largest mean per-shell CC1/2 loss of the all-lattice merge against the main lattice alone, over the main lattice's
# shells (to its CC1/2 = 0.3 limit): isomorphous lattices add observations and raise CC1/2, a non-isomorphous one
# lowers it at every resolution even when the CC1/2 = 0.3 limit holds. Looser in the triage (pass 1, fixed prior)
CC_LOSS_TOL = 0.05
TRIAGE_CC_TOL = 0.10
# the triage merge needs only its CC1/2 resolution limit and exclusions
TRIAGE_PRIOR = 0.3

class LatticeStage:
    """Pipeline mixin (see pipeline.Pipeline)."""

    def _triage_extra_lattices(self, merged, cell, sg):
        """Before pass 2: when the merge of all lattices from pass 1 would not be kept (it costs resolution or claims
        an implausible one: the main lattice alone is the output), the extra lattices are not integrated again.
        XPG_LATTICE_TRIAGE=0 keeps them."""
        import shutil
        td = self.wd / "lattice_triage"
        td.mkdir(exist_ok=True)
        for f in ("xpg_unmerged.mtz", "xpg_geometry.json"):
            if (self.wd / f).exists():
                shutil.copyfile(self.wd / f, td / f)
        self.log("  LATTICES: pass-1 evaluation of the extra lattices (keep them for pass 2 only if their merge is kept)")
        wd0 = self.wd
        try:
            self.wd = td
            keep, reason = self._process_extra_lattices(merged, cell, sg, triage=True)
        except Exception as e:  # pragma: no cover
            self.log(f"  LATTICES: evaluation failed ({e}); extra lattices kept")
            return
        finally:
            self.wd = wd0
        n = len(self.st.results["extra_refl"])
        self.decide("extra lattices in pass 2", "kept" if keep else "dropped", evidence=dict(reason=reason, n=n))
        if keep:
            self.log(f"  LATTICES: {n} extra lattice(s) kept for pass 2 ({reason})")
            return
        self.log(f"  LATTICES: {n} extra lattice(s) not integrated in pass 2 ({reason})")
        self.st.indexing.extra_lattices = []
        self.st.results["extra_refl"] = {}

    def _merge_verdict(self, d_all, d_main, n_all, tol=0.01, cc_loss=None, cc_tol=CC_LOSS_TOL):
        """(keep, reason, margin) for the merge of all lattices: kept unless it costs resolution (CC1/2 = 0.3 limit,
        tolerance tol), lowers CC1/2 at equal resolution (cc_loss: mean per-shell loss, tolerance cc_tol) or claims
        more than the observations can reach."""
        # a weak, partly non-isomorphous lattice degrades the merge; no CC1/2 = 0.3 limit for the merge (CC1/2 never
        # reaches 0.3) counts as a loss
        if d_main and (not d_all or d_all > (1.0 + tol) * d_main):
            return False, "costs resolution", (d_all / d_main - 1) if d_all else None
        if cc_loss is not None and cc_loss > cc_tol:
            return False, "lowers CC1/2", cc_loss
        d_best = self._plausible_merge_limit(n_all)
        if d_all and d_best and d_all < 0.97 * d_best:
            return False, "implausible", d_best / d_all - 1
        return True, "no resolution loss", (1 - d_all / d_main) if d_all and d_main else None

    def _cc_half_loss(self, r_all):
        """Mean CC1/2 loss of the all-lattice merge against the main lattice alone over the main lattice's shells (to
        its CC1/2 = 0.3 limit), with the per-shell values (main, merge); None without both."""
        main, merge = self.st.results.get("shells_cut"), r_all.get("shells_compare")
        if not main or not merge or len(main) != len(merge):
            return None
        a = np.array([s.cc_half for s in main], float)
        b = np.array([s.cc_half for s in merge], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            return None
        return float(np.mean(a[ok] - b[ok])), a, b

    def _process_extra_lattices(self, merged_main, cell, sg, triage=False):
        """Additional lattices: bring to the main conventional setting (lattice match + indexing-ambiguity
        resolution against the main data), write xpg_lattice<k>_unmerged.mtz and merge everything with the
        multi-data-set scaling (xpg_all_lattices*). triage: evaluation only (in the current work directory); returns
        (keep, reason) of the merge verdict."""
        from .analysis import best_reindex, merge_asu
        from .multiset import run_multiset
        g = self.st.exp.geom
        files = ["xpg_unmerged.mtz"]
        rk, rI = merge_asu(self._main_hkl, self._main_I, self._main_sig, sg)
        for k, refl in self.st.results["extra_refl"].items():
            sub = self.st.indexing.extra_lattices[k]
            cands = [c for c in lattice_candidates(sub.exp.crystal.cell, max_obliq=self.p.lattice_tolerance)
                     if c.bravais == self.st.lattice.bravais and c.point_group == self.st.lattice.point_group]
            if not cands:
                self.log(f"  LATTICE {k + 2}: not compatible with the main lattice ({self.st.lattice.bravais}); skipped")
                continue
            cand = min(cands, key=lambda c: np.abs(c.cell - cell).sum())
            sel = (refl["n_over"] == 0) & (refl["peak"] >= self.p.minpk) & np.isfinite(refl["I"]) & (refl["sig"] > 0)
            if "rho" in refl:
                sel &= ~(np.nan_to_num(refl["rho"]) > 0.8)
            hkl = refl["hkl"][sel] @ cand.P
            I, sig = refl["I"][sel], refl["sig"][sel]
            R, ccs = best_reindex(hkl, I, sig, rk, rI, cell, sg)
            hkl = apply_rot(hkl, R)
            cc = max(c for c, _ in ccs if np.isfinite(c)) if any(np.isfinite(c) for c, _ in ccs) else np.nan
            Ast = sub.exp.crystal.Astar @ np.linalg.inv(cand.P.T) @ np.linalg.inv(R.T)
            exp_k = Experiment(g.copy(), Crystal.from_Astar(Ast, cand.system, space_group=sg.xhm()))
            # same treatment as the main lattice (ice rings, Iprf/Isum combination, scaling with bad-range
            # pruning): scaling alone would only rescale, keeping empty stretches and misfits
            rs = {kk: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape else v) for kk, v in refl.items()}
            rs["hkl"] = hkl
            keep = np.ones(len(I), bool)
            try:
                I, sig, keep = self._correct_extra_lattice(rs, exp_k, sg, k + 2)
            except Exception as e:  # pragma: no cover
                self.log(f"  LATTICE {k + 2}: scaling failed ({e}); written unscaled")
            fn = f"xpg_lattice{k + 2}_unmerged.mtz"
            ws = np.flatnonzero(sel)[keep]
            # rejected misfits carry a negative sigma (legacy-format convention); the MTZ holds accepted observations only
            # (as the main lattice's): a sigma <= 0 in a merge input made the joint scaling objective non-finite
            acc = sig[keep] > 0
            self._write_unmerged(self.wd / fn, exp_k, sg, hkl[keep][acc], I[keep][acc], sig[keep][acc],
                                 refl["x"][ws][acc], refl["y"][ws][acc], refl["z"][ws][acc], refl["rlp"][ws][acc])
            if self.p.export_legacy and not triage:
                write_legacy_hkl(self.wd / f"XDS_ASCII_lattice{k + 2}.HKL", exp_k, sg.number, hkl[keep], I[keep], sig[keep],
                                refl["x"][ws], refl["y"][ws], refl["z"][ws], refl["rlp"][ws], refl["peak"][ws],
                                refl["corr"][ws], friedel=self.p.friedels_law, data_range=self.p.data_range,
                                sigma_m=self.st.model.sigma_m, sigma_d=self.st.model.sigma_d)
            self.log(f"  LATTICE {k + 2}: {sel.sum()} reflections, CC to main lattice {cc:.3f} -> {fn}")
            files.append(fn)
        if len(files) > 1:
            inp = ["SPACE_GROUP_NUMBER=%d" % sg.number,
                   "UNIT_CELL_CONSTANTS= " + " ".join(f"{x:.3f}" for x in cell),
                   "OUTPUT_FILE=xpg_all_lattices",
                   f"FRIEDEL'S_LAW={'TRUE' if self.p.friedels_law else 'FALSE'}"]
            ref = self._lattice_report(files, cell, sg)
            inp += [f"INPUT_FILE={'*' if i == ref else ''}{f}" for i, f in enumerate(files)]
            self.log(f"  LATTICES: merging all lattices ({', '.join(files)})")
            res = run_multiset(self.wd / "xpg_lattices", self.wd, log=self.log, anisotropy=self.anisotropy,
                             reject_nonisomorphous="worst", inp_text="\n".join(inp) + "\n",
                             export_legacy=self.p.export_legacy and not triage, stop_at_reference=triage,
                             prior=TRIAGE_PRIOR if triage else None,
                             compare_shells=self.st.results.get("shells_cut"))
            r_all = (res or {}).get("xpg_all_lattices") or {}
            cc = self._cc_half_loss(r_all)
            if cc is not None:
                self.log(f"  LATTICES: CC1/2 on the main lattice's shells, main lattice / all lattices: "
                         + " ".join(f"{x:.3f}/{y:.3f}" for x, y in zip(cc[1], cc[2]))
                         + f" (mean loss {cc[0]:+.3f})")
            if triage:
                # every extra lattice excluded from the merge (Delta-CC1/2 outliers): the 'merge' is the main lattice
                # alone and costs nothing, but the extra lattices contribute nothing either
                exc = {Path(f).name for f in (r_all.get("excluded") or [])}
                if len(files) > 1 and set(files[1:]) <= exc:
                    return False, "every extra lattice excluded from the merge (Delta-CC1/2)"
                keep, reason, _m = self._merge_verdict(r_all.get("cc_half_cutoff"), self.st.results.get("cc_half_cutoff"),
                                                       r_all.get("n_obs"), tol=TRIAGE_TOL,
                                                       cc_loss=cc[0] if cc else None, cc_tol=TRIAGE_CC_TOL)
                return keep, reason
            if r_all.get("excluded") is not None:
                exc = [Path(f).name for f in r_all["excluded"]]
                self.decide("lattices dropped from the merge", ", ".join(exc) if exc else "none",
                            evidence=dict(threshold=r_all.get("dcc_threshold"), noise=r_all.get("dcc_noise"),
                                          delta_cc_half=r_all.get("delta_cc_half")))
                if set(files[1:]) <= set(exc):
                    # the 'merge' is the main lattice alone, rescaled: the main lattice's own outputs stay primary
                    self.log("  LATTICES: every extra lattice excluded from the merge (Delta-CC1/2): the main lattice "
                             "stays the primary output")
                    self.st.results["primary_mtz"] = "first lattice (every extra lattice excluded from the merge)"
                    self.decide("lattice merge", "main lattice only (every extra lattice excluded from the merge)")
                    return
            self._promote_lattice_merge(r_all.get("cc_half_cutoff"), self.st.results.get("cc_half_cutoff"),
                                        r_all.get("n_obs"), cc)
        elif triage:
            return False, "no extra lattice compatible with the main lattice"

    def _promote_lattice_merge(self, d_all=None, d_main=None, n_all=None, cc=None):
        """The all-lattice merge becomes the primary MTZ output when it costs no resolution; the first lattice alone
        is kept as xpg_lattice1_*.mtz and xpg_all_lattices_unmerged.mtz holds all of them. XPG_LATTICES_PRIMARY=0:
        the first lattice stays primary."""
        import shutil
        if os.environ.get("XPG_LATTICES_PRIMARY", "1").upper() in ("0", "F", "FALSE", "N", "NO", "OFF"):
            return
        sfx = ("merged_iso.mtz", "anisotropic.mtz", "unmerged.mtz")
        keep, reason, margin = self._merge_verdict(d_all, d_main, n_all, cc_loss=cc[0] if cc else None)
        if reason == "costs resolution":
            self.log(f"  LATTICES: the merge of all lattices reaches {f'{d_all:.2f} A' if d_all else 'no CC1/2 = 0.3 limit'} vs {d_main:.2f} A for the main lattice "
                     f"alone (CC1/2 = 0.3): the main lattice stays the primary output (all lattices: "
                     f"xpg_all_lattices_*.mtz)")
            self.st.results["primary_mtz"] = "first lattice (all-lattice merge costs resolution)"
            self.decide("lattice merge", "main lattice only (all-lattice merge costs resolution)",
                        margin=margin, unit="relative resolution loss",
                        low=bool(d_all) and d_all < 1.03 * d_main, evidence=dict(d_all=d_all, d_main=d_main))
            return
        if reason == "lowers CC1/2":
            self.log(f"  LATTICES: the merge of all lattices lowers CC1/2 by {cc[0]:.3f} on average over the main "
                     f"lattice's shells (non-isomorphous lattices): the main lattice stays the primary output (all "
                     f"lattices: xpg_all_lattices_*.mtz)")
            self.st.results["primary_mtz"] = "first lattice (all-lattice merge lowers CC1/2)"
            self.decide("lattice merge", "main lattice only (all-lattice merge lowers CC1/2)", margin=cc[0] - CC_LOSS_TOL,
                        unit="mean CC1/2 loss above the tolerance", low=cc[0] < CC_LOSS_TOL + 0.03,
                        evidence=dict(cc_half_main=cc[1].tolist(), cc_half_all=cc[2].tolist(), d_all=d_all, d_main=d_main))
            return
        d_best = self._plausible_merge_limit(n_all)
        if reason == "implausible":
            self.log(f"  LATTICES: the merge of all lattices claims {d_all:.2f} A, beyond the {d_best:.2f} A that "
                     f"{n_all / self.st.results['n_obs_merged']:.1f}x the observations of the main lattice ({d_main:.2f} A) "
                     f"could reach (correlated errors between overlapping lattices): the main lattice stays the "
                     f"primary output (all lattices: xpg_all_lattices_*.mtz)")
            self.st.results["primary_mtz"] = "first lattice (all-lattice merge implausible)"
            self.decide("lattice merge", "main lattice only (merge claims more resolution than possible)",
                        margin=d_best / d_all - 1, unit="relative excess", low=d_all > 0.93 * d_best,
                        evidence=dict(d_all=d_all, d_main=d_main, d_plausible=d_best))
            return
        if not (self.wd / f"xpg_all_lattices_{sfx[0]}").exists():
            self.log("  LATTICES: no merged all-lattice MTZ; the first lattice stays the primary output")
            return
        for x in sfx:
            p, a = self.wd / f"xpg_{x}", self.wd / f"xpg_all_lattices_{x}"
            if p.exists():
                p.replace(self.wd / f"xpg_lattice1_{x}")
            if a.exists():
                shutil.copyfile(a, p)
        self.st.results["primary_mtz"] = "all lattices"
        self.decide("lattice merge", "all lattices merged", margin=(1 - d_all / d_main) if d_all and d_main else None,
                    unit="relative resolution gain", low=bool(d_all and d_main) and abs(d_all / d_main - 1) < 0.01,
                    evidence=dict(d_all=d_all, d_main=d_main))
        self.log("  LATTICES: primary outputs xpg_merged_iso.mtz / xpg_anisotropic.mtz / xpg_unmerged.mtz now "
                 "hold all lattices (first lattice alone: xpg_lattice1_*.mtz)")

    def _lattice_report(self, files, cell, sg):
        """Per-lattice quality and pairwise isomorphism (model-free, analysis.lattice_isomorphism); returns the index
        of the reference lattice for merging: the best data (CC1/2 in the shells where every lattice is
        informative), not the one with the most spots. All lattices are still merged (merging non-isomorphous
        lattices with the best one did not hurt; excluding them did)."""
        from .analysis import lattice_isomorphism
        sets = []
        from .multiset import read_input_set
        for f in files:
            st_ = read_input_set(self.wd / f)
            ok_ = st_.sig > 0
            sets.append((st_.hkl[ok_], st_.I[ok_], st_.sig[ok_]))
        try:
            iso = lattice_isomorphism(sets, cell, sg)
        except Exception as e:  # pragma: no cover
            self.log(f"  LATTICES: isomorphism analysis failed ({e})")
            return 0
        e, cch = iso["d_edges"], np.array(iso["cc_half"], float)
        self.log("  LATTICES: quality and isomorphism (kappa = CC_AB / sqrt(r_A r_B); 1 = differ only by noise)")
        self.log("     shells (A):   " + " ".join(f"{a:5.2f}-{b:<5.2f}" for a, b in zip(e[:-1], e[1:])))
        for i, f in enumerate(files):
            self.log(f"     CC1/2 {i + 1} {f[:22]:22s} " + " ".join(f"{x:11.3f}" for x in cch[i]))
        for (i, j), k in iso["kappa"].items():
            self.log(f"     kappa {i + 1}-{j + 1}{'':20s}" + " ".join(f"{x:11.3f}" if np.isfinite(x) else f"{'-':>11s}"
                                                             for x in k))
        inf = np.all(np.nan_to_num(cch, nan=0.0) >= 0.3, axis=0)
        if not inf.any():
            inf = np.zeros(cch.shape[1], bool)
            inf[0] = True
        score = np.nanmean(cch[:, inf], axis=1)
        ref = int(np.nanargmax(score))
        self.st.results["lattice_isomorphism"] = dict(cc_half=cch.tolist(), kappa={f"{i + 1}-{j + 1}": v for (i, j), v
                                                                                   in iso["kappa"].items()}, reference=ref + 1)
        self.log(f"  LATTICES: reference for merging: {files[ref]} (CC1/2 {score[ref]:.3f} in the informative shells"
                 + (f"; the first-indexed lattice has {score[0]:.3f})" if ref != 0 else ")"))
        bad = [f"{i + 1}-{j + 1}" for (i, j), k in iso["kappa"].items() if np.nanmin(np.r_[k, np.inf]) < 0.7]
        if bad:
            self.log(f"  LATTICES WARNING: non-isomorphous pairs (kappa < 0.7 in some shell): {', '.join(bad)}; all "
                     "lattices are merged ; separate files are kept for inspection")
        return ref

    def _correct_extra_lattice(self, r, exp_k, sg, label):
        """SCALE for an additional lattice (hkl already in the main setting): returns I, sigma with rejected
        observations (ice rings, misfits, pruned image ranges) marked by negative sigma."""
        p, g = self.p, exp_k.geom
        ok = np.ones(len(r["I"]), bool)
        rings = self.st.results.get("ice_rings_excluded")
        ranges = list(p.exclude_resolution) + list(self.st.results.get("ice_rings") or [] if rings is None else rings)
        if ranges:
            d = d_spacing_from_exp(r["hkl"], exp_k)
            for lo, hi in ranges:
                ok &= ~((d >= min(lo, hi)) & (d <= max(lo, hi)))
        hasu, _, plus = to_asu(r["hkl"], sg)
        from .merging import unique_ids
        strict = (not p.friedels_law) if p.strict_absorption is None else bool(p.strict_absorption)
        uid, _ = unique_ids(hasu, plus, anomalous=strict)
        I0, s0 = r["I"].copy(), r["sig"].copy()
        if "Isum" in r and p.combine_intensities:
            I0, s0 = self._combine_prf_sum(r, uid)
        rr = {kk: (v[ok] if isinstance(v, np.ndarray) and v.shape[:1] == ok.shape else v) for kk, v in r.items()}
        rr["I"], rr["sig"] = I0[ok], s0[ok]
        _, u = np.unique(uid[ok], return_inverse=True)
        s = 1.0 / d_spacing_from_exp(rr["hkl"], exp_k)
        phi = g.z_to_phi(rr["z"])
        Rm = rotation_matrices_np(g.axis, -np.radians(phi))
        s1 = rr["S"] / np.linalg.norm(rr["S"], axis=1, keepdims=True)
        s1c = np.einsum("nij,nj->ni", Rm, s1) @ exp_k.crystal.U
        from .symops import reflection_props
        cen_o, eps_o, _ = reflection_props(hasu[ok], sg)
        self.log(f"  LATTICE {label}: SCALE")
        sc = Scaler(rr["I"], rr["sig"], u, phi, s ** 2, s1c, rr["x"], rr["y"], g.nx, g.ny, corrections=p.corrections,
                    nsig=6.0 * p.wfac1, snrc=p.snrc, nbatch=p.nbatch, eps=eps_o, centric=cen_o)
        sc.run(log=lambda *a: None)
        if p.prune_image_groups:
            keep_removed = self.st.results.get("removed_image_groups")
            self._prune_image_groups(sc, rr, u)
            self.st.results["removed_image_groups"] = keep_removed
        Is, ss, act = sc.scaled()
        I_out = np.zeros(len(ok)); s_out = -np.abs(s0)
        I_out[ok] = Is
        s_out[ok] = np.where(act, ss, -ss)
        I_out[~ok] = I0[~ok]
        # written: accepted observations and misfits (negative sigma) in frames that still have accepted data;
        # ice-ring / excluded-resolution and pruned-range observations are omitted (as for the main lattice)
        fr = np.floor(rr["z"]).astype(int)
        live = np.bincount(fr - fr.min(), act.astype(int))[fr - fr.min()] > 0
        keep = np.zeros(len(ok), bool)
        keep[np.flatnonzero(ok)] = act | (sc.misfit & live)
        self.log(f"  LATTICE {label}: ISa {sc.isa():.1f}; {act.sum()} of {len(ok)} observations kept")
        return I_out, s_out, keep

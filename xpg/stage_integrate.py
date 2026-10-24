"""INTEGRATE stage: per-block geometry (post-refinement, position prior, smoothing), the shoebox integration, the
forward model (XPG_INTEGRATE_MODE) and the second-pass decision."""
from __future__ import annotations

import os
import time

import numpy as np

from .indexing import assign_indices
from .integrate import IntegrationModel, estimate_sigmas, integrate
from .io.legacy import write_integrate_hkl
from .models import Experiment
from .models import d_spacing_from_exp
from .refine import refine


FM_AUTO_OVERLAP = 0.5   # XPG_INTEGRATE_MODE=auto: forward model when at least this fraction has an overlap partner

def _fm_mode():
    """XPG_INTEGRATE_MODE: 'auto', 'fm' (forward model always), 'shoebox' (shoebox integrator only)."""
    return os.environ.get("XPG_INTEGRATE_MODE", "auto").lower()

class IntegrateStage:
    """Pipeline mixin (see pipeline.Pipeline)."""

    # ------------------------------------------------------------------
    def _block_models(self, exp: Experiment, flags, exclude_spots=None):
        """Per-phi-block refinement of the experiment on indexed spots (scan-varying model)."""
        g = exp.geom
        f0, f1 = self.p.data_range
        z0, z1 = f0 - 1.0, float(f1)
        nfr = max(int(round(self.p.delphi / abs(g.osc))), 1)
        edges = np.arange(z0, z1, nfr, dtype=float)
        edges = np.append(edges, z1)
        if len(edges) > 2 and edges[-1] - edges[-2] < 0.5 * nfr:
            edges = np.delete(edges, -2)
        sp = self.st.spots
        flags = [f for f in flags if f != "CELL"] if len(edges) > 2 else flags
        # per-block detector position: the detector does not move; what a block can show is the crystal moving in the
        # beam (tens of um) -> Gaussian prior around the global position (XPG_POS_PRIOR = sd orgx, orgy [px], distance
        # [mm]; 0 = unrestrained). Unrestrained, weak 5-deg blocks let the position wander empirical Bayes for the
        # prior widths (unless XPG_POS_PRIOR is given): first pass with a weak prior, the true block-to-block variance
        # tau^2 = var(estimates) - mean(esd^2) per parameter, then the prior sd = tau (real drift of the crystal in
        # the beam is kept, noise of weak blocks is shrunk)
        auto = "POSITION" in flags and not os.environ.get("XPG_POS_PRIOR")
        # physical bound on the per-block detector position (0.1 mm): the crystal moves by at most ~0.1 mm in the beam
        # (beams 5-100 um, sphere of confusion 1-50 um); a larger learnt 'scatter' is the per-block degeneracy of
        # origin / distance with beam direction and orientation, and on long axes a shift of one reflection spacing
        # re-indexes the block. The weak first pass stays within 3x.
        cap_mm = 0.1
        cap = [cap_mm / g.qx, cap_mm / g.qy, cap_mm]
        # and small against the reflection spacing on the detector (lambda D / (a_max q) near the beam): a shift of a
        # sizeable fraction of a spacing is a re-indexing along the long axis
        a_max = float(np.max(exp.crystal.cell[:3]))
        spacing = g.wavelength * g.distance / (a_max * min(g.qx, g.qy))
        cap = [min(cap[0], 0.05 * spacing), min(cap[1], 0.05 * spacing), cap[2]]
        pos_prior = self._pos_prior(flags, g, sds=[min(5.0, 3 * cap[0]), min(5.0, 3 * cap[1]), min(1.0, 3 * cap[2])]
                                    if auto else None)
        nb = len(edges) - 1
        allowed = np.ones(len(sp), bool) if exclude_spots is None else ~exclude_spots
        in_blk = [np.flatnonzero((sp.z >= edges[i]) & (sp.z < edges[i + 1]) & allowed) for i in range(nb)]

        def fit_block(i, start, tols=(0.3, 0.2, 0.12)):
            """Re-assign this block's spots from the neighbouring block's model, then refine: the orientation
            is tracked through the sweep (crystal slippage / drift), as INTEGRATE does batch by batch."""
            idx = in_blk[i]
            if idx.size < 50:
                return start, False
            cur = start
            # tightening tolerance: a second domain misoriented by ~1 deg must not capture the refinement
            for tol in tols:
                p0 = cur.geom.spot_to_p0(sp.x[idx], sp.y[idx], sp.z[idx])
                hkl, ok = assign_indices(cur.crystal.Astar, p0, tol=tol)
                if ok.sum() < 30:
                    return start, False
                r = refine(cur, hkl[ok], sp.x[idx][ok], sp.y[idx][ok], sp.z[idx][ok], flags, max_iter=15,
                           n_reject_cycles=2, prior=pos_prior)
                if not np.isfinite(r.rmsd_xy) or r.rmsd_xy > 3:
                    return start, False
                cur = r.exp
                if r.esd is not None:
                    esd_pos[i] = [r.esd[r.names.index(k)] if k in r.names else np.nan for k in ("orgx", "orgy", "distance")]
            return cur, True
        esd_pos = {}
        # seed: the block best described by the global model
        frac = []
        for i in range(nb):
            idx = in_blk[i]
            if idx.size < 50:
                frac.append(0.0)
                continue
            _, ok = assign_indices(exp.crystal.Astar, g.spot_to_p0(sp.x[idx], sp.y[idx], sp.z[idx]), tol=0.2)
            frac.append(ok.mean())
        seed = int(np.argmax(frac))

        def track():
            blocks = [exp] * nb
            n_fallback = 0
            m, okb = fit_block(seed, exp)
            blocks[seed] = m
            n_fallback += not okb
            for direction in (1, -1):
                prev = blocks[seed]
                i = seed + direction
                while 0 <= i < nb:
                    m, okb = fit_block(i, prev)
                    blocks[i] = m
                    n_fallback += not okb
                    prev = m
                    i += direction
            return blocks, n_fallback
        blocks, n_fallback = track()
        if auto and len(esd_pos) >= 4:
            ii = sorted(esd_pos)
            est = np.array([[blocks[i].geom.orgx, blocks[i].geom.orgy, blocks[i].geom.distance] for i in ii])
            esd = np.array([esd_pos[i] for i in ii])
            # robust spread (MAD) of the estimates minus their own noise; floors keep the prior proper
            spread = (1.4826 * np.median(np.abs(est - np.median(est, 0)), 0)) ** 2
            tau2 = spread - np.nanmedian(esd ** 2, 0)
            tau = np.sqrt(np.maximum(np.nan_to_num(tau2, nan=0.0), np.array([0.05, 0.05, 0.01]) ** 2))
            capped = bool(np.any(tau > np.array(cap)))
            tau = np.minimum(tau, np.array(cap))
            self.st.results["pos_prior_sd"] = [float(v) for v in tau]
            self.log(f"  INTEGRATE: per-block position prior by empirical Bayes: block scatter (robust sd) "
                     f"{np.sqrt(spread[0]):.2f}/{np.sqrt(spread[1]):.2f} px, {np.sqrt(spread[2]):.3f} mm; noise (median esd) "
                     f"{np.nanmedian(esd[:, 0]):.2f}/{np.nanmedian(esd[:, 1]):.2f} px, {np.nanmedian(esd[:, 2]):.3f} mm -> prior sd "
                     f"{tau[0]:.2f}/{tau[1]:.2f} px, {tau[2]:.3f} mm" + (f" (capped: {cap_mm:g} mm, 0.05 x the {spacing:.1f} px "
                                                                          f"reflection spacing)" if capped else ""))
            pinned = tau[:2].max() <= 0.051
            self.decide("per-block detector position", "fixed (block scatter = noise)" if pinned else
                        f"restrained (prior sd {tau[0]:.2f}/{tau[1]:.2f} px, {tau[2]:.3f} mm)",
                        evidence=dict(scatter=np.sqrt(spread).tolist(), noise=np.nanmedian(esd, 0).tolist()))
            pos_prior = self._pos_prior(flags, g)
            esd_pos.clear()
            # restrained pass from the converged weak-prior models: the spot assignment is already tight (no
            # tracking cascade needed), one refinement per block at the final tolerance
            first, n_fallback = blocks, 0
            blocks = []
            for i in range(nb):
                m, okb = fit_block(i, first[i], tols=(0.12,))
                blocks.append(m)
                n_fallback += not okb
        rot = [np.degrees(np.arccos(np.clip((np.trace(b.crystal.U @ exp.crystal.U.T) - 1) / 2, -1, 1))) for b in blocks]
        if "POSITION" in flags:
            P_ = np.array([[b.geom.orgx - g.orgx, b.geom.orgy - g.orgy, b.geom.distance - g.distance] for b in blocks])
            self.log(f"  INTEGRATE: per-block detector position vs global (median |d| / max |d|): orgx "
                     f"{np.median(np.abs(P_[:, 0])):.2f}/{np.abs(P_[:, 0]).max():.2f} px, orgy {np.median(np.abs(P_[:, 1])):.2f}/"
                     f"{np.abs(P_[:, 1]).max():.2f} px, distance {np.median(np.abs(P_[:, 2])):.3f}/{np.abs(P_[:, 2]).max():.3f} mm"
                     + (f" (prior sd {pos_prior['orgx'][1]:g} px, {pos_prior['distance'][1]:g} mm)" if pos_prior else " (unrestrained)"))
        self.log(f"  INTEGRATE: {len(blocks)} phi blocks of {self.p.delphi:g} deg refined ({' '.join(flags)}), orientation "
                 f"tracked from block {seed + 1}; {n_fallback} keep the neighbouring model; max deviation from the global "
                 f"orientation {max(rot):.2f} deg")
        return blocks, edges

    def _smooth_blocks(self, blocks, ref: Experiment, lam=4.0, n_iter=4, label=""):
        """Continuity of the crystal orientation (and beam direction) through the sweep: per-block fits are
        replaced by a robust Whittaker smooth of their rotation vectors (second-difference penalty ``lam``,
        blocks deviating > 3 robust sd from the smooth curve get zero weight and follow their neighbours).
        Independent 5-degree fits jump by up to ~1.5 deg on split / multi-domain crystals (a block captured by a
        neighbouring domain or poorly determined); real crystals turn continuously (cf. DIALS scan-varying
        smoothers, batch-to-batch refinement)."""
        from scipy.spatial.transform import Rotation
        nb = len(blocks)
        if nb < 5:
            return blocks
        U0 = ref.crystal.U
        rv = np.array([Rotation.from_matrix(b.crystal.U @ U0.T).as_rotvec() for b in blocks])
        bm = np.array([b.geom.beam / np.linalg.norm(b.geom.beam) for b in blocks])
        D = np.diff(np.eye(nb), 2, axis=0)
        DtD = D.T @ D

        def gcv_lambda(w):
            """Smoothing strength by generalised cross-validation (Craven & Wahba) on the inlying blocks."""
            best_ = None
            n_eff = w.sum()
            for lm in np.logspace(-1, 3, 17):
                H = np.linalg.solve(np.diag(w) + lm * DtD, np.diag(w))
                z = H @ rv
                rss = float(np.sum(w[:, None] * (rv - z) ** 2))
                dof = n_eff - np.trace(H)
                if dof <= 1:
                    continue
                g = n_eff * rss / dof ** 2
                if best_ is None or g < best_[0]:
                    best_ = (g, lm)
            return best_[1] if best_ else lam
        w = np.ones(nb)
        for _ in range(n_iter):
            # GCV under-smooths when block errors are serially correlated (a captured block drags its neighbours):
            # floor at lam = 4
            lam = max(gcv_lambda(w), 4.0)
            zr = np.linalg.solve(np.diag(w) + lam * DtD, w[:, None] * rv)
            res = np.linalg.norm(rv - zr, axis=1)
            mad = 1.4826 * np.median(res) + 1e-9
            w_new = (res < 3.0 * mad + np.radians(0.05)).astype(float)
            if np.array_equal(w_new, w) or w_new.sum() < 0.5 * nb:
                break
            w = w_new
        P = lam * DtD
        zb = np.linalg.solve(np.diag(w) + P, w[:, None] * bm)
        out = []
        for i, b in enumerate(blocks):
            g = b.geom.copy()
            g.beam = zb[i] / np.linalg.norm(zb[i])
            out.append(Experiment(g, b.crystal.copy(U=Rotation.from_rotvec(zr[i]).as_matrix() @ U0)))
        jumps0 = np.degrees(np.linalg.norm(np.diff(rv, axis=0), axis=1))
        jumps1 = np.degrees(np.linalg.norm(np.diff(zr, axis=0), axis=1))
        self.log(f"  INTEGRATE: orientation trajectory smoothed{label} (lambda {lam:.2g} by GCV): block-to-block jumps max {jumps0.max():.2f} -> "
                 f"{jumps1.max():.2f} deg (median {np.median(jumps0):.3f} -> {np.median(jumps1):.3f}); "
                 f"{int((w == 0).sum())} outlying block(s) follow their neighbours")
        return out

    def _pos_prior(self, flags, g, sds=None):
        """Gaussian prior of the per-block detector position around the global one (see _block_models). Widths:
        explicit XPG_POS_PRIOR (sd orgx, orgy [px], distance [mm]; 0 = unrestrained), else the empirical-Bayes
        estimate from the block-to-block scatter (results['pos_prior_sd']), else 0.5, 0.5, 0.1."""
        if "POSITION" not in flags:
            return None
        env = os.environ.get("XPG_POS_PRIOR")
        if sds is None:
            # ',' or ':' separated (sbatch --export splits on commas)
            sds = [float(v) for v in env.replace(":", ",").split(",")] if env else \
                self.st.results.get("pos_prior_sd", [0.5, 0.5, 0.1])
        if not all(v > 0 for v in sds):
            return None
        return dict(orgx=(g.orgx, sds[0]), orgy=(g.orgy, sds[1]), distance=(g.distance, sds[2]))

    def _refine_blocks_on_centroids(self, blocks, edges, refl, flags, min_refl=40):
        """INTEGRATE-style batch refinement: each phi block's model is refined on the observed centroids (xo, yo, zo)
        of strong, fully recorded, unoverlapped reflections integrated in the previous pass. Their indices are
        known, so spots of other crystals / domains cannot be mis-assigned (SPOTS-based block refinement assigns
        indices by tolerance, which on multi-crystal data captures foreign spots). The cell stays global."""
        flags = [f for f in flags if f != "CELL"]
        isg = refl["isum"] / np.sqrt(np.maximum(refl["vsum"], 1e-6))
        ok = (isg > 10) & (refl["peak"] > 90) & (refl["n_over"] == 0) & np.isfinite(refl["xo"]) & \
            np.isfinite(refl["zo"]) & ~(np.nan_to_num(refl.get("rho", np.zeros(len(isg)))) > 0.3)
        if "overlap" in refl:
            ok &= ~refl["overlap"].astype(bool)
        out, n_ref, dz_before, dz_after = list(blocks), 0, [], []
        for i in range(len(blocks)):
            m = ok & (refl["zo"] >= edges[i]) & (refl["zo"] < edges[i + 1])
            if m.sum() < min_refl:
                continue
            try:
                r = refine(blocks[i], refl["hkl"][m], refl["xo"][m], refl["yo"][m], refl["zo"][m], flags, max_iter=15,
                           n_reject_cycles=2, prior=self._pos_prior(flags, self.st.exp.geom))
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                continue
            if not np.isfinite(r.rmsd_xy) or r.rmsd_xy > 3:
                continue
            dz_before.append(np.median(refl["zo"][m] - refl["z"][m]))
            out[i] = r.exp
            n_ref += 1
            dz_after.append(r.rmsd_phi)
        if n_ref:
            self.log(f"  INTEGRATE: {n_ref}/{len(blocks)} phi blocks re-refined on integrated centroids of strong "
                     f"reflections (previous pass); median phi offset before {np.median(np.abs(dz_before)):.2f} frames, "
                     f"rmsd after {np.median(dz_after):.3f} deg")
        return out

    def _extra_lattices(self):
        """Extra lattices integrated separately. Domains misoriented by less than
        ``XPG_MIN_LATTICE_SEPARATION`` (deg) are integrated together with the main lattice (as is standard for
        split crystals): their spots overlap and splitting the intensity between lattices only adds noise,
        while both domains carry the same intensities."""
        if not self.p.extra_lattices or not self.st.indexing:
            return []
        from .indexing import _misorientation_sym
        out = []
        for sub in self.st.indexing.extra_lattices:
            mis = _misorientation_sym(self.st.indexing.exp.crystal, sub.exp.crystal)
            if np.isfinite(mis) and mis < self.p.min_lattice_separation:
                if not getattr(self, "_logged_merge", False):
                    self.log(f"  INTEGRATE: lattice misoriented by {mis:.2f} deg (< {self.p.min_lattice_separation:g}) "
                             f"is integrated together with the main lattice (split crystal)")
                    self._logged_merge = True
                continue
            out.append(sub)
        return out

    def _background_template(self):
        """Static background shape for integration (rings, polarisation, ...), from the BACKGROUND background."""
        if not self.p.background_template:
            return None
        if getattr(self.st, "bkg_template", None) is None and getattr(self.st, "background", None) is not None:
            from .init import background_template
            self.st.bkg_template = background_template(self.st.geom, self.st.background, self.st.mask)
            if self.st.bkg_template is not None:
                t = self.st.bkg_template[self.st.mask]
                self.log(f"  INTEGRATE: background template (radial profile incl. rings x polarisation x smooth 2D): "
                         f"range {np.percentile(t, 1):.2f}-{np.percentile(t, 99):.2f} of mean")
        return getattr(self.st, "bkg_template", None)

    def job_integrate(self, second=False):
        self.log("\n=== INTEGRATE ===" if not second else "\n=== INTEGRATE (pass 2) ===")
        t_stage = time.time()
        exp = self.st.exp
        g = exp.geom
        dmax, dmin = self.p.include_resolution
        dmin_det = g.max_resolution_at_edge()
        dmin = max(dmin, dmin_det) if dmin > 0 else dmin_det
        tmpl = self._background_template()
        if self.st.model is None or not second:
            sd0 = self.p.beam_divergence or 0.05
            sm0 = self.p.reflecting_range or 0.25
            model = IntegrationModel(sigma_d=sd0, sigma_m=sm0, gain=self.st.gain)
            if self.p.beam_divergence is None or self.p.reflecting_range is None:
                extra = [sub.exp for sub in self._extra_lattices()]
                model = estimate_sigmas(self.images, exp, model, max(dmin, 2.0), self.p.data_range, log=self.log,
                                        extra_exps=extra, bkg_template=tmpl)
                if not getattr(model, "history", None) and extra:
                    # weak extra lattices of a large cell predict a reflection next to nearly every main-lattice one,
                    # leaving no isolated strong spot: estimate from the main lattice with only the extra lattices'
                    # OBSERVED spots masked
                    sp = self.st.spots
                    obs = np.zeros(len(sp), bool)
                    for sub in self._extra_lattices():
                        m = getattr(sub, "spot_mask", None)
                        if m is not None and m.size == obs.size:
                            obs |= m
                    fx, fy, fz = sp.x[obs], sp.y[obs], sp.z[obs]
                    self.log(f"  INTEGRATE: no isolated strong spots with the {len(extra)} extra lattice(s) predicted; "
                             f"spot widths from the main lattice with their {int(obs.sum())} observed spots masked")
                    model = estimate_sigmas(self.images, exp, IntegrationModel(sigma_d=sd0, sigma_m=sm0, gain=self.st.gain),
                                            max(dmin, 2.0), self.p.data_range, log=self.log, extra_exps=(),
                                            bkg_template=tmpl, foreign_spots=(fx, fy, fz))
                if not getattr(model, "history", None):
                    self.log(f"  INTEGRATE WARNING: spot widths could not be estimated (no isolated strong spots): "
                             f"defaults sigma_D {model.sigma_d:.3f}, sigma_M {model.sigma_m:.3f} deg used "
                             f"(set [processing] beam_divergence / reflecting_range if known)")
                self.st.results["width_history"] = list(getattr(model, "history", []))
        else:
            model = self.st.model
        self.st.model = model
        t_est = time.time() - t_stage
        t_blk0 = time.time()
        extras = self._extra_lattices()
        claimed = [getattr(sub, "spot_mask", None) for sub in extras]
        nspots = len(self.st.spots)
        any_extra = np.zeros(nspots, bool)
        for c in claimed:
            if c is not None and c.size == nspots:
                any_extra |= c
        # each lattice is tracked with the spots not claimed by the other lattices
        blocks, edges = self._block_models(exp, self.p.refine_integrate, exclude_spots=any_extra if extras else None)
        if second and self.st.refl is not None:
            blocks = self._refine_blocks_on_centroids(blocks, edges, self.st.refl, self.p.refine_integrate)
        if self.p.smooth_orientation:
            blocks = self._smooth_blocks(blocks, exp)
        main_claim = self.st.indexing.indexed if (self.st.indexing is not None and extras) else None
        extra_blocks = []
        for k, sub in enumerate(extras):
            excl = np.zeros(nspots, bool) if main_claim is None or main_claim.size != nspots else main_claim.copy()
            for j, c in enumerate(claimed):
                if j != k and c is not None and c.size == nspots:
                    excl |= c
            eb, _ = self._block_models(self.st.results.get("extra_exps", {}).get(k, sub.exp), ["ORIENTATION"],
                                       exclude_spots=excl)
            extra_blocks.append(eb)
        if extra_blocks:
            self.log(f"  INTEGRATE: {len(extra_blocks)} additional lattice(s) integrated simultaneously "
                     f"(overlaps between lattices deconvoluted / masked)")
        excl = [f for f in range(self.p.data_range[0], self.p.data_range[1] + 1) if self.p.is_excluded(f)]
        if excl:
            self.log(f"  INTEGRATE: {len(excl)} frames excluded (EXCLUDE_DATA_RANGE)")
        from . import integrate as _integ
        t_blk = time.time() - t_blk0
        t_int0 = time.time()
        res = integrate(self.images, blocks, edges, model, dmin, dmax, frames=self.p.data_range, mask=self.st.mask,
                        minimum_zeta=self.p.minimum_zeta, log=self.log, exclude_frames=excl,
                        extra_lattices=extra_blocks or None, bkg_template=tmpl)
        refl = res.refl
        self.log(f"  INTEGRATE: stage time: spot widths {t_est:.1f}s, block models {t_blk:.1f}s, integration "
                 f"{time.time() - t_int0:.1f}s")
        bp = getattr(res, "block_params", None) or {}
        if bp:
            ks = sorted(bp)
            self.events.emit("metric", name="block_sigmas", title="Per-block spot widths", x=[int(k) + 1 for k in ks],
                             xlabel="phi block", series={"sigma_M (deg)": [float(bp[k]["sigma_m"]) for k in ks],
                                                         "sigma_D (deg)": [float(bp[k]["sigma_d"]) for k in ks]})
        # the learnt 2D reference profiles of this integration (block -> (regions, G, G) density in sigma_eff
        # units, ~2 MB): the pixel-level forward model (xpg/forward) fits with exactly these profiles
        self.st.results["integration_profiles"] = {int(b): np.asarray(p_, np.float32) for b, p_ in (res.profiles or {}).items()}
        self.st.results["integration_box_n"] = float(_integ.N_BOX)
        # per-block widths and rocking-curve shape of this integration (the learnt profiles are in units of the
        # block's sigma_D; the forward model needs both)
        self.st.results["integration_block_params"] = {
            int(k): dict(sigma_d=float(v["sigma_d"]), sigma_m=float(v["sigma_m"]),
                         rcw=[float(x) for x in v.get("rcw", [])]) for k, v in bp.items()}
        if extra_blocks:
            n_ov = int(np.sum(refl["overlap"] & (refl["lattice"] >= 0)))
            self.log(f"  INTEGRATE: {n_ov} reflections had an overlapping neighbour and were deconvoluted")
        # LP correction
        zeta_l, pol = g.lp_factor(refl["S"])
        # LP plus explicit sensor efficiency and air absorption (SCALE applies the same terms)
        rlp = np.abs(zeta_l) / pol / g.absorption_factors(refl["S"])
        refl["rlp"] = rlp
        refl["I"] = refl["iprf"] * rlp
        refl["sig"] = np.sqrt(np.maximum(refl["vprf"], 0)) * rlp
        refl["Isum"] = refl["isum"] * rlp
        refl["sigsum"] = np.sqrt(np.maximum(refl["vsum"], 0)) * rlp
        refl["peak"] = 100 * np.clip(refl["p_valid"] / np.maximum(refl["p_total"], 1e-9), 0, 1) * \
            refl.get("coverage", 1.0)
        refl["corr"] = 100 * np.nan_to_num(refl["corr"])
        good = np.isfinite(refl["I"]) & np.isfinite(refl["sig"]) & (refl["sig"] > 0)
        refl = {k: (v[good] if isinstance(v, np.ndarray) and v.shape[:1] == good.shape else v) for k, v in refl.items()}
        lat = refl.get("lattice", np.zeros(len(refl["I"]), int))
        self.st.results["extra_refl"] = {}
        for k in range(1, int(lat.max()) + 1 if lat.size else 1):
            m = lat == k
            self.st.results["extra_refl"][k - 1] = {kk: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v)
                                                    for kk, v in refl.items()}
        m0 = lat == 0
        refl = {k: (v[m0] if isinstance(v, np.ndarray) and v.shape[:1] == m0.shape else v) for k, v in refl.items()}
        self.st.refl = refl
        self.st.results["integration_blocks"] = blocks
        # crowding (reflections with a deconvoluted overlap partner): the quantity XPG_INTEGRATE_MODE=auto decides on
        frac_ov = float(np.mean(np.asarray(refl.get("overlap", np.zeros(len(refl["I"]), bool)), bool))) if len(refl["I"]) else 0.0
        self.st.results["overlap_fraction"] = frac_ov
        if _fm_mode() not in ("fm", "auto"):
            n_extra = len(self.st.results.get("extra_refl") or {})
            self.log(f"  INTEGRATE: {100 * frac_ov:.1f}% of the reflections have an overlap partner"
                     + (" (XPG_INTEGRATE_MODE=auto would use the forward model)" if frac_ov >= FM_AUTO_OVERLAP
                        and not n_extra else ""))
        if _fm_mode() in ("fm", "auto") and (second or not self.second_pass):
            self._forward_model_integrate(auto=_fm_mode() == "auto")
            refl = self.st.refl
        if self.p.export_legacy:
            write_integrate_hkl(self.wd / "INTEGRATE.HKL", exp, refl, self.p.data_range, model.sigma_m, model.sigma_d,
                                self.p.name_template)
        np.savez_compressed(self.wd / ("xpg_integrated_pass2.npz" if second else "xpg_integrated.npz"),
                            **{k: v for k, v in refl.items() if isinstance(v, np.ndarray) and k != "fz"})

    def _forward_model_integrate(self, auto=False):
        """INTEGRATE mode XPG_INTEGRATE_MODE=fm: the final integration's intensities are replaced by the pixel-level
        forward model's (joint Poisson fit of all reflections of each window with a global smooth background, the
        shoebox integrator's learnt profiles and geometry; xpg.forward.driver). Reflections the forward model does
        not fit (phi-binned very wide ones, supports too long for a window) keep the shoebox values. Options:
        XPG_FM_WINDOW (frames or auto), XPG_FM_MEM_GB, XPG_FM_TILES (NXxNY or auto)."""
        from .forward.driver import integrate_sweep
        r = self.st.refl
        extra = self.st.results.get("extra_refl") or {}
        joint = bool(extra) and os.environ.get("XPG_FM_MULTI_LATTICE", "0") == "1"
        self.st.results["fm_used"] = False
        if auto:
            # XPG_INTEGRATE_MODE=auto: the forward model only where it wins: crowded single-lattice data. Clean data
            # keep the shoebox intensities (FM level or less precise on fine-sliced Eiger data).
            frac = float(np.mean(np.asarray(r.get("overlap", np.zeros(len(r["I"]), bool)), bool)))
            lim = FM_AUTO_OVERLAP
            if frac < lim or (extra and not joint):
                self.log(f"\n=== INTEGRATE (forward model) ===\n  FM auto: {100 * frac:.1f}% of the reflections with "
                         f"an overlap partner ({'<' if frac < lim else '>='} {100 * lim:.0f}%)"
                         + (f", {len(extra)} additional lattice(s)" if extra else "") + ": shoebox intensities kept")
                self.decide("integration", "shoebox (forward model not needed: uncrowded)" if frac < lim else
                            "shoebox (forward model skipped: multiple lattices)",
                            evidence=dict(overlap_fraction=frac, extra_lattices=len(extra)))
                return
        if extra and not joint:
            # the window model holds the main lattice only: the other lattices' spots would sit in its smooth
            # background. Shoebox intensities are kept until the forward model fits all lattices jointly.
            self.log(f"\n=== INTEGRATE (forward model) ===\n  FM: {len(self.st.results['extra_refl'])} additional "
                     f"lattice(s): not supported by the forward model yet, shoebox intensities kept")
            self.decide("integration", "shoebox (forward model skipped: multiple lattices)",
                        evidence=dict(extra_lattices=len(self.st.results["extra_refl"])))
            return
        f0, f1 = self.p.data_range
        wedges, a_ = [], None                              # contiguous runs of frames not excluded
        for fr in range(f0, f1 + 2):
            ok_ = fr <= f1 and not self.p.is_excluded(fr)
            if ok_ and a_ is None:
                a_ = fr
            elif not ok_ and a_ is not None:
                wedges.append((a_, fr - 1))
                a_ = None
        self.log(f"\n=== INTEGRATE (forward model) ===\n  FM: pixel-level joint fit over {len(wedges)} wedge(s)")
        env = os.environ.get
        # multi-lattice (XPG_FM_MULTI_LATTICE=1): all lattices' reflections in one model, fitted jointly (each with
        # its own predicted positions), results split back afterwards
        parts = [r] + ([extra[j] for j in sorted(extra)] if joint else [])
        # extra lattices: only their reflections with signal in the shoebox integration (I/sigma >= 3) enter the joint
        # model. Real extra lattices are mostly empty predictions; fitting them all shared the main lattice's photons
        # with empty partners.
        thr = 3.0
        subsel = [np.arange(len(r["I"]))] + [
            np.flatnonzero(np.isfinite(p_["I"]) & (np.asarray(p_["sig"]) > 0) & (p_["I"] / np.where(p_["sig"] > 0, p_["sig"], 1)
                                                                               >= thr)) for p_ in parts[1:]]
        fit_parts = [parts[0]] + [{k: (v[s_] if isinstance(v, np.ndarray) and v.shape[:1] == (len(p_["I"]),) else v)
                                   for k, v in p_.items()} for p_, s_ in zip(parts[1:], subsel[1:])]
        sizes = [len(p_["I"]) for p_ in fit_parts]
        if joint:
            keys = [k for k, v in r.items() if isinstance(v, np.ndarray) and v.shape[:1] == (sizes[0],)
                    and all(isinstance(p_.get(k), np.ndarray) and p_[k].shape[:1] == (n_,) for p_, n_ in
                            zip(fit_parts[1:], sizes[1:]))]
            rc = {k: np.concatenate([p_[k] for p_ in fit_parts]) for k in keys}
            if "block" in rc:              # the model's geometry per block: the main lattice's (same detector)
                rc["block"] = np.minimum(rc["block"], len(self.st.results["integration_blocks"]) - 1)
            self.log(f"  FM: {len(parts)} lattices fitted jointly ({sizes[0]} + " + " + ".join(
                f"{n_} of {len(p_['I'])}" for n_, p_ in zip(sizes[1:], parts[1:])) +
                f" reflections; extra lattices: shoebox I/sigma >= {thr:g})")
        else:
            rc = r
        res = integrate_sweep(rc, self.st.model, self.st.results["integration_blocks"], self.st.exp.geom, self.images,
                              wedges, mask=self.st.mask, bkg_template=self.st.bkg_template,
                              profiles=self.st.results.get("integration_profiles"),
                              box_n=self.st.results.get("integration_box_n", 5.0), window=env("XPG_FM_WINDOW", "auto"),
                              block_params=self.st.results.get("integration_block_params"),
                              mem_gb=float(env("XPG_FM_MEM_GB")) if env("XPG_FM_MEM_GB") else None,
                              tiles=env("XPG_FM_TILES", "auto"),
                              outer_tol=0.01, superpixel=float(env("XPG_FM_SUPERPIXEL", "0")),
                              widths=(("auto" if env("XPG_FM_WIDTHS") == "auto" else
                                       [float(v) for v in env("XPG_FM_WIDTHS").split(",")]) if env("XPG_FM_WIDTHS")
                                      else None), log=self.log,
                              I_start=(np.where(rc["fm_fitted"], rc["I"] / np.maximum(rc["rlp"], 1e-12), rc["iprf"])
                                       if "fm_fitted" in rc else None))
        bounds = np.r_[0, np.cumsum(sizes)]
        if auto and not self._fm_sane(r, {k: (v[:sizes[0]] if isinstance(v, np.ndarray) and v.shape[:1] == (bounds[-1],)
                                               else v) for k, v in res.items()}):
            return
        self.st.results["fm_used"] = True
        res_all = res
        for i_p in range(len(parts) - 1, -1, -1):                 # main lattice last: it defines the summary
            r = parts[i_p]
            sl = slice(bounds[i_p], bounds[i_p + 1])
            res = {k: (v[sl] if isinstance(v, np.ndarray) and v.shape[:1] == (bounds[-1],) else v)
                   for k, v in res_all.items()}
            if i_p > 0:                    # back to the lattice's full table (rows not fitted: not done)
                n_full = len(r["I"])

                def full(v):
                    if not (isinstance(v, np.ndarray) and v.shape[:1] == (sizes[i_p],)):
                        return v
                    out = np.zeros((n_full,) + v.shape[1:], v.dtype) if v.dtype.kind in "biu" else \
                        np.full((n_full,) + v.shape[1:], np.nan, v.dtype)
                    out[subsel[i_p]] = v
                    return out
                res = {k: full(v) for k, v in res.items()}
            self._fm_apply(r, res, env)
        ok = r["fm_fitted"]
        res = res_all
        self.st.results["forward_model"] = dict(fitted=int(ok.sum()), total=len(ok), seconds=res["seconds"],
                                                widths=res.get("widths"),
                                                window=res["window"], tiles=res["tiles"],
                                                superpixel=res.get("superpixel", 1),
                                                lattices=len(parts),
                                                overload_recovered=int(r["fm_overload_recovered"].sum()))
        self.log(f"  FM: {ok.sum()} of {len(ok)} reflections from the forward model ({100 * ok.mean():.1f}%), "
                 f"{r['fm_overload_recovered'].sum()} overloaded ones recovered; {res['seconds']:.0f} s")
        self.decide("integration", "forward model (pixel-level joint fit)", evidence=dict(
            fitted_fraction=float(ok.mean())))
        self.st.refl = r

    def _fm_sane(self, r, res, nbins=10):
        """Safety check of an automatic forward-model run against the shoebox intensities: in the outer shells the
        forward model's mean I/sigma must not turn negative where the shoebox's is positive."""
        ok = res["done"] & np.isfinite(res["I"]) & np.isfinite(res["sig"]) & (res["sig"] > 0) & \
            np.isfinite(r["I"]) & (r["sig"] > 0)
        if ok.sum() < 1000:
            return True
        d = d_spacing_from_exp(r["hkl"], self.st.exp)[ok]
        zf = (res["I"] / res["sig"])[ok]
        zs = (r["I"] / r["sig"])[ok]
        edges = np.quantile(1 / d ** 2, np.linspace(0, 1, nbins + 1))
        sh = np.clip(np.searchsorted(edges, 1 / d ** 2, side="right") - 1, 0, nbins - 1)
        bad = []
        for b in range(nbins - 3, nbins):
            m = sh == b
            mf, ms = float(np.mean(zf[m])), float(np.mean(zs[m]))
            if mf < 0.0 and ms > 0.2:
                bad.append((float(np.min(d[m])), mf, ms))
        if bad:
            self.log("  FM auto: outer shells negative in the forward model only (" + "; ".join(
                f"d > {dd:.2f} A: <I/sig> FM {a:+.2f} vs shoebox {b_:+.2f}" for dd, a, b_ in bad) +
                "): shoebox intensities kept")
            self.decide("integration", "shoebox (forward model failed the outer-shell check)",
                        evidence=dict(shells=bad))
            return False
        return True

    def _fm_apply(self, r, res, env):
        """Forward-model results of one lattice's table r (in place)."""
        ok = res["done"] & np.isfinite(res["I"]) & np.isfinite(res["sig"]) & (res["sig"] > 0)
        if "I_shoebox" not in r:
            r["I_shoebox"], r["sig_shoebox"] = r["I"].copy(), r["sig"].copy()
        r["I"] = np.where(ok, res["I"] * r["rlp"], r["I"])
        r["sig"] = np.where(ok, res["sig"] * r["rlp"], r["sig"])
        # recovered overloads: profile-model error (5% of I) on top of counting statistics
        rr = res["recovered"] & ok
        r["sig"] = np.where(rr, np.sqrt(r["sig"] ** 2 + (0.05 * r["I"]) ** 2), r["sig"])
        r["fm_fitted"] = ok
        r["fm_overload_recovered"] = res["recovered"] & ok
        # SCALE's profile / summation combination (summation for strong reflections, chosen by block Rmeas) stays on
        # with the FM as the profile estimate, but summation only for reflections without an overlap partner (crowded
        # data: the FM deconvolves them, the summation foreground holds the neighbour's counts). Without it the FM's
        # strong intensities depend on the profile shape: simulated Gaussian spots, merged R vs truth of the strongest
        # 10% 0.0133 (FM alone) vs 0.0032 (shoebox with the combination).
        r["sum_ok"] = ~np.asarray(r.get("overlap", np.zeros(len(ok), bool)), bool) | ~ok
        for k in ("I3", "sig3"):
            r.pop(k, None)

    def _fm_will_run(self):
        """Whether the forward model will replace the final integration: always with XPG_INTEGRATE_MODE=fm; in auto mode
        when pass 1 found the data crowded (overlap-partner fraction >= XPG_FM_AUTO_OVERLAP) with one lattice."""
        mode = _fm_mode()
        if mode != "auto":
            return mode == "fm"
        frac = self.st.results.get("overlap_fraction")
        extra = self.st.results.get("extra_refl") or {}
        joint = bool(extra) and os.environ.get("XPG_FM_MULTI_LATTICE", "0") == "1"
        return frac is not None and frac >= FM_AUTO_OVERLAP and (not extra or joint)

    def _pass2_needed(self, max_xy=0.5, max_z=0.3):
        """--pass2 auto: the second pass (post-refined geometry, adapted widths) is skipped only when the pass-1
        predictions of strong reflections are already within max_xy spot sigma / max_z rocking sigma (median) and
        the data have a single lattice."""
        pe = self.st.results.get("prediction_error_pass1")
        idx = self.st.indexing
        multi = bool(idx is not None and (getattr(idx, "extra_lattices", None) or getattr(idx, "domains", None)))
        if pe is None:
            self.log("  DECISION second pass: on (no pass-1 prediction error measured)")
            return True
        need = multi or pe[0] > max_xy or pe[1] > max_z
        self.log(f"  DECISION second pass: {'on' if need else 'skipped'} (pass-1 prediction error {pe[0]:.3f} spot sigma "
                 f"/ {pe[1]:.3f} rocking sigma; limits {max_xy} / {max_z}; "
                 f"{'several lattices / domains' if multi else 'one lattice'})")
        return need

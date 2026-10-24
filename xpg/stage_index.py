"""INDEX stage: indexing with retries (window, axis screen, perpendicular axis), lattice choice (superlattice, pseudo-
centring, user cell) and the indexed experiment."""
from __future__ import annotations

import time

import gemmi
import numpy as np

from .ice import ICE_RINGS
from .indexing import index_spots
from .integrate import IntegrationModel, integrate
from .io.legacy import write_xparm
from .io.xpgfiles import write_geometry
from .lattice import best_lattice_per_bravais, lattice_candidates
from .models import Crystal, Experiment

class IndexStage:
    """Pipeline mixin (see pipeline.Pipeline)."""

    def _early_large_cell(self, k, kw, strategies):
        """The uncapped max-cell estimate exceeds the 400 A cap: a capped FFT cannot find the long axis, so the 1000 A
        cap is tried next instead of after all other strategies."""
        from . import indexing as _ix
        raw = _ix.LAST_MAX_CELL_RAW
        if self.p.max_cell is None and k == 0 and raw is not None and raw > kw.get("max_cell_cap", 400.0) \
                and not any(s_.get("max_cell_cap", 0) >= 1000 for s_ in strategies[k + 1:k + 2]):
            self.log(f"  INDEX: uncapped max-cell estimate {raw:.0f} A exceeds the cap: trying the 1000 A cap next "
                     f"(3D FFT, then 1D FFTs over directions)")
            strategies.insert(k + 1, dict(max_cell_cap=1000.0, fft1d=True))
            strategies.insert(k + 1, dict(max_cell_cap=1000.0))

    def _window_retry(self, k, kw, res, strategies, upper):
        """Max-cell window too small. The default window is the smaller of two neighbour estimates: on the detector
        and in reciprocal space (split spots bias it high: 368 / 1005 / 866 A). Too small a window is fatal (FFT
        vectors only up to 1.2 x), too large one only coarsens the FFT. So when an attempt indexes poorly (< the
        required fraction) or its cell fills the window (longest axis > 0.9 x, the 1.5 x margin used up), the same
        strategy is retried next with twice the window, then with the reciprocal estimate (at least 2 x, at most
        1000 A); attempts at the 400 A cap are left to the cap retries. Returns k + 1 if a retry was queued."""
        from . import indexing as _ix
        p = self.p
        if p.max_cell is not None:
            return None
        mc = getattr(res, "max_cell", None) if res is not None else kw.get("max_cell")
        if mc is None:
            raw = _ix.LAST_MAX_CELL_RAW
            mc = None if raw is None else float(np.clip(raw, 30.0, kw.get("max_cell_cap", 400.0)))
        if mc is None or mc >= kw.get("max_cell_cap", 400.0) - 1e-6:
            return None
        if "hi" not in upper:
            rec = _ix.LAST_MAX_CELL_RECIP
            upper["hi"] = float(min(1000.0, max(2.0 * mc, rec if rec else 0.0)))
        # once the default window has failed the estimate is unreliable: a window retry must explain nearly all spots
        need = 0.9 if upper.get("n") else p.min_fraction_indexed
        if res is None or res.fraction < need:
            why = "failed" if res is None else f"indexed {100 * res.fraction:.0f}%"
        elif float(max(res.exp.crystal.cell[:3])) > 0.9 * mc and res.fraction < 0.9:
            why = f"longest cell axis {float(max(res.exp.crystal.cell[:3])):.1f} A fills the window"
        else:
            return None
        # 2 x first, then straight to the upper bracket (at most two extra attempts on genuinely poor data)
        w = min(2.0 * mc, upper["hi"]) if not upper.get("n") else upper["hi"]
        if w < 1.2 * mc:
            return None
        w = round(w, 1)
        upper["n"] = upper.get("n", 0) + 1
        self.log(f"  INDEX: max-cell window {mc:.0f} A: {why}; retrying with {w:.0f} A (neighbour estimates up to "
                 f"{upper['hi']:.0f} A)")
        strategies.insert(k + 1, dict(kw, max_cell=w, max_cell_cap=1000.0 if w > 400.0 else kw.get("max_cell_cap", 400.0)))
        return k + 1

    def _screen_axes(self, f0):
        """The first attempt indexed (almost) nothing: one quick attempt each with the reversed and the two
        perpendicular rotation axes (in the detector plane). Returns the axis that indexes clearly well (>= 30% and
        25 points above the given one) or None. A wrong header axis otherwise went through the full retries with
        the given and the reversed axis first (~100 s of attempts below 1%)."""
        from .geometry import unit
        g0, p = self.st.geom, self.p
        a0 = unit(np.asarray(g0.axis, float))
        perp = unit(np.cross(unit(np.asarray(g0.beam, float)), a0))
        quick = []
        for a in (-a0, perp, -perp):
            gf = g0.copy()
            gf.axis = a
            try:
                r = index_spots(gf, self.st.spots, max_cell=p.max_cell, refine_flags=p.refine_idxref,
                                min_fraction=p.min_fraction_indexed, log=self.log, verbose=False,
                                search_extra_lattices=False)
                quick.append((r.fraction, a))
            except (RuntimeError, np.linalg.LinAlgError, ValueError):
                quick.append((0.0, a))
        self.log("  INDEX: first attempt indexed {:.1f}%: rotation axes screened (one attempt each): ".format(100 * f0) +
                 "; ".join(f"{(np.round(a, 4) + 0.0).tolist()} {100 * fr:.0f}%" for fr, a in quick))
        fr, a = max(quick, key=lambda q: q[0])
        return a if fr >= 0.3 and fr > f0 + 0.25 else None

    def _index_with_retries(self, axis_screen=False):
        """If the indexed fraction is low, retry with other spot selections / resolution
        ranges and keep the best solution; fail loudly rather than integrate a wrong lattice. axis_screen: a
        (nearly) failed first attempt screens the other rotation axes first (_screen_axes) and restarts with the
        one that indexes."""
        p = self.p
        strategies = [dict(), dict(basis_rank=1), dict(basis_rank=2), dict(n_fft=5000), dict(dmin_index=5.0, n_fft=8000),
                      dict(dmin_index=2.5, n_fft=40000), dict(max_cell_cap=1000.0)]
        best, sols, window_retry, upper = None, [], None, {}
        for k, kw in enumerate(strategies):
            if k:
                self.log(f"  INDEX: retry {k} with {kw}")
            try:
                res = index_spots(self.st.geom, self.st.spots, dmin_index=kw.get("dmin_index"),
                                  max_cell=kw.get("max_cell", p.max_cell), refine_flags=p.refine_idxref,
                                  min_fraction=p.min_fraction_indexed, log=self.log,
                                  n_fft=kw.get("n_fft", 20000), max_cell_cap=kw.get("max_cell_cap", 400.0),
                                  basis_rank=kw.get("basis_rank", 0), search_extra_lattices=False,
                                  fft1d=kw.get("fft1d", False))
                cap = kw.get("max_cell_cap", 400.0)
                if p.max_cell is None and max(res.exp.crystal.cell[:3]) > 0.75 * cap and cap < 1000:
                    self.log(f"  INDEX: longest cell axis close to the max-cell cap ({cap:.0f} A): retrying with 1000 A")
                    strategies.insert(k + 1, dict(max_cell_cap=1000.0))
                window_retry = self._window_retry(k, kw, res, strategies, upper)
            except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
                self.log(f"  INDEX: attempt failed ({type(e).__name__}: {e})")
                if axis_screen and k == 0:
                    alt = self._switch_axis(0.0)
                    if alt is not None:
                        return alt
                self._early_large_cell(k, kw, strategies)
                window_retry = self._window_retry(k, kw, None, strategies, upper)
                continue
            if axis_screen and k == 0 and res.fraction < 0.05:
                alt = self._switch_axis(res.fraction)
                if alt is not None:
                    return alt
            if res.fraction < p.min_fraction_indexed:
                self._early_large_cell(k, kw, strategies)
            sols.append(res)
            if best is None or res.fraction > best.fraction + 0.02:
                best = res
            if window_retry == k + 1:
                continue
            if best.fraction >= p.min_fraction_indexed:
                break
            # three independent spot selections reproducing the best lattice (the acceptance rule below for
            # split / multi-crystal data): further strategies would only find it again
            if best.fraction < p.min_fraction_indexed and len(self._same_lattice(sols, best)) >= 3 \
                    and best.indexed.sum() >= 1000:
                self.log(f"  INDEX: lattice reproduced by {len(self._same_lattice(sols, best))} spot selections; "
                         f"no further retries")
                break
        if best is not None:
            # further lattices / weak domains: searched once, for the chosen solution only
            from .indexing import search_extra_lattices_fn
            best.extra_lattices, best.domains = search_extra_lattices_fn(
                best.exp, self.st.spots, best.indexed, getattr(best, "dmin_index", None), p.max_cell,
                n_fft=getattr(best, "n_fft", 20000), log=self.log)
        if best is not None:
            best.n_reproduced = len(self._same_lattice(sols, best)) if best.indexed.sum() >= 1000 else 0
        if best is not None and best.fraction < 0.2:
            # Accept when independent spot selections reproduce the same lattice (cell within 1%, orientation within
            # 0.5 deg) with enough indexed spots: a chance lattice is not reproducible.
            same = self._same_lattice(sols, best)
            if len(same) >= 3 and best.indexed.sum() >= 1000:
                self.log(f"  INDEX WARNING: only {100 * best.fraction:.0f}% of the spots belong to the main lattice, but "
                         f"{len(same)} independent spot selections give the same lattice ({best.indexed.sum()} spots): "
                         f"accepted (several crystals / split spots; check the unindexed spots)")
                return best
        if best is None or best.fraction < 0.2:
            raise RuntimeError("INDEX: indexing failed (best indexed fraction "
                               f"{0 if best is None else 100 * best.fraction:.0f}%); check geometry / spot range")
        return best

    def _switch_axis(self, f0):
        """_screen_axes, and on success the full retries with the new axis (None: keep the given axis)."""
        from .geometry import unit
        a = self._screen_axes(f0)
        if a is None:
            return None
        g0 = self.st.geom
        gf = g0.copy()
        gf.axis = a
        self.st.geom = gf
        try:
            res = self._index_with_retries()
        except RuntimeError:
            res = None
        if res is None or res.fraction <= f0 + 0.25:
            self.st.geom = g0
            return None
        a0 = unit(np.asarray(g0.axis, float))
        kind = "reversed" if float(np.dot(a, a0)) < -0.99 else "perpendicular"
        self.p.rotation_axis = np.asarray(a, float)
        self.decide("rotation axis", f"{kind}: {(np.round(a, 4) + 0.0).tolist()}", margin=res.fraction - f0,
                    unit="indexed fraction", alternatives=[f"as given {np.round(a0, 4).tolist()} ({100 * f0:.0f}%)"])
        self.log(f"  INDEX: rotation axis {(np.round(a, 4) + 0.0).tolist()} indexes {100 * res.fraction:.0f}% (as given: "
                 f"{100 * f0:.0f}% on the first attempt): used from here on")
        return res

    def _index_perpendicular_axis(self, res, f_best):
        """Neither sense of the given rotation axis indexes: try the perpendicular axis in the detector plane (both
        senses). Single-axis goniometers rotate about a horizontal or a vertical axis and mini-CBF headers record
        neither. One quick attempt per candidate; the full retries only for the best, kept if it indexes clearly
        more spots. Returns the result to use (``res`` unchanged when nothing is better)."""
        from .geometry import unit
        g0, p = self.st.geom, self.p
        a0 = unit(np.asarray(g0.axis, float))
        perp = unit(np.cross(unit(np.asarray(g0.beam, float)), a0))
        quick = []
        for a in (perp, -perp):
            gf = g0.copy()
            gf.axis = a
            try:
                r = index_spots(gf, self.st.spots, max_cell=p.max_cell, refine_flags=p.refine_idxref,
                                min_fraction=p.min_fraction_indexed, log=self.log, verbose=False,
                                search_extra_lattices=False)
                quick.append((r.fraction, a))
            except (RuntimeError, np.linalg.LinAlgError, ValueError):
                quick.append((0.0, a))
        self.log("  INDEX: perpendicular rotation axes (one attempt each): " +
                 "; ".join(f"{(np.round(a, 4) + 0.0).tolist()} {100 * fr:.0f}%" for fr, a in quick))
        fr, a = max(quick, key=lambda q: q[0])
        if fr <= f_best + 0.1:
            return res
        gf = g0.copy()
        gf.axis = a
        self.st.geom = gf
        try:
            res_p = self._index_with_retries()
        except RuntimeError:
            res_p = None
        if res_p is None or res_p.fraction <= f_best + 0.1:
            self.st.geom = g0
            return res
        self.p.rotation_axis = np.asarray(a, float)
        self.decide("rotation axis", f"perpendicular: {(np.round(a, 4) + 0.0).tolist()}", margin=res_p.fraction - f_best,
                    unit="indexed fraction", alternatives=[f"as given / reversed {np.round(a0, 4).tolist()} "
                                                           f"({100 * f_best:.0f}%)"])
        self.log(f"  INDEX: rotation axis {(np.round(a, 4) + 0.0).tolist()} indexes {100 * res_p.fraction:.0f}% (given axis, either "
                 f"sense: {100 * f_best:.0f}%): used from here on")
        return res_p

    @staticmethod
    def _same_lattice(sols, best):
        """Solutions with the best lattice: cell lengths within 1%, orientation within 0.5 deg."""
        from .indexing import _misorientation_sym
        return [r_ for r_ in sols if np.all(np.abs(r_.exp.crystal.cell[:3] - best.exp.crystal.cell[:3])
                                            < 0.01 * best.exp.crystal.cell[:3])
                and _misorientation_sym(r_.exp.crystal, best.exp.crystal) < 0.5]

    def job_index(self):
        self.log("\n=== INDEX ===")
        try:
            res, err = self._index_with_retries(axis_screen=True), None
        except RuntimeError as e:
            res, err = None, e
        # rotation sense: image headers (CBF mini-headers) do not record it and several beamlines rotate the other way
        # (ROTATION_AXIS= -1 0 0). With the wrong sense the spots' reciprocal positions are smeared and no lattice
        # indexes consistently: when indexing fails or is poor, the reversed axis is tried and kept only if it indexes
        # clearly more spots (multi-crystal data with a low fraction keep the given axis).
        sweep = (self.p.data_range[1] - self.p.data_range[0] + 1) * abs(self.st.geom.osc)
        axis_ok = res is not None and sweep >= 30.0 and (res.fraction >= 0.2 or getattr(res, "n_reproduced", 0) >= 3)
        if axis_ok and res.fraction < 0.5:
            self.log(f"  INDEX: {100 * res.fraction:.0f}% indexed"
                     + (f", lattice reproduced by {res.n_reproduced} spot selections" if res.fraction < 0.2 else "")
                     + f" over {sweep:.0f} deg: a wrong rotation sense / axis is ruled out (no axis tests)")
        if (res is None or res.fraction < 0.5) and not axis_ok:
            g0 = self.st.geom
            gf = g0.copy()
            gf.axis = -np.asarray(g0.axis, float)
            self.log("  INDEX: indexing " + ("failed" if res is None else f"indexed {100 * res.fraction:.0f}%") +
                     " with the rotation axis as given; trying the reversed rotation sense")
            self.st.geom = gf
            try:
                res_f = self._index_with_retries()
            except RuntimeError:
                res_f = None
            f0 = res.fraction if res is not None else 0.0
            if res_f is not None and res_f.fraction > f0 + 0.1:
                self.p.rotation_axis = -np.asarray(self.p.rotation_axis, float)
                self.decide("rotation axis", f"reversed: {np.round(gf.axis, 4).tolist()}", margin=res_f.fraction - f0,
                            unit="indexed fraction", alternatives=[f"as given {np.round(g0.axis, 4).tolist()} ({100 * f0:.0f}%)"])
                self.log(f"  INDEX: the reversed rotation axis indexes {100 * res_f.fraction:.0f}% (as given: {100 * f0:.0f}%): "
                         f"rotation axis {np.round(gf.axis, 4).tolist()} used from here on")
                res = res_f
            else:
                self.st.geom = g0
                self.log(f"  INDEX: the reversed rotation axis does not index better "
                         f"({'failed' if res_f is None else f'{100 * res_f.fraction:.0f}%'}): axis as given kept")
                res = self._index_perpendicular_axis(res, max(f0, res_f.fraction if res_f is not None else 0.0))
                if res is None:
                    raise err
        if self.st.geom.segmented:
            from .indexing import segment_offset_search
            g_seg, moved = segment_offset_search(res.exp.geom, self.st.spots, res.exp.crystal.Astar, log=self.log)
            if g_seg is not None:
                for attr in ("refine_idxref", "refine_integrate", "refine_correct"):
                    flags = getattr(self.p, attr)
                    if "SEGMENT" not in flags:
                        setattr(self.p, attr, list(flags) + ["SEGMENT"])
                self.log("  INDEX: re-indexing with corrected segment positions; SEGMENT added to REFINE(...) "
                         "keywords for this run")
                self.st.geom = g_seg
                res = index_spots(g_seg, self.st.spots, dmin_index=None, max_cell=self.p.max_cell,
                                  refine_flags=self.p.refine_idxref, min_fraction=self.p.min_fraction_indexed,
                                  log=self.log)
        self.st.indexing = res
        self.st.exp = res.exp
        self.st.spots.hkl = res.hkl
        self._write_spots()
        user_cell = self.p.space_group_number > 0 and self.p.unit_cell is not None
        if not user_cell and self._superlattice():
            res = self.st.indexing
        cands = lattice_candidates(res.exp.crystal.cell, max_obliq=self.p.lattice_tolerance)
        if not user_cell and self._pseudo_centring(cands):
            res = self.st.indexing
            cands = lattice_candidates(res.exp.crystal.cell, max_obliq=self.p.lattice_tolerance)
        self.st.results["lattice_candidates"] = cands
        try:
            from .indexing import classify_unindexed
            claimed, rows = classify_unindexed(res, self.st.spots, ICE_RINGS)
            if rows and (~claimed).mean() > 0.3:
                self.log(f"  INDEX: {100 * (~claimed).mean():.0f}% of the spots are not indexed by any lattice; relative to "
                         f"the main lattice they are:")
                for lab, cnt, f, ch in rows:
                    note = ""
                    if ch is not None:
                        note = f"   (chance {100 * ch:.1f}%" + (", EXCESS" if f > 2 * ch + 0.02 else "") + ")"
                    self.log(f"     {lab:52s} {cnt:8d} {100 * f:5.1f}%{note}")
                self.st.results["unindexed_classes"] = rows
        except Exception as e:  # pragma: no cover
            self.log(f"  INDEX: unindexed-spot classification failed ({e})")
        self.log("  INDEX: lattice candidates (best per Bravais type):")
        for c in best_lattice_per_bravais(cands):
            self.log(f"     {c.bravais:3s} {c.point_group:4s} penalty {c.penalty:6.2f}   cell "
                     f"{np.array2string(c.cell, precision=2, floatmode='fixed')}")
        # known cell + space group: integrate directly in that lattice with constrained cell
        if self.p.space_group_number > 0 and self.p.unit_cell is not None:
            cand = self._match_user_lattice(cands)
            if cand is not None:
                sg = gemmi.find_spacegroup_by_number(self.p.space_group_number)
                self._apply_lattice(cand, sg)
                self.log(f"  INDEX: using SPACE_GROUP_NUMBER={sg.number} ({sg.xhm()}) with cell "
                         f"{np.array2string(self.st.exp.crystal.cell, precision=3)} (penalty {cand.penalty:.2f})")
            else:
                self.log("  INDEX WARNING: UNIT_CELL_CONSTANTS / SPACE_GROUP_NUMBER do not match the indexed lattice "
                         "(within 3% / 3 deg); continuing with the reduced cell")
        write_geometry(self.wd / "xpg_geometry_indexed.json", self.st.exp,
                       self.st.space_group.number if self.st.space_group else 1, self.p.data_range, self.p.friedels_law)
        if self.p.export_legacy:
            write_xparm(self.wd / "XPARM.XDS", self.st.exp, self.st.space_group.number if self.st.space_group else 1)

    def _superlattice(self, tol=0.1, min_share=0.2, min_excess=10.0, min_gain=0.05):
        """Unindexed spots on fractional positions of the found lattice: for N = 2, 3 the class v in (Z/N)^3 holding
        most of them (> min_share of the unindexed spots, > min_excess x chance) defines the supercell whose
        reciprocal lattice adds v / N; it is adopted when re-indexing every spot with it raises the indexed
        fraction by > min_gain. Returns True if the indexing basis was replaced."""
        idx, sp = self.st.indexing, self.st.spots
        if idx is None or sp is None or getattr(idx, "indexed", None) is None or idx.indexed.size != len(sp):
            return False
        from .geometry import real_basis
        from .indexing import assign_indices, niggli_reduce
        exp = idx.exp
        g = exp.geom
        p0 = g.spot_to_p0(sp.x, sp.y, sp.z)
        claimed = idx.indexed.copy()
        for e in idx.extra_lattices:
            m = getattr(e, "spot_mask", None)
            if m is not None and m.size == claimed.size:
                claimed |= m
        U = ~claimed
        if U.sum() < 500:
            return False
        f = p0[U] @ real_basis(exp.crystal.Astar).T
        base_int = np.all(np.abs(f - np.round(f)) < tol, axis=1)
        # spots near integer positions of the found lattice are its own (misfits outside the indexing tolerance): the
        # share is taken among the spots the lattice does not explain (8T55, pseudo-hexagonal: 39% near-integer
        # misfits diluted the 1/2 class to 19.7% of all unindexed spots, 32% of the rest)
        n_rest = max(int((~np.all(np.abs(f - np.round(f)) < 0.25, axis=1)).sum()), 1)
        best = None
        for N in (2, 3):
            Nf = N * f
            near = np.all(np.abs(Nf - np.round(Nf)) < N * tol, axis=1) & ~base_int
            cls = np.mod(np.round(Nf[near]).astype(int), N)
            keys, cnt = np.unique(cls, axis=0, return_counts=True)
            if not len(keys):
                continue
            # a superlattice generated by v holds the classes k v (k = 1 .. N-1): counted under one canonical key
            tot = {}
            for kv, c in zip(map(tuple, keys), cnt):
                canon = min(tuple(int(x) for x in np.mod(np.array(kv) * j, N)) for j in range(1, N))
                tot[canon] = tot.get(canon, 0) + int(c)
            chance = U.sum() * (N - 1) * (2 * tol) ** 3
            for v, c in tot.items():
                share, excess = c / n_rest, c / max(chance, 1.0)
                if share > min_share and excess > min_excess and (best is None or c > best[2]):
                    best = (N, np.array(v), c, share, excess)
        if best is None:
            return False
        N, v, c, share, excess = best
        # reciprocal basis of base* + Z v/N: [v'/N, e_j, e_k] with v'_i = 1 (v' = k v mod N)
        for k in range(1, N):
            vp = np.mod(v * k, N)
            if 1 in vp:
                break
        i = int(np.flatnonzero(vp == 1)[0])
        Bm = np.eye(3)
        Bm[:, i] = vp / N
        Ast_new = exp.crystal.Astar @ Bm
        Rr, _ = niggli_reduce(real_basis(Ast_new))
        Ast_new = np.linalg.inv(Rr)
        _, ok_old = assign_indices(exp.crystal.Astar, p0, tol=0.15)
        _, ok_new = assign_indices(Ast_new, p0, tol=0.15)
        gain = float(ok_new.mean() - ok_old.mean())
        self.log(f"  INDEX: {100 * share:.0f}% of the unindexed spots lie on 1/{N}-fractional positions {v.tolist()} of the "
                 f"lattice ({excess:.0f}x chance): supercell with {N}x the volume indexes {100 * ok_new.mean():.1f}% of all "
                 f"spots vs {100 * ok_old.mean():.1f}%")
        adopt = gain > min_gain
        self.decide("superlattice", f"{N}x supercell (pseudo-translation)" if adopt else "lattice as indexed",
                    margin=gain - min_gain, unit="indexed fraction", low=abs(gain - min_gain) < 0.05,
                    evidence=dict(N=N, v=v.tolist(), share=share, excess=excess, gain=gain))
        if not adopt:
            return False
        from .refine import refine
        new = Experiment(g.copy(), Crystal.from_Astar(Ast_new, "a"))
        hkl, ok = assign_indices(Ast_new, p0, tol=0.3)
        try:
            rr = refine(new, hkl[ok], sp.x[ok], sp.y[ok], sp.z[ok], self.p.refine_idxref, reject=True, n_reject_cycles=2)
            new = rr.exp
            fin = np.zeros(len(sp), bool)
            fin[np.flatnonzero(ok)[rr.inliers]] = True
        except Exception as e:  # pragma: no cover
            self.log(f"  INDEX: supercell refinement failed ({e}); kept unrefined")
            fin = ok
        idx.exp = new
        idx.indexed = fin
        idx.hkl = np.where(fin[:, None], hkl, 0)
        idx.fraction = float(fin.mean())
        self.st.exp = new
        self.st.spots.hkl = idx.hkl
        self.st.results["superlattice"] = dict(N=N, v=v.tolist(), share=share, gain=gain)
        self.log(f"  INDEX: supercell adopted: {fin.sum()}/{len(sp)} spots indexed ({100 * fin.mean():.1f}%); reduced cell "
                 f"{np.array2string(new.crystal.cell, precision=2)}")
        return True

    def _pseudo_centring(self, cands, z_min=6.0, net_min=0.03, rel_min=0.3, rel_max=0.8, n_wedges=3, wedge_deg=5.0):
        """tNCS close to a centring translation makes the reflections that violate the centring so weak that spot
        finding sees none of them: indexing finds the centred sub-lattice and integration never measures the weak
        class. For each centred lattice candidate, the PRIMITIVE cell on its conventional axes is integrated over a
        few short wedges; the centring-violating positions hold only background if the centring is exact (as many
        below -3 sigma as above +3 sigma), so a significant excess of positive ones (z > z_min, net > net_min of
        the positions) means the centring is not exact and the primitive supercell is the lattice. Returns True
        when the indexing basis was replaced."""
        if not getattr(self.p, "pseudo_centring_check", True) or self.st.indexing is None:
            return False
        from .indexing import niggli_reduce
        from .geometry import real_basis
        idx = self.st.indexing
        exp = idx.exp
        g = exp.geom
        tests, seen = [], set()
        # the C-centred settings of an acceptable hexagonal / tetragonal lattice (orthohexagonal a x a sqrt3, the
        # diagonal a sqrt2 cell) follow from its metric
        hex_tet = any(c.bravais in ("hP", "tP") and c.penalty <= self.p.lattice_tolerance for c in cands)
        # likewise the rhombohedral settings of an acceptable cubic lattice (hexagonal axes along a body diagonal)
        cubic = any(c.bravais[0] == "c" and c.penalty <= self.p.lattice_tolerance for c in cands)
        for c in sorted(cands, key=lambda c: (-len(c.ops), c.penalty)):
            cen = c.bravais[1]
            if cen not in "CIRFAB" or c.penalty > self.p.lattice_tolerance or (hex_tet and cen == "C") or \
                    (cubic and c.bravais == "hR"):
                continue
            # the test integrates the conventional axes as a primitive lattice: candidates with the same conventional
            # basis (up to sign and order) are the same test
            Rb = real_basis(exp.crystal.Astar @ np.linalg.inv(c.P.T))
            Rb = np.array([v if v[np.argmax(np.abs(v))] > 0 else -v for v in Rb])
            key = (cen, tuple(sorted(tuple(np.round(v * 2) / 2) for v in Rb)))
            if key in seen:
                continue
            seen.add(key)
            tests.append(c)
        if not tests:
            return False
        f0, f1 = self.p.data_range
        nfr = max(int(round(wedge_deg / abs(g.osc))), 5)
        starts = [f0 + int(i * (f1 - f0 + 1 - nfr) / max(n_wedges - 1, 1)) for i in range(n_wedges)]
        dmax, dmin = self.p.include_resolution
        dmin = max(dmin, g.max_resolution_at_edge(), 1.5) if dmin > 0 else max(g.max_resolution_at_edge(), 1.5)
        # fixed widths, as its thresholds were calibrated: given widths change the boxes and the leakage of allowed
        # neighbours into violating positions
        model = IntegrationModel(sigma_d=0.05, sigma_m=0.25,
                                 gain=self.st.gain)
        for c in tests[:3]:
            Ast_c = exp.crystal.Astar @ np.linalg.inv(c.P.T)          # conventional axes, treated as primitive
            exp_c = Experiment(g.copy(), Crystal.from_Astar(Ast_c, "a"))
            # typical cells unaffected
            vol_c = 1.0 / abs(np.linalg.det(Ast_c))
            d_work = (2.0 * wedge_deg / 360.0 * 4.0 / 3.0 * np.pi * vol_c / 2e5) ** (1.0 / 3.0)
            dmin_c = max(dmin, d_work)
            cen_ops = [np.array(t, int) for t in gemmi.SpaceGroup(c.space_group).operations().cen_ops]
            iz, viol, dd = [], [], []
            t_pc = time.time()
            for a in starts:
                b = min(a + nfr - 1, f1)
                try:
                    rr = integrate(self.images, [exp_c], np.array([a - 1.0, float(b)]), model, dmin_c, dmax,
                                   frames=(max(f0, a - 5), min(f1, b + 5)), mask=self.st.mask, verbose=False,
                                   learn_profiles=False, adaptive=False, bkg_template=self._background_template(),
                                   moments_only=True).refl
                except Exception as e:  # pragma: no cover
                    self.log(f"  INDEX: pseudo-centring test failed ({type(e).__name__}: {e})")
                    return False
                ok = (np.isfinite(rr["isum"]) & (rr["vsum"] > 0) & (rr["frac_rec"] > 0.9) & ~rr["overlap"] &
                      (rr["z"] >= a - 1) & (rr["z"] < b))
                h = np.asarray(rr["hkl"], int)[ok]
                v = np.zeros(len(h), bool)
                for t in cen_ops:
                    v |= (h @ t) % gemmi.Op.DEN != 0
                iz.append(rr["isum"][ok] / np.sqrt(rr["vsum"][ok]))
                viol.append(v)
                dd.append(1.0 / np.linalg.norm(h @ Ast_c.T, axis=1))
            iz, viol, dd = np.concatenate(iz), np.concatenate(viol), np.concatenate(dd)
            if viol.sum() < 200 or (~viol).sum() < 200:
                continue
            npos, nneg = int(np.sum(iz[viol] > 3)), int(np.sum(iz[viol] < -3))
            z = (npos - nneg) / np.sqrt(npos + nneg + 1.0)
            net = (npos - nneg) / viol.sum()
            allowed = float(np.mean(iz[~viol] > 3))
            # intensity of the allowed neighbours leaks into the boxes of the violating positions
            rel = net / max(allowed, 1e-6)
            # a missed pseudo-centring is a WEAK violating class: one about as strong as the allowed class would have
            # been found by spot finding and indexed (or caught by the superlattice test from the unindexed spots).
            # Near-equal intensity at the violating positions is the neighbours' tails on crowded data
            leak = rel > rel_max
            exact = leak or not (z > z_min and net > net_min and rel > rel_min)
            self.log(f"  INDEX: pseudo-centring test {c.bravais} ({c.bravais[1]}-centring, d >= {dmin_c:.2f} A, "
                     f"{time.time() - t_pc:.0f} s): {int(viol.sum())} centring-violating "
                     f"positions, {100 * net:.1f}% net above 3 sigma (z {z:.1f}; allowed class {100 * allowed:.0f}%, ratio {rel:.2f}) -> "
                     f"{'violating class as strong as the allowed one: neighbour leakage (crowded spots), centring kept' if leak else 'centring exact' if exact else 'centring NOT exact: primitive supercell used (tNCS / pseudo-centring)'}")
            self.decide("pseudo-centring", "exact centring" if exact else f"primitive cell on the {c.bravais} axes",
                        margin=rel - rel_min, unit="relative to the allowed class", low=abs(rel - rel_min) < 0.1,
                        evidence=dict(n_violating=int(viol.sum()), net=net, z=z, allowed_frac=allowed, ratio=rel))
            if exact:
                continue
            Rr, _ = niggli_reduce(real_basis(Ast_c))
            Ast_r = np.linalg.inv(Rr)
            new = Experiment(g.copy(), Crystal.from_Astar(Ast_r, "a"))
            idx.exp = new
            self.st.exp = new
            self.st.results["pseudo_centring"] = dict(bravais=c.bravais, n_violating=int(viol.sum()), net=net, z=z)
            self.log(f"  INDEX: reduced cell now {np.array2string(new.crystal.cell, precision=2)} "
                     f"({abs(round(np.linalg.det(c.P)))}x the volume)")
            return True
        return False

    def _match_user_lattice(self, cands, tol_len=0.03, tol_ang=3.0):
        want = gemmi.find_spacegroup_by_number(self.p.space_group_number)
        uc = np.array(self.p.unit_cell, float)
        best = None
        for c in cands:
            if gemmi.SpaceGroup(c.space_group).point_group_hm() != want.point_group_hm() or c.bravais[1] != want.hm[0]:
                continue
            dl = np.abs(c.cell[:3] - uc[:3]) / uc[:3]
            da = np.abs(c.cell[3:] - uc[3:])
            if dl.max() <= tol_len and da.max() <= tol_ang:
                score = dl.sum() + da.sum() / 57.3
                if best is None or score < best[0]:
                    best = (score, c)
        return best[1] if best else None

    def _apply_lattice(self, cand, sg, refl=None):
        """Convert the experiment (and optionally reflections) to the conventional cell of ``cand``,
        then apply REIDX if given."""
        exp = self.st.exp
        Astar_c = exp.crystal.Astar @ np.linalg.inv(cand.P.T)
        if refl is not None:
            refl["hkl"] = refl["hkl"] @ cand.P
        if self.p.reidx is not None:
            M = np.round(self.p.reidx[:, :3]).astype(int)
            if round(abs(np.linalg.det(M))) >= 1:
                Astar_c = Astar_c @ np.linalg.inv(M)
                if refl is not None:
                    refl["hkl"] = refl["hkl"] @ M.T
                self.log(f"  REIDX applied: {M.tolist()}")
        system = cand.system
        self.st.exp = Experiment(exp.geom.copy(), Crystal.from_Astar(Astar_c, system, space_group=sg.xhm()))
        self.st.lattice = cand
        self.st.space_group = sg

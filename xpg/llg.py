"""Log-likelihood gain of a placed model against merged amplitudes.

sigma_A likelihood (Read 1986; Read 2001 for the LLG against the Wilson null): normalised amplitudes E_o (observed)
and E_c (model, gemmi structure factors), both normalised in resolution shells; per resolution bin a sigma_A = D
refined by maximum likelihood; measurement error enters as extra variance (sigma_E^2 = sigma_F^2 / (eps Sigma)),
so that D = 0 gives exactly the Wilson null (LLG 0):

  acentric  p(E_o | E_c) = 2 E_o / V exp(-(E_o^2 + D^2 E_c^2) / V) I0(2 D E_o E_c / V),   V = 1 - D^2 + sigma_E^2
  centric   p(E_o | E_c) = sqrt(2 / (pi V)) exp(-(E_o^2 + D^2 E_c^2) / (2 V)) cosh(D E_o E_c / V)
  null      the same with D = 0

LLG = sum over reflections of log p(E_o | E_c) - log p_null(E_o). The model is not moved or refined (a common
solution scored against several data sets). Flat bulk solvent (mask of the model, k_sol / B_sol fitted at low
resolution) with data to d_max 20 A; without it (solvent=False) data below 6 A are left out.

Data pathologies: reflections the Wilson distribution makes implausible (p < 1e-6, after removing anisotropy and a
tNCS modulation) are left out; with a twin law and fraction alpha the model is compared through its twinned
intensities (1 - alpha) |F_c(h)|^2 + alpha |F_c(T h)|^2. tNCS-related copies are modelled by placing them as pairs
(xpg.mrsearch), so the data keep their modulation here.

    from xpg.llg import placed_model_llg
    res = placed_model_llg("data.mtz", "model.pdb")          # F / SIGF columns found by name
    python -m xpg.llg data.mtz model.pdb [--dmin 2.0] [--dmax 6.0]"""
from __future__ import annotations

import argparse

import gemmi
import numpy as np
from scipy.special import i0e


def _shells(s2, nbins):
    edges = np.quantile(s2, np.linspace(0, 1, nbins + 1))
    return np.clip(np.searchsorted(edges, s2, side="right") - 1, 0, nbins - 1)


def _shell_mean(x, sh, nbins):
    n = np.bincount(sh, minlength=nbins)
    return np.bincount(sh, x, nbins) / np.maximum(n, 1)


def _lookup(mi, vals, hkl):
    key = lambda h: ((h[:, 0] + 512) * 1024 + (h[:, 1] + 512)) * 1024 + (h[:, 2] + 512)
    k_c, k_o = key(np.asarray(mi, np.int64)), key(hkl)
    o = np.argsort(k_c)
    pos = np.clip(np.searchsorted(k_c[o], k_o), 0, len(k_c) - 1)
    found = k_c[o][pos] == k_o
    return np.where(found, vals[o][pos], np.nan + 0j)


def model_fcalc(model, cell_sg_from, hkl, dmin, solvent=True, sg=None):
    """Complex F_calc of the model (gemmi density + FFT, density blur removed) and of a flat solvent mask (1 outside
    the model, standard radii) at the given Miller indices: (F_model, F_mask or None)."""
    st = gemmi.read_structure(str(model)) if not isinstance(model, gemmi.Structure) else model
    st.setup_entities()
    st.cell = cell_sg_from.cell
    st.spacegroup_hm = (sg or cell_sg_from.spacegroup).hm
    dc = gemmi.DensityCalculatorX()
    dc.d_min = dmin
    dc.rate = 1.5
    dc.set_grid_cell_and_spacegroup(st)
    dc.set_refmac_compatible_blur(st[0])
    dc.put_model_density_on_grid(st[0])
    fg = gemmi.transform_map_to_f_phi(dc.grid)
    asu = fg.prepare_asu_data(dmin=dmin * 0.999, with_000=False, with_sys_abs=False)
    inv_d2 = np.asarray(asu.make_1_d2_array(), float)
    Fm = np.asarray(asu.value_array) * np.exp(dc.blur * inv_d2 / 4.0)          # de-blur
    F_model = _lookup(asu.miller_array, Fm, hkl)
    if not solvent:
        return F_model, None
    mg = gemmi.FloatGrid()
    mg.setup_from(st, spacing=min(0.6, dmin / 3.0))
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Refmac)
    masker.put_mask_on_float_grid(mg, st[0])                 # 1 = solvent, 0 = model
    ma = gemmi.transform_map_to_f_phi(mg).prepare_asu_data(dmin=dmin * 0.999, with_000=False, with_sys_abs=False)
    return F_model, _lookup(ma.miller_array, np.asarray(ma.value_array), hkl)


def _fit_solvent(Fo, Fm, Fs, s2, eps, nbins=8):
    """Flat bulk solvent F = F_model + k exp(-B s^2 / 4) F_mask: (k, B) maximising the mean per-shell correlation
    of F_o^2 with |F|^2 at low resolution (d > 4 A; per-shell scales are free, as in the normalisation)."""
    low = s2 < 1.0 / 16.0
    if low.sum() < 200:
        return 0.0, 0.0
    sh = _shells(s2[low], nbins)
    best = (-np.inf, 0.0, 0.0)
    for k in np.arange(0.0, 0.61, 0.025):
        for B in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0):
            Fc2 = np.abs(Fm[low] + k * np.exp(-B * s2[low] / 4.0) * Fs[low]) ** 2 / eps[low]
            cc = np.mean([np.corrcoef(Fo[low][sh == b] ** 2 / eps[low][sh == b], Fc2[sh == b])[0, 1]
                          for b in range(nbins) if (sh == b).sum() > 20])
            if cc > best[0]:
                best = (cc, k, B)
    return best[1], best[2]


def _aniso_factor(hkl, cell, Eo2, cen):
    """Anisotropic normalisation of the observed E^2: mean
    m(h) = exp(c - s^T beta s) with s = B* h (Cartesian reciprocal vector), beta symmetric (6), fitted by the Wilson
    likelihood (acentric E^2 ~ Exp(m), centric E^2 ~ m chi^2_1). Returns m (E_o^2 / m is isotropic)."""
    from scipy.optimize import minimize
    Bm = np.array(cell.frac.mat.tolist()).T            # s = B* h: rows of the fractionalisation matrix
    s = hkl.astype(float) @ Bm.T
    Q = np.stack([s[:, 0] ** 2, s[:, 1] ** 2, s[:, 2] ** 2, 2 * s[:, 0] * s[:, 1], 2 * s[:, 0] * s[:, 2],
                  2 * s[:, 1] * s[:, 2]], 1)
    w = np.where(cen, 0.5, 1.0)

    def f(p):
        lm = p[0] - Q @ p[1:]
        e = Eo2 * np.exp(-lm)
        return float(np.sum(w * (lm + e))), np.concatenate([[np.sum(w * (1 - e))], -(w * (1 - e)) @ Q])
    r = minimize(f, np.zeros(7), jac=True, method="L-BFGS-B")
    return np.exp(r.x[0] - Q @ r.x[1:])


def tncs_modulation(hkl, E2, s2, t, nbins=10):
    """Expected-intensity modulation of tNCS with fractional translation t: m(h) = 1 + f cos(2 pi h.t), f fitted per
    resolution shell to the normalised intensities (least squares of E^2 - 1 on the cosine; 0 <= f <= 0.95)."""
    c = np.cos(2 * np.pi * (hkl @ np.asarray(t, float)))
    sh = _shells(s2, nbins)
    f = np.bincount(sh, (E2 - 1.0) * c, nbins) / np.maximum(np.bincount(sh, c * c, nbins), 1e-12)
    return 1.0 + np.clip(f, 0.0, 0.95)[sh] * c


def wilson_outliers(E2, cen, p=1e-6):
    """Normalised intensities with a Wilson probability below p: acentric E^2 > -ln p, centric E^2 > 2 erfc^-1(p)^2
    (13.8 and 23.9 for p = 1e-6)."""
    from scipy.special import erfcinv
    return np.where(cen, E2 > 2.0 * erfcinv(p) ** 2, E2 > -np.log(p))


def twin_mates(hkl, sg, law):
    """Index of each reflection's twin mate (law: twin operator triplet acting on hkl) in hkl, -1 where missing."""
    from .symops import apply_rot, hkl_key, op_matrix, to_asu
    h2, _, _ = to_asu(apply_rot(hkl, op_matrix(gemmi.Op(law))), sg)
    lut = {k: i for i, k in enumerate(hkl_key(to_asu(hkl, sg)[0]).tolist())}
    return np.array([lut.get(int(k), -1) for k in hkl_key(h2)])


def _llg_terms(Eo, Ec, sE2, cen, D):
    """Per-reflection log p(E_o | E_c; D) - log p_null(E_o) for one D (scalar or per reflection)."""
    V = 1.0 - D * D + sE2
    V0 = 1.0 + sE2
    xa = 2.0 * D * Eo * Ec / V
    la = -np.log(V) - (Eo ** 2 + (D * Ec) ** 2) / V + np.log(i0e(xa)) + xa
    la0 = -np.log(V0) - Eo ** 2 / V0
    xc = D * Eo * Ec / V
    lc = -0.5 * np.log(V) - (Eo ** 2 + (D * Ec) ** 2) / (2 * V) + xc + np.log1p(np.exp(-2 * xc)) - np.log(2.0)
    lc0 = -0.5 * np.log(V0) - Eo ** 2 / (2 * V0)
    return np.where(cen, lc - lc0, la - la0)


def _sigmaa_fit(Eo, Ec, sE2, cen, sh, nbins, grid=np.linspace(0.0, 0.99, 100)):
    """Per-bin maximum-likelihood sigma_A (= D): grid maximum, then golden-section refinement, all bins at once.
    Returns (D per bin, LLG per bin)."""
    tot = np.empty((len(grid), nbins))
    for g0 in range(0, len(grid), 20):                         # (grid chunk, reflections) term matrices
        T = _llg_terms(Eo[None], Ec[None], sE2[None], cen[None], grid[g0:g0 + 20, None])
        for j, t in enumerate(T):
            tot[g0 + j] = np.bincount(sh, t, nbins)
    k = np.argmax(tot, 0)
    lo, hi = grid[np.maximum(k - 1, 0)], grid[np.minimum(k + 1, len(grid) - 1)]
    for _ in range(30):
        a = hi - 0.618 * (hi - lo)
        c = lo + 0.618 * (hi - lo)
        fa = np.bincount(sh, _llg_terms(Eo, Ec, sE2, cen, a[sh]), nbins)
        fc = np.bincount(sh, _llg_terms(Eo, Ec, sE2, cen, c[sh]), nbins)
        up = fa > fc
        hi, lo = np.where(up, c, hi), np.where(up, lo, a)
    D = 0.5 * (lo + hi)
    return D, np.bincount(sh, _llg_terms(Eo, Ec, sE2, cen, D[sh]), nbins)


class LLGTarget:
    """sigma_A LLG against fixed data for many models (rigid-body refinement of a placed copy). The data side
    (normalisation, anisotropy, epsilon / centric) and the bulk-solvent model (mask, k_sol / B_sol) of a reference
    model are set up once; evaluate(F) scores complex model structure factors at self.hkl (without solvent).
    tncs: fractional tNCS translation (for the outlier test); twin: dict(law=triplet, alpha=fraction)."""

    def __init__(self, data, model, dmin=None, dmax=None, nbins=20, f_label=None, sigf_label=None, solvent=True,
                 aniso=True, sg=None, tncs=None, twin=None):
        mtz = gemmi.read_mtz_file(str(data))
        labels = [c.label for c in mtz.columns]
        fl = f_label or next(l for l in ("F", "FP", "F-obs", "F-obs-filtered", "FOBS") if l in labels)
        sl = sigf_label or next(l for l in ("SIG" + fl, "SIGF", "SIGFP", "SIGF-obs", "SIGF-obs-filtered") if l in labels)
        hkl = np.asarray(mtz.make_miller_array(), np.int64)
        F = np.asarray(mtz.column_with_label(fl), float)
        sF = np.asarray(mtz.column_with_label(sl), float)
        d = np.asarray(mtz.make_d_array(), float)
        sg = gemmi.SpaceGroup(sg) if isinstance(sg, str) else (sg or mtz.spacegroup)
        if dmax is None:
            dmax = 20.0 if solvent else 6.0
        ok = np.isfinite(F) & np.isfinite(sF) & (F > 0) & (d <= dmax)
        if dmin is not None:
            ok &= d >= dmin
        hkl, F, sF, d = hkl[ok], F[ok], sF[ok], d[ok]
        self.dmin_ = float(d.min())
        Fm, Fs = model_fcalc(model, mtz, hkl, self.dmin_, solvent=solvent, sg=sg)
        ok = np.isfinite(Fm) & (np.abs(Fm) > 0)
        if Fs is not None:
            ok &= np.isfinite(Fs)
        hkl, F, sF, d, Fm = hkl[ok], F[ok], sF[ok], d[ok], Fm[ok]
        eps = np.array([sg.operations().epsilon_factor(list(h)) for h in hkl], float)
        cen = np.array([sg.operations().is_reflection_centric(list(h)) for h in hkl], bool)
        s2 = 1.0 / d ** 2
        self.k_sol, self.B_sol = 0.0, 0.0
        self.F_sol = np.zeros(len(F), complex)
        if Fs is not None:
            Fs = Fs[ok]
            self.k_sol, self.B_sol = _fit_solvent(F, Fm, Fs, s2, eps)
            self.F_sol = self.k_sol * np.exp(-self.B_sol * s2 / 4.0) * Fs
        # normalisation in fine shells (smooth Sigma(s)); <E^2> = 1 per shell for both
        self.nfine = max(nbins, min(60, len(F) // 500))
        self.shf = _shells(s2, self.nfine)
        So = _shell_mean(F ** 2 / eps, self.shf, self.nfine)[self.shf]
        Eo = F / np.sqrt(eps * So)
        sE2 = sF ** 2 / (eps * So)
        if aniso:
            m = _aniso_factor(hkl, mtz.cell, Eo ** 2, cen)
            Eo, sE2 = Eo / np.sqrt(m), sE2 / m
        E2 = Eo ** 2 / (tncs_modulation(hkl, Eo ** 2, s2, tncs) if tncs is not None else 1.0)
        keep = ~wilson_outliers(E2, cen)
        self.n_outliers = int((~keep).sum())
        hkl, F, d, s2, eps, cen, Eo, sE2, Fm = (x[keep] for x in (hkl, F, d, s2, eps, cen, Eo, sE2, Fm))
        self.F_sol, self.shf = self.F_sol[keep], self.shf[keep]
        self.mtz, self.sg, self.hkl, self.F, self.d, self.s2 = mtz, sg, hkl, F, d, s2
        self.eps, self.cen, self.Eo, self.sE2, self.nbins = eps, cen, Eo, sE2, nbins
        self.sh = _shells(s2, nbins)
        self.F_model = Fm
        self.twin_alpha, self.twin_idx = 0.0, None
        if twin is not None:
            self.twin_alpha, self.twin_idx = float(twin.get("alpha", 0.0)), twin_mates(hkl, sg, twin["law"])

    def calc_amplitudes(self, Fm=None):
        """|F_calc| with bulk solvent at self.hkl; twinned: sqrt((1 - alpha) |F(h)|^2 + alpha |F(T h)|^2)."""
        Fm = self.F_model if Fm is None else np.nan_to_num(Fm)
        Fc2 = np.abs(Fm + self.F_sol) ** 2
        if self.twin_idx is not None:
            j = self.twin_idx
            Fc2 = (1 - self.twin_alpha) * Fc2 + self.twin_alpha * np.where(j >= 0, Fc2[np.maximum(j, 0)], Fc2)
        return np.sqrt(Fc2)

    def evaluate(self, Fm=None, full=False):
        """LLG of model structure factors Fm at self.hkl (default: the reference model's); full: the result dict."""
        Fc = self.calc_amplitudes(Fm)
        Sc = _shell_mean(Fc ** 2 / self.eps, self.shf, self.nfine)[self.shf]
        Ec = Fc / np.sqrt(self.eps * np.maximum(Sc, 1e-30))
        D, llg_bin = _sigmaa_fit(self.Eo, Ec, self.sE2, self.cen, self.sh, self.nbins)
        if not full:
            return float(llg_bin.sum())
        d_bin = np.array([1.0 / np.sqrt(np.median(self.s2[self.sh == b])) for b in range(self.nbins)])
        return dict(LLG=float(llg_bin.sum()), n=int(len(self.F)), sigmaA=D, d_bin=d_bin, LLG_bin=llg_bin,
                    cc=float(np.corrcoef(self.Eo, Ec)[0, 1]), dmin=self.dmin_, dmax=float(self.d.max()),
                    k_sol=self.k_sol, B_sol=self.B_sol)


def model_r_factors(data, model, sg=None, dmax=None, twin=None):
    """R and R-free of a placed model as it is (no atomic refinement): F_calc with a flat bulk solvent (twinned with
    twin = dict(law, alpha)), an overall scale and B fitted to the working set. Free reflections are those with
    FreeR_flag 0. Returns (R_work, R_free)."""
    t = LLGTarget(data, model, dmax=dmax, sg=sg, aniso=False, twin=twin)
    mtz = t.mtz
    flags = np.full(len(t.hkl), -1)
    if "FreeR_flag" in [c.label for c in mtz.columns]:
        key = lambda h: (h[:, 0] + 512) * 1048576 + (h[:, 1] + 512) * 1024 + (h[:, 2] + 512)
        allh = np.asarray(mtz.make_miller_array(), np.int64)
        fr = np.asarray(mtz.column_with_label("FreeR_flag"), float)
        order = np.argsort(key(allh))
        pos = np.searchsorted(key(allh)[order], key(t.hkl))
        flags = fr[order][np.clip(pos, 0, len(order) - 1)]
    free = flags == 0
    Fo, Fc = t.F, t.calc_amplitudes()
    ok = (Fc > 0) & ~free
    A = np.c_[np.ones(ok.sum()), -t.s2[ok] / 4.0]
    lnk, B = np.linalg.lstsq(A, np.log(Fo[ok] / Fc[ok]), rcond=None)[0]
    Fs = np.exp(lnk - B * t.s2 / 4.0) * Fc
    r = lambda m: float(np.sum(np.abs(Fo[m] - Fs[m])) / np.sum(Fo[m])) if m.any() else float("nan")
    return r(~free), r(free)


def refine_twin_fraction(data, model, law, sg=None, grid=np.arange(0.0, 0.5, 0.025)):
    """Twin fraction of the law that maximises the LLG of a placed model (0 included: no twinning). Returns
    (alpha, LLG gain over alpha = 0)."""
    t = LLGTarget(data, model, sg=sg, twin=dict(law=law, alpha=0.0))
    ll = []
    for a in grid:
        t.twin_alpha = float(a)
        ll.append(t.evaluate())
    k = int(np.argmax(ll))
    return float(grid[k]), float(ll[k] - ll[0])


def placed_model_llg(data, model, dmin=None, dmax=None, nbins=20, f_label=None, sigf_label=None, solvent=True,
                     aniso=True, sg=None, tncs=None, twin=None):
    """data: merged MTZ path with amplitudes (F / SIGF, FP / SIGFP, F-obs / SIGF-obs, ...). Returns dict(LLG,
    n, sigmaA per bin with its resolution, LLG per bin, CC(E_o, E_c)).
    sg: space group of the model's packing (default: the data's), e.g. an enantiomorph or another screw-axis
    alternative of the same point group."""
    return LLGTarget(data, model, dmin, dmax, nbins, f_label, sigf_label, solvent, aniso, sg, tncs,
                     twin).evaluate(full=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="LLG of a placed model against merged amplitudes")
    ap.add_argument("data")
    ap.add_argument("model")
    ap.add_argument("--dmin", type=float, default=None)
    ap.add_argument("--dmax", type=float, default=None, help="default 20 A with bulk solvent, 6 A without")
    ap.add_argument("--no-solvent", action="store_true")
    ap.add_argument("--bins", type=int, default=20)
    a = ap.parse_args(argv)
    r = placed_model_llg(a.data, a.model, a.dmin, a.dmax, a.bins, solvent=not a.no_solvent)
    print(f"LLG {r['LLG']:.1f} from {r['n']} reflections ({r['dmax']:.2f}-{r['dmin']:.2f} A); CC(E_o, E_c) {r['cc']:.4f}; "
          f"bulk solvent k {r['k_sol']:.3f} B {r['B_sol']:.0f}")
    print("  d (A)   sigmaA   LLG")
    for dd, s, l in zip(r["d_bin"], r["sigmaA"], r["LLG_bin"]):
        print(f"  {dd:6.2f}  {s:6.3f}  {l:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

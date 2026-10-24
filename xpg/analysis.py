"""Data analysis: Wilson statistics, twinning tests, per-batch statistics, indexing ambiguity
resolution against a reference, free-R flag transfer."""
from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np

from .symops import (apply_rot, as_spacegroup, d_spacing, hkl_key, key_hkl, op_matrix, reflection_props,
                     rot_matrices, to_asu)


# ---------------------------------------------------------------------------
# Wilson / twinning
# ---------------------------------------------------------------------------

@dataclass
class TwinningReport:
    wilson_b: float
    i2_over_i_sq: float          # acentric <I^2>/<I>^2 : 2.0 untwinned, 1.5 perfect twin
    mod_e2_minus_1: float        # acentric <|E^2-1|>   : 0.736 untwinned, 0.541 perfect twin
    l_mean: float                # <|L|> : 0.5 untwinned, 0.375 perfect twin (Padilla & Yeates 2003)
    l2_mean: float               # <L^2> : 1/3 untwinned, 1/5 perfect twin
    twin_laws: list              # [(operator, <H>, alpha_H, n)]
    dmin_used: float

    def lines(self):
        out = [f"  Wilson B = {self.wilson_b:.1f} A^2 (fit 4.5 A - {self.dmin_used:.2f} A)",
               f"  acentric <I^2>/<I>^2 = {self.i2_over_i_sq:.3f} (untwinned 2.0, perfect twin 1.5); "
               f"<|E^2-1|> = {self.mod_e2_minus_1:.3f} (0.736 / 0.541)",
               f"  L-test: <|L|> = {self.l_mean:.3f} (0.500 untwinned, 0.375 perfect twin), <L^2> = {self.l2_mean:.3f}"]
        if not self.twin_laws:
            out.append("  no twin laws possible for this lattice/space group (no merohedral twinning)")
        for op, h, a, n, ab in self.twin_laws:
            out.append(f"  twin law {op:14s}: H-test <H> = {h:.3f} -> alpha {a:.3f}; Britton alpha {ab:.3f} ({n} pairs)")
        return out


def twinning_analysis(hkl, I, sig, cell, sg, isig_min=0.0, tncs=None):
    sg = as_spacegroup(sg)
    d = d_spacing(hkl, cell)
    centric, eps, absent = reflection_props(hkl, sg)
    ok = ~absent & np.isfinite(I) & (sig > 0)
    s2 = 1.0 / d ** 2
    # resolution range: where <I/sigma> > 3 (statistics are meaningless for noise)
    dmin_use = d.min()
    nb = 20
    edges = np.quantile(s2[ok], np.linspace(0, 1, nb + 1))
    for i in range(nb):
        b = ok & (s2 >= edges[i]) & (s2 <= edges[i + 1])
        if b.sum() > 20 and np.mean(I[b] / sig[b]) < 3.0:
            dmin_use = 1 / np.sqrt(edges[i])
            break
    use = ok & (d >= dmin_use)
    # Wilson B from ln(<I/eps>) vs s^2 in 4.5 A .. dmin
    wb = use & (d <= 4.5)
    wilson_b = np.nan
    if wb.sum() > 100:
        e2 = np.quantile(s2[wb], np.linspace(0, 1, 11))
        xs, ys = [], []
        for i in range(10):
            b = wb & (s2 >= e2[i]) & (s2 < e2[i + 1])
            m = np.mean(I[b] / eps[b])
            if m > 0:
                xs.append(np.mean(s2[b]))
                ys.append(np.log(m))
        if len(xs) > 3:
            wilson_b = -2.0 * np.polyfit(xs, ys, 1)[0]
    # normalised intensities E^2 = I / Sigma_aniso(h): anisotropy must be removed first ),
    # otherwise isotropic shells mix strong and weak directions and inflate <I^2>/<I>^2
    ac = use & ~centric
    E2 = np.full(I.size, np.nan)
    try:
        from .aniso import fit_anisotropy
        model = fit_anisotropy(hkl[use], I[use], sig[use], centric[use], eps[use], cell, sg, verbose=False)
        Sig = model.sigma(hkl, cell, eps)
        if tncs is not None:   # tNCS modulation of expected intensities (otherwise moments look 'anti-twinned')
            tv, st_ = np.array(tncs[0]), tncs[1]
            Sig = Sig * np.maximum(1 + st_ * np.cos(2 * np.pi * (hkl @ tv)), 0.05)
        E2[ac] = I[ac] / Sig[ac]
        # renormalise residual profile mismatch in thin shells
        e3 = np.quantile(s2[ac], np.linspace(0, 1, 21))
        for i in range(20):
            b = ac & (s2 >= e3[i]) & (s2 <= e3[i + 1])
            mb = np.mean(E2[b]) if b.any() else 0.0
            # a shell whose intensity falls far below the model has (almost) no signal: excluding it
            # avoids dividing by ~0 (normalising noise to <E^2> = 1 would fake huge moments)
            E2[b] = E2[b] / mb if mb > 0.3 else np.nan
    except Exception:
        e3 = np.quantile(s2[ac], np.linspace(0, 1, 21))
        for i in range(20):
            b = ac & (s2 >= e3[i]) & (s2 <= e3[i + 1])
            E2[b] = I[b] / eps[b] / max(np.mean(I[b] / eps[b]), 1e-12)
    a = np.isfinite(E2)
    # error-corrected second moment: <E^4> = <(I/S)^2> - <(sigma/S)^2> (measurement variance removed),
    # otherwise noise-dominated weak reflections inflate the moment
    sE = np.full(I.size, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        sE[a] = np.where(I[a] != 0, sig[a] * E2[a] / I[a], 0.0)      # sigma / expected intensity
    i2 = float((np.mean(E2[a] ** 2) - np.mean(sE[a] ** 2)) / np.mean(E2[a]) ** 2) if a.any() else np.nan
    me = float(np.mean(np.abs(E2[a] - 1))) if a.any() else np.nan
    # L-test: pairs h, h + 2*delta (same parity class, robust to tNCS)
    key = {k: i for i, k in enumerate(hkl_key(hkl).tolist())}
    Ls = []
    idx = np.flatnonzero(ac)
    for dv in ((2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 0), (2, 0, 2), (0, 2, 2)):
        h2 = hkl[idx] + np.array(dv)
        k2 = hkl_key(h2)
        j = np.array([key.get(int(k), -1) for k in k2])
        m = (j >= 0)
        if not m.any():
            continue
        jj = j[m]
        m2 = ac[jj]
        i1, i2_ = I[idx[m][m2]], I[jj[m2]]
        den = i1 + i2_
        g = den > 0
        Ls.append((i1[g] - i2_[g]) / den[g])
    L = np.clip(np.concatenate(Ls), -1, 1) if Ls else np.array([np.nan])
    # H-test for each possible twin law
    laws = []
    try:
        tl = gemmi.find_twin_laws(gemmi.UnitCell(*cell), sg, 3.0, False)
    except Exception:
        tl = []
    hasu_key = {k: i for i, k in enumerate(hkl_key(hkl).tolist())}
    for op in tl:
        R = op_matrix(op)
        h2 = apply_rot(hkl[ac], R)
        h2a, _, _ = to_asu(h2, sg)
        j = np.array([hasu_key.get(int(k), -1) for k in hkl_key(h2a)])
        m = (j >= 0)
        i1 = I[ac][m]
        i2_ = I[j[m]]
        good = (i1 > 0) & (i2_ > 0) & (j[m] != np.flatnonzero(ac)[m])
        if good.sum() < 50:
            continue
        H = np.abs(i1[good] - i2_[good]) / (i1[good] + i2_[good])
        hm = float(np.mean(H))
        ab, _ = britton_alpha(i1[good], i2_[good], sig[ac][m][good], sig[j[m]][good])
        laws.append((op.triplet(), hm, max(0.0, 0.5 - hm), int(good.sum()), ab))
    return TwinningReport(float(wilson_b), i2, me, float(np.mean(np.abs(L))), float(np.mean(L ** 2)), laws, float(dmin_use))


def l_test_alpha(l_mean):
    """Twin fraction implied by <|L|> (Padilla & Yeates 2003; linear between 0.5 untwinned and 0.375 perfect)."""
    return float(np.clip((0.5 - l_mean) / 0.125 * 0.5, 0.0, 0.5))


def index2_subgroups(sg):
    """Space groups of index 2 in ``sg`` (same lattice / centring), from the closure of pairs of its operations."""
    sg = as_spacegroup(sg)
    go = sg.operations()
    ops = list(go.sym_ops)
    n = len(ops)
    out = {}
    for i in range(n):
        for j in range(i, n):
            g = gemmi.GroupOps([ops[0], ops[i], ops[j]])
            g.cen_ops = list(go.cen_ops)
            try:
                g.add_missing_elements()
            except Exception:
                continue
            if len(g.sym_ops) * 2 != n:
                continue
            sub = gemmi.find_spacegroup_by_ops(g)
            if sub is None:
                # subgroup elements may sit off the origin (e.g. a 2_1 axis at x = 1/4 in P212121): origin shifts
                # change neither intensities nor systematic absences
                import itertools
                for t in itertools.product(sorted({f"{k}/8" for k in range(8)} | {f"{k}/12" for k in range(12)}), repeat=3):
                    g2 = gemmi.GroupOps(list(g.sym_ops))
                    g2.cen_ops = list(g.cen_ops)
                    g2.change_basis_forward(gemmi.Op(f"x+{t[0]},y+{t[1]},z+{t[2]}"))
                    sub = gemmi.find_spacegroup_by_ops(g2)
                    if sub is not None:
                        break
            if sub is not None:
                out[sub.xhm()] = sub
    return list(out.values())


def hidden_twinning_check(hkl, I, sig, cell, sg, report, min_alpha_l=0.1):
    """Twinning seen by the L-test (alpha_L >= min_alpha_l): the apparent symmetry may contain twin
    operators (true space group a subgroup, twinned to look like ``sg``). For each index-2 subgroup the data are
    merged there and the elements it loses are tested as twin laws (H-test, Britton). Returns a list of
    (subgroup, operator, alpha_H, alpha_Britton), best first, or [] when the check does not apply."""
    a_l = l_test_alpha(report.l_mean)
    if a_l < min_alpha_l:
        return []
    res = []
    for sub in index2_subgroups(sg):
        hasu, _, _ = to_asu(hkl, sub)
        key = hkl_key(hasu)
        uk, inv = np.unique(key, return_inverse=True)
        w = 1.0 / np.maximum(sig, 1e-6) ** 2
        Im = np.bincount(inv, w * I, uk.size) / np.bincount(inv, w, uk.size)
        sm = 1.0 / np.sqrt(np.bincount(inv, w, uk.size))
        hm = key_hkl(uk)
        try:
            rep = twinning_analysis(hm, Im, sm, cell, sub)
        except Exception:
            continue
        for op, h, a, nn, ab in rep.twin_laws:
            res.append((sub, op, a, ab))
    res.sort(key=lambda t: -(t[2] + t[3]))
    return res


def lattice_isomorphism(sets, cell, sg, nbins=6, seed=0):
    """Isomorphism of several lattices (or data sets) of one crystal form, model-free. ``sets``: list of
    (hkl, I, sigma) unmerged observations in a common setting. Per resolution shell (equal-count bins over the
    union): each set's CC1/2 (random half split), and for each pair the CC between merged intensities and
    kappa = CC_AB / sqrt(r_A r_B), r = 2 CC1/2 / (1 + CC1/2) (reliability of the merged set). kappa = 1: the sets
    differ only by noise; kappa < 1: genuine structure / cell differences. Shells with CC1/2 < 0.1 in either set
    carry no information (kappa NaN). Returns dict(d_edges, cc_half [set][shell], kappa {(i, j): [shell]},
    cc {(i, j): [shell]})."""
    from .symops import hkl_key, key_hkl
    rng = np.random.default_rng(seed)
    gc = gemmi.UnitCell(*cell)
    merged = []
    for hkl, I, sig in sets:
        key = hkl_key(to_asu(np.asarray(hkl, int), sg)[0])
        uk, inv = np.unique(key, return_inverse=True)
        w = 1.0 / np.maximum(sig, 1e-6) ** 2
        Im = np.bincount(inv, w * I, uk.size) / np.bincount(inv, w, uk.size)
        half = rng.random(len(I)) < 0.5
        hs = []
        for h in (half, ~half):
            num = np.bincount(inv[h], (w * I)[h], uk.size)
            den = np.bincount(inv[h], w[h], uk.size)
            hs.append(np.where(den > 0, num / np.maximum(den, 1e-30), np.nan))
        merged.append((uk, Im, hs[0], hs[1]))
    allk = np.unique(np.concatenate([m[0] for m in merged]))
    dall = np.array([gc.calculate_d(list(map(int, h))) for h in key_hkl(allk)])
    edges = np.quantile(dall, np.linspace(1, 0, nbins + 1))
    dmap = dict(zip(allk.tolist(), dall.tolist()))

    def cc(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 30 else np.nan
    cch = []
    for uk, Im, A1, A2 in merged:
        d = np.array([dmap[k] for k in uk.tolist()])
        cch.append([cc(A1[(d <= hi) & (d >= lo)], A2[(d <= hi) & (d >= lo)]) for hi, lo in zip(edges[:-1], edges[1:])])
    kap, ccs = {}, {}
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            ki, Ii = merged[i][0], merged[i][1]
            kj, Ij = merged[j][0], merged[j][1]
            common, ia, ib = np.intersect1d(ki, kj, return_indices=True)
            d = np.array([dmap[k] for k in common.tolist()])
            kk, cc_ = [], []
            for b, (hi, lo) in enumerate(zip(edges[:-1], edges[1:])):
                m = (d <= hi) & (d >= lo)
                c = cc(Ii[ia][m], Ij[ib][m])
                ca, cb = cch[i][b], cch[j][b]
                ok = np.isfinite(c) and np.isfinite(ca) and np.isfinite(cb) and min(ca, cb) >= 0.1
                ra, rb = (2 * ca / (1 + ca), 2 * cb / (1 + cb)) if ok else (np.nan, np.nan)
                kk.append(float(c / np.sqrt(ra * rb)) if ok else np.nan)
                cc_.append(c)
            kap[(i, j)], ccs[(i, j)] = kk, cc_
    return dict(d_edges=edges, cc_half=cch, kappa=kap, cc=ccs)


# ---------------------------------------------------------------------------
# per-batch statistics (radiation damage diagnostics)
# ---------------------------------------------------------------------------

def batch_statistics(z, uid, I, sig, scale, block_frames=20):
    """Per-frame-block: N, mean I/sigma, Rmeas against the overall merged mean, mean scale."""
    n = uid.max() + 1
    w = 1 / sig ** 2
    Im = np.bincount(uid, w * I, n) / np.maximum(np.bincount(uid, w, n), 1e-30)
    mult = np.bincount(uid, minlength=n)
    fb = (np.floor(z) // block_frames).astype(int)
    rows = []
    for b in np.unique(fb):
        s = (fb == b) & (mult[uid] > 1)
        if s.sum() < 10:
            continue
        m = mult[uid][s]
        rmeas = np.sum(np.sqrt(m / (m - 1)) * np.abs(I[s] - Im[uid][s])) / max(np.sum(np.abs(I[s])), 1e-30)
        rows.append((int(b * block_frames + 1), int((b + 1) * block_frames), int(s.sum()),
                     float(np.mean(I[s] / sig[s])), float(rmeas), float(np.mean(scale[s]))))
    return rows


def repeat_symmetry_cc(hkl, I, z, cell, sg, nshell=8):
    """Do repeated measurements of a reflection share an error its symmetry mates do not?

    Observations are split into the two halves of the sweep (by frame). Per resolution shell: CC_rep = CC(<I>_A(h),
    <I>_B(h)) for the same reflection (Friedel mates merged), CC_sym = CC(<I>_A(h), <I>_B(Rh)) over the space group's
    rotations R != 1. True symmetry with errors independent between observations gives CC_sym ~ CC_rep (absorption
    etc. ~0.9-1.0 x); CC_sym << CC_rep means errors shared by the repeats (same reflection geometry: mirror detector
    position, local background, overlapping neighbours) which also inflate CC1/2 and the resolution limit. Returns
    rows (d_low, d_high, cc_rep, cc_sym, n_rep, n_sym)."""
    hkl = np.asarray(hkl, np.int64)
    I = np.asarray(I, float)
    half = np.asarray(z, float) < np.median(z)

    def fkey(h):
        neg = (h[:, 0] < 0) | ((h[:, 0] == 0) & (h[:, 1] < 0)) | ((h[:, 0] == 0) & (h[:, 1] == 0) & (h[:, 2] < 0))
        return hkl_key(np.where(neg[:, None], -h, h))

    k = fkey(hkl)
    uk = np.unique(k)
    means = []
    for m in (half, ~half):
        idx = np.searchsorted(uk, k[m])
        c = np.bincount(idx, minlength=uk.size)
        means.append((np.bincount(idx, I[m], uk.size) / np.maximum(c, 1), c > 0))
    (mA, okA), (mB, okB) = means
    hu = key_hkl(uk)
    d = d_spacing(hu, cell)
    s2 = 1.0 / d ** 2
    both = okA & okB
    if both.sum() < 50 * nshell:
        return []
    edges = np.quantile(s2[both], np.linspace(0, 1, nshell + 1))
    shell = np.clip(np.searchsorted(edges, s2, side="right") - 1, 0, nshell - 1)
    xs, ys, ss = [], [], []
    for R in rot_matrices(as_spacegroup(sg)):
        if np.array_equal(R, np.eye(3, dtype=R.dtype)):
            continue
        k2 = fkey(hu @ R)
        j = np.minimum(np.searchsorted(uk, k2), uk.size - 1)
        f = (uk[j] == k2) & okA & okB[j] & (j != np.arange(uk.size))
        xs.append(mA[f]), ys.append(mB[j[f]]), ss.append(shell[f])
    xs = np.concatenate(xs) if xs else np.zeros(0)
    ys = np.concatenate(ys) if ys else np.zeros(0)
    ss = np.concatenate(ss) if ss else np.zeros(0, int)
    rows = []
    for s_ in range(nshell):
        mr = both & (shell == s_)
        ms = ss == s_
        cr = float(np.corrcoef(mA[mr], mB[mr])[0, 1]) if mr.sum() > 30 else np.nan
        cs = float(np.corrcoef(xs[ms], ys[ms])[0, 1]) if ms.sum() > 30 else np.nan
        dlo = 1 / np.sqrt(edges[s_]) if edges[s_] > 0 else 99.0
        rows.append((float(dlo), float(1 / np.sqrt(edges[s_ + 1])), cr, cs, int(mr.sum()), int(ms.sum())))
    return rows


# ---------------------------------------------------------------------------
# indexing ambiguity & reference data
# ---------------------------------------------------------------------------

def ambiguity_operators(cell, sg, tol=3.0):
    """Alternative indexing operators: lattice symmetry not in the point group (identity first)."""
    sg = as_spacegroup(sg)
    ops = [np.eye(3, dtype=int)]
    try:
        for op in gemmi.find_twin_laws(gemmi.UnitCell(*cell), sg, tol, False):
            ops.append(op_matrix(op))
    except Exception:
        pass
    return ops


def merge_asu(hkl, I, sig, sg, anomalous=False):
    hasu, _, plus = to_asu(hkl, sg)
    key = hkl_key(hasu) * 2 + ((~plus).astype(np.int64) if anomalous else 0)
    uk, inv = np.unique(key, return_inverse=True)
    w = 1 / np.maximum(sig, 1e-6) ** 2
    Im = np.bincount(inv, w * I, uk.size) / np.bincount(inv, w, uk.size)
    return uk, Im


def best_reindex(hkl, I, sig, ref_keys, ref_I, cell, sg, ops=None):
    """Operator maximising CC(I, I_ref) over common reflections. Returns (R, cc_list)."""
    if ops is None:
        ops = ambiguity_operators(cell, sg)
    ref = dict(zip(ref_keys.tolist(), ref_I.tolist()))
    res = []
    for R in ops:
        uk, Im = merge_asu(apply_rot(hkl, R), I, sig, sg)
        rv = np.array([ref.get(k, np.nan) for k in uk.tolist()])
        m = np.isfinite(rv)
        cc = float(np.corrcoef(Im[m], rv[m])[0, 1]) if m.sum() > 20 else np.nan
        res.append((cc, int(m.sum())))
    best = int(np.nanargmax([c for c, _ in res])) if any(np.isfinite(c) for c, _ in res) else 0
    return ops[best], res


def read_reference(path, sg):
    """Reference intensities (or amplitudes squared) from MTZ or a legacy-format unmerged file -> (asu keys, I)."""
    path = str(path)
    if path.lower().endswith(".mtz"):
        m = gemmi.read_mtz_file(path)
        hkl = np.asarray(m.make_miller_array(), int)
        for lab in ("IMEAN", "I", "I-obs", "IOBS", "F", "FP", "F-obs", "FOBS"):
            col = m.column_with_label(lab)
            if col is not None:
                v = np.asarray(col, float)
                if lab.upper().startswith("F"):
                    v = v ** 2
                ok = np.isfinite(v)
                return merge_asu(hkl[ok], v[ok], np.ones(ok.sum()), sg)
        raise ValueError(f"{path}: no intensity/amplitude column found")
    from .io.legacy import read_legacy_hkl
    _, a = read_legacy_hkl(path)
    a = a[a[:, 4] > 0]
    return merge_asu(a[:, :3].astype(int), a[:, 3], a[:, 4], sg)


def transfer_free_flags(hkl_asu, sg, ref_mtz, fraction=0.05, seed=1234):
    """Copy FreeR_flag from a reference MTZ for matching reflections; extend new reflections at
    random with the same test fraction (keeps the test set consistent across re-processing)."""
    m = gemmi.read_mtz_file(str(ref_mtz))
    col = None
    for lab in ("FreeR_flag", "FREE", "R-free-flags", "FreeRflag"):
        col = m.column_with_label(lab)
        if col is not None:
            break
    if col is None:
        raise ValueError(f"{ref_mtz}: no free-R column")
    rh = np.asarray(m.make_miller_array(), int)
    rf = np.asarray(col, float)
    rhasu, _, _ = to_asu(rh, sg)
    lut = dict(zip(hkl_key(rhasu).tolist(), rf.tolist()))
    flags = np.array([lut.get(k, np.nan) for k in hkl_key(hkl_asu).tolist()])
    known = np.isfinite(flags)
    vals = rf[np.isfinite(rf)]
    nflag = int(round(vals.max())) + 1 if vals.size else int(round(1 / fraction))
    rng = np.random.default_rng(seed)
    flags[~known] = rng.integers(0, nflag, (~known).sum())
    return flags, int(known.sum()), int((~known).sum())


# ---------------------------------------------------------------------------
# translational NCS (pseudo-translation) from the native Patterson
# ---------------------------------------------------------------------------

def tncs_patterson(hkl, I, cell, sg, dmax=10.0, dmin=5.0, min_dist=15.0):
    """Largest off-origin native-Patterson peak (the usual convention: 10-5 A data, > 15 A from origin).

    Returns (peak/origin ratio, fractional vector, distance A). Ratios > 0.2 indicate tNCS, which
    makes reflection classes systematically weak (can mimic absences, inflates <I^2>/<I>^2)."""
    import torch
    from .aniso import expand_p1
    from .geometry import cell_to_B
    d = d_spacing(hkl, cell)
    sel = (d <= dmax) & (d >= max(dmin, d.min())) & np.isfinite(I)
    if sel.sum() < 100:
        return np.nan, None, np.nan
    H, V = expand_p1(hkl[sel], np.clip(I[sel], 0, None), sg)
    n = [int(2 ** np.ceil(np.log2(2 * np.abs(H[:, i]).max() * 2 + 2))) for i in range(3)]
    grid = np.zeros(n, np.float32)
    grid[tuple(H[:, i] % n[i] for i in range(3))] = V
    P = torch.real(torch.fft.ifftn(torch.as_tensor(grid))).numpy()
    P0 = P[0, 0, 0]
    # Cartesian minimum-image distance of each grid point from the origin
    f = [np.fft.fftfreq(n[i]) * 1.0 for i in range(3)]       # fractional coords in [-0.5, 0.5)
    F = np.stack(np.meshgrid(*f, indexing="ij"), -1)
    real = np.linalg.inv(cell_to_B(cell))                       # rows a, b, c
    dist = np.linalg.norm(F @ real, axis=-1)
    # lattice-centring translations (C, I, F, R) give Patterson peaks as high as the origin by symmetry, not tNCS:
    # excluded like the origin
    far = dist > min_dist
    for c in as_spacegroup(sg).operations().cen_ops:
        t = np.array(c, float) / gemmi.Op.DEN
        if np.any(t % 1.0 != 0):
            dc = F - t
            dc -= np.round(dc)
            far &= np.linalg.norm(dc @ real, axis=-1) > min_dist
    Pm = np.where(far, P, -np.inf)
    k = np.unravel_index(np.argmax(Pm), P.shape)
    return float(P[k] / P0), np.array([F[k][i] for i in range(3)]), float(dist[k])


def britton_alpha(I1, I2, s1, s2, alphas=np.linspace(0, 0.49, 50)):
    """Britton plot: fraction of negative detwinned intensities vs alpha; alpha = onset of the linear rise."""
    frac = []
    for a in alphas:
        d1 = ((1 - a) * I1 - a * I2) / (1 - 2 * a)
        d2 = ((1 - a) * I2 - a * I1) / (1 - 2 * a)
        frac.append(np.mean(np.r_[d1, d2] < -2.0 * np.r_[s1, s2]))
    frac = np.array(frac)
    base = frac[0]
    above = np.flatnonzero(frac > base + 0.01)
    return (float(alphas[above[0]]) if above.size else float(alphas[-1])), frac


def detwin(hkl, I, sig, sg, twin_op, alpha):
    """Algebraic detwinning (Yeates 1997): I_A = ((1-a) I1 - a I2)/(1-2a), errors propagated.
    Only sensible for alpha well below 0.5; refinement against twinned data (twin law + fraction) is
    preferred when alpha > ~0.3."""
    R = op_matrix(gemmi.Op(twin_op))
    h2a, _, _ = to_asu(apply_rot(hkl, R), sg)
    lut = {k: i for i, k in enumerate(hkl_key(hkl).tolist())}
    j = np.array([lut.get(int(k), -1) for k in hkl_key(h2a)])
    ok = j >= 0
    Id = np.full(I.size, np.nan)
    sd = np.full(I.size, np.nan)
    I2, s2 = I[np.maximum(j, 0)], sig[np.maximum(j, 0)]
    Id[ok] = ((1 - alpha) * I[ok] - alpha * I2[ok]) / (1 - 2 * alpha)
    sd[ok] = np.sqrt((1 - alpha) ** 2 * sig[ok] ** 2 + alpha ** 2 * s2[ok] ** 2) / (1 - 2 * alpha)
    return Id, sd, ok

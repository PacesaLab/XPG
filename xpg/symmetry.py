"""Point-group / space-group determination from integrated intensities.

1. Intensities are normalised (E^2) in resolution shells and phi blocks.
2. For every rotation of the lattice holohedry, the correlation between E^2(h) and E^2(hR)
   is computed over all measured pairs (Friedel's law assumed).
3. Each candidate subgroup (from lattice.lattice_candidates) is scored by a likelihood:
   operators in the group should correlate like repeated measurements, operators not in the
   group should not correlate.
4. Within the best point group, screw axes are identified from axial systematic absences.
"""
from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np

from .lattice import LatticeCandidate
from .symops import hkl_key


@dataclass
class OpScore:
    R: np.ndarray        # rotation in reduced basis (acts on fractional coords)
    order: int
    cc: float
    n: int
    null_mean: float = 0.0
    null_sd: float = 0.05
    z: float = float("nan")
    pseudo: str = ""      # "twin" / "ncs": a high CC explained by twinning or by NCS pseudo-symmetry (see score_symmetry)
    ratio_lo: float = float("nan")  # CC / expected CC in the low- and high-resolution halves (NCS test)
    ratio_hi: float = float("nan")
    twin_cc: float = float("nan")   # lowest CC a twin law predicts (twin fraction >= the L-test's lower bound)


@dataclass
class GroupScore:
    cand: LatticeCandidate
    score: float
    cc_in: float
    cc_out: float
    prob: float = 0.0


def normalise_intensities(I, sig, s, phi, nshell=20, block_deg=10.0):
    """E^2 = I / <I>(s, phi-block) with a multiplicative phi-block factor."""
    edges = np.quantile(s, np.linspace(0, 1, nshell + 1))
    sb = np.clip(np.searchsorted(edges, s) - 1, 0, nshell - 1)
    mean_s = np.array([np.mean(I[sb == i]) if np.any(sb == i) else 1.0 for i in range(nshell)])
    mean_s = np.maximum(mean_s, 1e-6)
    E2 = I / mean_s[sb]
    pb = np.floor((phi - phi.min()) / block_deg).astype(int)
    nb = pb.max() + 1
    f = np.bincount(pb, E2, nb) / np.maximum(np.bincount(pb, minlength=nb), 1)
    f = np.where(f > 0, f, 1.0)
    E2 = E2 / f[pb]
    sE = sig / mean_s[sb] / f[pb]
    return E2, sE


def _friedel_canon(h):
    k1 = hkl_key(h)
    k2 = hkl_key(-h)
    return np.maximum(k1, k2)


def symmetry_resolution_limit(I, sig, s, isig_limit=6.0, nbins=20, min_fraction=0.05):
    """Automatic limit: highest resolution where <I>/<sigma> stays >= isig_limit; at least
    ``min_fraction`` of the observations. A 25% floor pushed weak data deep into noise (9SO6: 1.73 A instead of ~3
    A; the 2-folds then decayed like NCS and P1 won). Scan of all
    runs."""
    edges = np.quantile(s, np.linspace(0, 1, nbins + 1))
    smax = edges[-1]
    for i in range(nbins):
        b = (s >= edges[i]) & (s <= edges[i + 1])
        if b.sum() > 20 and np.mean(I[b]) / max(np.mean(sig[b]), 1e-12) < isig_limit:
            smax = edges[i]
            break
    smax = max(smax, np.quantile(s, min_fraction))
    return smax


def _lorentz_logpdf(x, mu, gamma):
    """log of a Lorentzian truncated to [-1, 1] (normalised)."""
    norm = (np.arctan((1 - mu) / gamma) - np.arctan((-1 - mu) / gamma)) / np.pi
    return -np.log(np.pi * gamma * (1 + ((x - mu) / gamma) ** 2)) - np.log(norm)


# <|L|> vs twin fraction (simulated, acentric, noise-free): alpha 0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5
_L_CAL = (np.array([0.0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]),
          np.array([.5, .464, .4406, .4222, .4083, .3971, .3891, .3832, .3776, .3756, .3748]))


def _L_from_alpha(a):
    al, l_ = _L_CAL
    return float(np.interp(np.clip(a, 0.0, 0.5), al, l_))


def _alpha_from_ratio(r):
    """Twin fraction whose twin-mate intensity CC is r x the expected CC: r = 2a(1-a)/(a^2+(1-a)^2)."""
    r = float(np.clip(r, 0.0, 1.0))
    return 0.5 - 0.5 * np.sqrt((1 - r) / (1 + r))


def _alpha_from_L(L):
    if not np.isfinite(L):
        return np.nan
    a, l_ = _L_CAL
    return float(np.interp(-L, -l_, a))


def _pseudo_symmetry(cc, cc_true, alpha_L, Em, ii, jj, s_u, s_med, cc_true_half, info=None):
    """'twin' if the element's CC deficit is what the L-test's twin fraction predicts (a twin law: mates correlate at
    2a(1-a)/(a^2+(1-a)^2) of the expected CC); 'ncs' if its CC ratio falls significantly from the low- to the
    high-resolution half (imperfect NCS decays with resolution; systematic errors of true symmetry do not);
    '' otherwise."""
    ratio = cc / max(cc_true, 1e-3)
    if np.isfinite(alpha_L) and alpha_L >= 0.1:
        a = alpha_L
        twin_ratio = 2 * a * (1 - a) / (a * a + (1 - a) ** 2)
        if ratio > 0.5 * twin_ratio:      # a twin of this fraction explains (at least) this much correlation
            return "twin"
    lo = s_u[ii] <= s_med
    out = []
    for m, ct in ((lo, cc_true_half[0]), (~lo, cc_true_half[1])):
        if m.sum() < 30:
            return ""
        c = np.corrcoef(Em[ii][m], Em[jj][m])[0, 1]
        out.append((c / ct, m.sum(), c))
    (r_lo, n_lo, c_lo), (r_hi, n_hi, c_hi) = out
    if info is not None:
        info.update(ratio_lo=r_lo, ratio_hi=r_hi)
    # Fisher z difference of the two CCs (the CC ratio's own precision); significant decay beyond the decay of the
    # expected CC itself
    se = np.sqrt(1 / max(n_lo - 3, 1) + 1 / max(n_hi - 3, 1))
    z_hi_expect = np.arctanh(np.clip(r_lo * cc_true_half[1], -0.999, 0.999))
    z_hi = np.arctanh(np.clip(c_hi, -0.999, 0.999))
    # effect size AND significance: true symmetry also decays a little, imperfect NCS collapses at high resolution
    # (simulated c 0.40-0.45: >= 29%); with 10^5 pairs any decay is 'significant'
    if r_hi < (1.0 - NCS_DECAY) * r_lo and (z_hi_expect - z_hi) / se > 3.0:
        return "ncs"
    return ""


NCS_DECAY = 0.15       # relative drop of CC / expected CC from the low- to the high-resolution half that marks NCS
L_SD = 0.015           # uncertainty of <|L|> as a twin-fraction measure (modelling, not counting statistics)
L_UNTWINNED = 0.44     # <|L|> above which twin fractions > 0.1 are excluded (simulated: 0.441 at alpha 0.1)


def score_symmetry(hkl_red, I, sig, s, phi, cands, isig_limit=6.0, verbose=True, log=print, seed=0):
    """Score lattice-candidate point groups. ``hkl_red``: indices in the reduced cell.

    * automatic resolution cut-off where <I>/<sigma> < isig_limit
    * CC on E^2 for each symmetry element; the null distribution (mean, sd) is obtained from the
      same number of resolution-matched pairs of *unrelated* reflections
    * expected CC for a true symmetry element = CC of repeated measurements of the same
      reflection (half-set CC corrected with Spearman-Brown)
    * likelihoods with truncated Lorentzian distributions, combined
      over all elements present / absent in each candidate group
    """
    smax = symmetry_resolution_limit(I, sig, s, isig_limit)
    use = np.isfinite(I) & (sig > 0) & (s <= smax)
    E2, sE = normalise_intensities(I[use], sig[use], s[use], phi[use])
    E2 = np.clip(E2, -3, 20)       # robustness against outliers / ice
    key = _friedel_canon(hkl_red[use])
    uk, inv = np.unique(key, return_inverse=True)
    n = uk.size
    cnt = np.bincount(inv, minlength=n)
    Em = np.bincount(inv, E2, n) / np.maximum(cnt, 1)
    s_u = np.zeros(n)
    s_u[inv] = s[use]
    rng = np.random.default_rng(seed)
    # expected CC for true symmetry: repeated measurements (half sets, Spearman-Brown)
    r = rng.random(E2.size)
    srt = np.lexsort((r, inv))
    first = np.r_[0, np.flatnonzero(np.diff(inv[srt])) + 1]
    pos = np.arange(E2.size) - np.repeat(first, np.diff(np.r_[first, E2.size]))
    half = np.empty(E2.size, bool)
    half[srt] = pos % 2 == 1
    c1 = np.bincount(inv[~half], minlength=n)
    c2 = np.bincount(inv[half], minlength=n)
    e1 = np.bincount(inv[~half], E2[~half], n) / np.maximum(c1, 1)
    e2 = np.bincount(inv[half], E2[half], n) / np.maximum(c2, 1)
    both = (c1 > 0) & (c2 > 0)
    cc_half = np.corrcoef(e1[both], e2[both])[0, 1] if both.sum() > 20 else np.nan
    cc_true = 2 * cc_half / (1 + cc_half) if np.isfinite(cc_half) else 0.9
    cc_true = float(np.clip(cc_true, 0.05, 0.995))
    from .symops import key_hkl
    hu = key_hkl(uk)
    # L-test (Padilla & Yeates 2003) on the merged P1 data, independent of the symmetry hypotheses: pairs h, h + 2 delta
    L_abs = []
    # lower-resolution half only (highest signal-to-noise): noise biases <|L|> towards 'untwinned'
    s_lo_half = float(np.median(s_u))
    for dv in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
        k2 = _friedel_canon(hu + np.array(dv))
        j = np.minimum(np.searchsorted(uk, k2), n - 1)
        f = (uk[j] == k2) & (Em + Em[j] > 0) & (s_u <= s_lo_half)
        # as in analysis.twinning_analysis: all pairs with a positive sum, |L| clipped at 1 (selecting positive
        # intensities only would bias <|L|> towards twinning)
        L_abs.append(np.minimum(np.abs(Em[f] - Em[j[f]]) / (Em[f] + Em[j[f]]), 1.0))
    L_abs = np.concatenate(L_abs) if L_abs else np.array([])
    L_mean = float(L_abs.mean()) if L_abs.size > 200 else np.nan
    # width of the 'symmetry present' CC distribution: symmetry mates are measured in different geometry, so
    # systematic errors (absorption, damage, corrections) lower their CC below that of repeated measurements of the
    # same reflection. That deficit is tolerated (10% of the expected CC) only when twinning is excluded by the L-test
    # (<|L|> >= 0.44: twin fraction <= 0.1, CC of twin mates <= 0.22); otherwise a partial merohedral twin (alpha 0.3:
    # CC ratio 0.72) must not pass for symmetry and the floor stays 0.03.
    gam_sys = 0.1 * cc_true if np.isfinite(L_mean) and L_mean >= L_UNTWINNED else 0.03
    alpha_L = _alpha_from_L(L_mean)
    # lower bound on the twin fraction (one L_SD): <|L|> is flat above alpha ~0.35 (0.383 -> 0.375 at 0.5), so the
    # estimate alone reads a 40% twin as a perfect one; the twin mates' CC (0.93 of the expected) then fell outside
    # the twin-law range, and true symmetry elements (CC as expected) scored higher as 'perfect twin laws' than as
    # symmetry (P31 twinned by -h,-k,l was taken for point group 2, or 1)
    alpha_lo = _alpha_from_L(L_mean + L_SD)
    # expected CC of true symmetry mates per resolution half (repeats, half sets + Spearman-Brown): the NCS test
    # compares each element's CC ratio in the two halves (systematic errors ~flat in resolution, NCS decays)
    s_med = float(np.median(s_u[both])) if both.any() else 0.0
    cc_true_half = []
    for hm in (s_u <= s_med, s_u > s_med):
        bb = both & hm
        ch = np.corrcoef(e1[bb], e2[bb])[0, 1] if bb.sum() > 20 else np.nan
        cc_true_half.append(float(np.clip(2 * ch / (1 + ch), 0.05, 0.995)) if np.isfinite(ch) else cc_true)
    sbins = np.clip(np.searchsorted(np.quantile(s_u, np.linspace(0, 1, 11)), s_u) - 1, 0, 9)
    allops = {}
    for c in cands:
        Pi = np.linalg.inv(c.P)
        for Rc in c.ops:
            Rr = np.round(c.P @ Rc @ Pi).astype(int)
            allops.setdefault(Rr.tobytes(), Rr)
    ident = np.eye(3, dtype=int).tobytes()
    opscores = {}
    for kb, R in allops.items():
        if kb == ident:
            continue
        k2 = _friedel_canon(hu @ R)
        j = np.minimum(np.searchsorted(uk, k2), n - 1)
        found = (uk[j] == k2) & (j != np.arange(n))
        ii = np.flatnonzero(found)
        jj = j[found]
        M = np.eye(3, dtype=int)
        order = 0
        for kk in range(1, 7):
            M = M @ R
            if np.array_equal(M, np.eye(3, dtype=int)):
                order = kk
                break
        if ii.size < 10:
            opscores[kb] = OpScore(R, order, np.nan, int(ii.size))
            continue
        cc = float(np.corrcoef(Em[ii], Em[jj])[0, 1])
        info = {}
        pseudo = _pseudo_symmetry(cc, cc_true, alpha_L, Em, ii, jj, s_u, s_med, cc_true_half, info)
        # null: same number of resolution-matched unrelated pairs (partner shuffled within bin)
        null = []
        for _ in range(20):
            jr = jj.copy()
            for b in range(10):
                m = sbins[ii] == b
                jr[m] = rng.permutation(jr[m])
            null.append(np.corrcoef(Em[ii], Em[jr])[0, 1])
        o = OpScore(R, order, cc, int(ii.size))
        o.pseudo = pseudo
        o.ratio_lo, o.ratio_hi = info.get("ratio_lo", np.nan), info.get("ratio_hi", np.nan)
        if pseudo == "twin":
            a = alpha_lo
            o.twin_cc = cc_true * 2 * a * (1 - a) / (a * a + (1 - a) ** 2)
        o.null_mean, o.null_sd = float(np.mean(null)), float(np.std(null) + 1e-6)
        o.z = (cc - o.null_mean) / o.null_sd
        opscores[kb] = o
    results = []
    for c in cands:
        Pi = np.linalg.inv(c.P)
        inset = {np.round(c.P @ Rc @ Pi).astype(int).tobytes() for Rc in c.ops}
        sc = 0.0
        cin, cout = [], []
        for kb, o in opscores.items():
            if not np.isfinite(o.cc):
                continue
            # present: CC as for repeated measurements (width = sampling error, floored at 0.03 for
            # systematic differences between symmetry mates measured in different geometry);
            # absent: either unrelated (null distribution) or *partially* related - merohedral twinning
            # or pseudo-symmetry give intermediate CCs - modelled as a uniform density between the null
            # mean and the expected CC. Without that component, a 30% twin (CC ~0.7) is taken for symmetry.
            # pseudo-symmetry (twin- or NCS-like evidence) gets no tolerance for a CC deficit
            gam = max(o.null_sd, 0.03 if o.pseudo else gam_sys)
            if kb in inset:
                # twin evidence (L-test twinned and this element's CC as a twin law of that fraction predicts):
                # symmetry and twin law compete on equal terms; at a tie the lower symmetry is preferred (merging
                # pseudo-equivalent reflections corrupts the data, too low a symmetry only lowers the multiplicity)
                sc += _lorentz_logpdf(o.cc, cc_true, gam) + (np.log(0.5) if o.pseudo == "twin" else 0.0)
                cin.append(o.cc)
            elif o.pseudo == "twin":
                # measurement noise biases <|L|> towards 'untwinned': the L-test gives a LOWER bound on the twin
                # fraction, so a twin law predicts any CC between twin_cc (alpha = the bound) and the expected CC
                # (alpha = 0.5): uniform there, mixed with the unrelated (null) hypothesis
                hi = cc_true + gam
                p_tw = 1.0 / max(hi - o.twin_cc, 1e-3) if o.twin_cc <= o.cc <= hi else 0.0
                p_null = np.exp(_lorentz_logpdf(o.cc, o.null_mean, max(o.null_sd, 0.05)))
                sc += np.log(0.5 * p_tw + 0.5 * p_null + 1e-300)
                cout.append(o.cc)
            else:
                lo, hi = o.null_mean, max(cc_true, o.null_mean + 0.1)
                p_null = np.exp(_lorentz_logpdf(o.cc, o.null_mean, max(o.null_sd, 0.05)))
                p_part = (1.0 / (hi - lo)) if lo <= o.cc <= hi else 0.0
                sc += np.log(0.5 * p_null + 0.5 * p_part + 1e-300)
                cout.append(o.cc)
        # global consistency with the L-test: a candidate group allows twinning only by elements outside it, at
        # most the fraction their CCs imply; data MORE twinned than that (lower <|L|>) rule the group out. One-sided:
        # noise and NCS only raise <|L|> (look less twinned), so observing less twinning never counts against it.
        if np.isfinite(L_mean) and alpha_L >= 0.1:
            a_max = max((_alpha_from_ratio(c_ / cc_true) for c_ in cout if np.isfinite(c_)), default=0.0)
            L_pred = _L_from_alpha(a_max)
            if L_mean < L_pred:
                sc += -0.5 * ((L_pred - L_mean) / L_SD) ** 2
        results.append(GroupScore(c, sc, float(np.mean(cin)) if cin else np.nan,
                                  float(np.mean(cout)) if cout else np.nan))
    sc = np.array([r.score for r in results])
    p = np.exp(sc - sc.max())
    p /= p.sum()
    for r, pp in zip(results, p):
        r.prob = float(pp)
    results.sort(key=lambda r: -r.score)
    if verbose:
        log(f"  SYMMETRY: using {use.sum()} observations to {1 / smax:.2f} A (<I>/<sigma> >= {isig_limit}); "
            f"expected CC for true symmetry {cc_true:.3f}; L-test <|L|> {L_mean:.3f} -> systematic width {gam_sys:.3f}")
        log("  SYMMETRY: symmetry elements:")
        for o in sorted(opscores.values(), key=lambda o: -np.nan_to_num(o.cc, nan=-9)):
            z = getattr(o, "z", np.nan)
            extra = (f"  CC/expected lo {o.ratio_lo:.3f} hi {o.ratio_hi:.3f}" if np.isfinite(o.ratio_lo) else "") + \
                (f"  [{o.pseudo}]" if o.pseudo else "")
            log(f"     {o.order}-fold  CC {o.cc:6.3f}  Z {z:7.1f}  pairs {o.n:7d}  R={o.R.tolist()}{extra}")
        Pi0 = np.linalg.inv(results[0].cand.P)
        chosen = {np.round(results[0].cand.P @ Rc @ Pi0).astype(int).tobytes() for Rc in results[0].cand.ops}
        for kb, o in opscores.items():
            if kb not in chosen and np.isfinite(o.cc) and o.cc > o.null_mean + 0.25:
                log(f"  SYMMETRY WARNING: {o.order}-fold R={o.R.tolist()} correlates with CC {o.cc:.2f} "
                    f"(expected {cc_true:.2f} for symmetry): partial merohedral twinning or pseudo-symmetry")
        log("  SYMMETRY: Laue-group candidates (best first; accepted if L >= 0.2 L_max):")
        for r in results[:8]:
            acc = "*" if r.prob >= 0.2 * results[0].prob else " "
            log(f"   {acc} {r.cand.bravais:3s} {r.cand.point_group:4s} prob {r.prob:6.3f}  CCin {r.cc_in:6.3f} "
                f"CCout {r.cc_out:6.3f}  penalty {r.cand.penalty:5.2f}  cell {np.array2string(r.cand.cell, precision=2)}")
    return results, opscores, cc_true


# ---------------------------------------------------------------------------
# space group from systematic absences
# ---------------------------------------------------------------------------

def _neighbour_corrected(hkl, I, frac=0.02):
    """Neighbour correction for axial reflections: I' = I - frac * (I(h-1) + I(h+1)),
    not allowed to become negative, removing contamination from strong adjacent reflections."""
    from .symops import hkl_key
    Ic = I.copy()
    lut = dict(zip(hkl_key(hkl).tolist(), I.tolist()))
    axial = np.sum(hkl == 0, axis=1) == 2
    for i in np.flatnonzero(axial):
        ax = int(np.argmax(hkl[i] != 0))
        nb = 0.0
        for d in (-1, 1):
            h2 = hkl[i].copy()
            h2[ax] += d
            for hh in (h2, -h2):
                v = lut.get(int(hkl_key(hh[None])[0]))
                if v is not None:
                    nb += max(v, 0.0)
                    break
        # the correction may reduce a positive intensity at most to zero and never makes an
        # already-negative intensity more negative
        Ic[i] = max(I[i] - frac * nb, 0.0) if I[i] > 0 else I[i]
    return Ic


def _wilson_loglik(I, sig, Sigma, centric, n=400):
    """log p(I_obs | Wilson prior with mean Sigma, Gaussian error sig), centric or acentric (quadrature)."""
    I, sig, Sigma = (np.asarray(v, float) for v in (I, sig, Sigma))
    Fmax = np.sqrt(np.maximum(I, 0) + 8 * sig + 6 * Sigma)
    u = (np.arange(n) + 0.5) / n
    F = Fmax[:, None] * u[None]
    J = F * F
    lp = np.where(centric[:, None], 0.5 * np.log(2 / (np.pi * Sigma[:, None])) - J / (2 * Sigma[:, None]),
                  np.log(2 * F / Sigma[:, None] + 1e-300) - J / Sigma[:, None])
    ll = -0.5 * ((I[:, None] - J) / sig[:, None]) ** 2 - np.log(np.sqrt(2 * np.pi) * sig[:, None])
    t = lp + ll
    mx = t.max(1, keepdims=True)
    return (mx[:, 0] + np.log(np.sum(np.exp(t - mx), 1) * Fmax / n))


def _robust_axial_merge(hkl_obs, I_obs, sig_obs, keys_u):
    """Median-based merge (resistant to multiple-diffraction spikes, which occur only at specific
    phi/psi) for the given unique keys. Returns (I, sig, n)."""
    k = hkl_key(hkl_obs)
    out_I, out_s, out_n = [], [], []
    order = np.argsort(k)
    ks = k[order]
    for key in keys_u:
        lo, hi = np.searchsorted(ks, key), np.searchsorted(ks, key, "right")
        idx = order[lo:hi]
        if idx.size == 0:
            out_I.append(np.nan); out_s.append(np.nan); out_n.append(0)
            continue
        v, sv = I_obs[idx], sig_obs[idx]
        if idx.size >= 3:
            med = np.median(v)
            out_I.append(med)
            out_s.append(1.2533 * np.sqrt(1.0 / np.sum(1 / sv ** 2)))
        else:
            w = 1 / sv ** 2
            out_I.append(np.sum(w * v) / np.sum(w))
            out_s.append(np.sqrt(1 / np.sum(w)))
        out_n.append(idx.size)
    return np.array(out_I), np.array(out_s), np.array(out_n)


def candidate_space_groups(cand: LatticeCandidate, chiral=True):
    """All gemmi space-group settings whose rotations equal the candidate's and with its centring."""
    rot_set = {R.tobytes() for R in cand.ops}
    cen = cand.bravais[1]
    out = []
    for sg in gemmi.spacegroup_table_itb():
        if chiral and not sg.is_sohncke():
            continue
        if sg.hm[0] != cen:
            continue
        ops = sg.operations()
        rs = {(np.array(op.rot, int) // gemmi.Op.DEN).tobytes() for op in ops.sym_ops}
        if rs == rot_set:
            out.append(sg)
    uniq = {}
    for sg in out:
        k = tuple(sorted(str(op) for op in sg.operations()))
        if k not in uniq or sg.is_reference_setting():
            uniq[k] = sg
    return list(uniq.values())


def score_space_groups(hkl_conv, Im, sm, cand: LatticeCandidate, chiral=True, verbose=True, log=print,
                       obs=None, cell=None, sigma_of_d=None, p_outlier=0.03, tncs=None):
    """Space group from systematic absences by likelihood (merged data in the conventional cell).

    present: Wilson prior (epsilon-weighted expected intensity at that resolution) convolved with the
    measurement error; absent: true intensity zero. Both models have a small outlier component so a
    single contaminated reflection cannot decide. Axial intensities are corrected for neighbour
    contamination and, when unmerged observations are given (``obs`` = (hkl, I, sig)), merged with
    a median to resist multiple-diffraction (Renninger) spikes.
    """
    sgs = candidate_space_groups(cand, chiral)
    if not sgs:
        return [(gemmi.SpaceGroup(cand.space_group), 0.0)]
    h = np.ascontiguousarray(hkl_conv, dtype=np.int32)
    absent_sets = [np.asarray(sg.operations().systematic_absences(h), bool) for sg in sgs]
    union = np.any(absent_sets, axis=0)
    if not union.any():
        return [(g, 0.0) for g in sorted(sgs, key=lambda g: g.number)]
    I_u, s_u = Im.copy(), sm.copy()
    if obs is not None:
        ho, Io, so = obs
        keys = hkl_key(h[union])
        Ir, sr, nr = _robust_axial_merge(ho, Io, so, keys)
        okr = np.isfinite(Ir)
        I_u[np.flatnonzero(union)[okr]] = Ir[okr]
        s_u[np.flatnonzero(union)[okr]] = sr[okr]
    I_u = _neighbour_corrected(h, I_u)
    pg = gemmi.SpaceGroup(cand.space_group)
    from .symops import reflection_props
    centric, eps, _ = reflection_props(h[union], pg)
    if sigma_of_d is not None and cell is not None:
        from .symops import d_spacing
        Sig = eps * sigma_of_d(d_spacing(h[union], cell))
    else:
        Sig = eps * max(np.mean(np.clip(Im[~union], 0, None)), 1e-6) * np.ones(union.sum())
    if tncs is not None:
        tv, st_ = tncs
        Sig = Sig * np.maximum(1 + st_ * np.cos(2 * np.pi * (h[union] @ tv)), 0.05)
    Iz, sz = I_u[union], np.maximum(s_u[union], 1e-6)
    l_pres = _wilson_loglik(Iz, sz, np.maximum(Sig, 1e-9), centric)
    l_abs = -0.5 * (Iz / sz) ** 2 - np.log(np.sqrt(2 * np.pi) * sz)
    # outlier component: flat density over the plausible intensity range
    l_out = -np.log(np.maximum(np.abs(Iz) + 10 * sz + 6 * Sig, 1e-9))
    l_pres = np.logaddexp(np.log(1 - p_outlier) + l_pres, np.log(p_outlier) + l_out)
    l_abs = np.logaddexp(np.log(1 - p_outlier) + l_abs, np.log(p_outlier) + l_out)
    scores = []
    for sg, ab in zip(sgs, absent_sets):
        a = ab[union]
        scores.append(float(np.sum(l_abs[a]) + np.sum(l_pres[~a])))
    order = np.argsort([-x for x in scores])
    res = [(sgs[i], scores[i]) for i in order]
    if verbose:
        log(f"  SYMMETRY: space groups in point group {cand.point_group} ({union.sum()} zone reflections, "
            f"Wilson-likelihood absences):")
        for sg, sc in res[:8]:
            log(f"     {sg.xhm():14s} #{sg.number:3d}  log-likelihood {sc - res[0][1]:8.2f}")
        # which axial reflections were measured at all (blind region: axis parallel to the spindle)
        cover = []
        for ax, name in enumerate("abc"):
            on_axis = (np.sum(h != 0, axis=1) == 1) & (h[:, ax] != 0)
            idx = np.unique(np.abs(h[on_axis, ax]))
            cover.append(f"{name}*: {idx.size}" + (f" ({idx.min()}-{idx.max()})" if idx.size else " (none)"))
        log("     axial reflections measured: " + ", ".join(cover))
        tied = [sg.xhm() for sg, sc in res if res[0][1] - sc < 1.0]
        if len(tied) > 1:
            log(f"  SYMMETRY WARNING: absences cannot distinguish {', '.join(tied)}. Causes: axial zones not "
                f"measured (axis parallel to the spindle lies in the blind region -> reorient the crystal / kappa) "
                f"or enantiomorphic screw axes (P41/P43, P31/P32, P61/P65, P62/P64...) which no intensity "
                f"statistic can separate -> decide by molecular replacement or anomalous substructure hand.")
    return res

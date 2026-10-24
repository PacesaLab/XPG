"""Merging and data-quality statistics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import generate_hkl
from .symops import as_spacegroup, d_spacing, hkl_key, reflection_props, to_asu


def unique_ids(hasu, plus=None, anomalous=False):
    key = hkl_key(hasu)
    if anomalous and plus is not None:
        key = key * 2 + (~plus).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    return inv, uk


def weighted_merge(I, sig, uid, n):
    w = 1.0 / np.maximum(sig, 1e-6) ** 2
    sw = np.bincount(uid, w, n)
    swi = np.bincount(uid, w * I, n)
    mult = np.bincount(uid, minlength=n)
    Im = swi / np.maximum(sw, 1e-30)
    sm = 1.0 / np.sqrt(np.maximum(sw, 1e-30))
    # inflate sigma by observed scatter when larger (chi^2 > 1)
    resid = np.bincount(uid, w * (I - Im[uid]) ** 2, n)
    chi2 = np.where(mult > 1, resid / np.maximum(mult - 1, 1), 1.0)
    sm = sm * np.sqrt(np.maximum(chi2, 1.0))
    return Im, sm, mult


def possible_unique(cell, sg, dmin, dmax=1e9, anomalous=False):
    """Unique reflections (asu) theoretically observable to dmin (excluding systematic absences)."""
    from .geometry import cell_to_B
    Astar = cell_to_B(cell)
    H = generate_hkl(Astar, dmin, dmax)
    hasu, _, _ = to_asu(H, sg)
    u = np.unique(hkl_key(hasu))
    from .symops import key_hkl
    ha = key_hkl(u)
    centric, eps, absent = reflection_props(ha, sg)
    ha = ha[~absent]
    centric = centric[~absent]
    d = d_spacing(ha, cell)
    return ha, d, centric


@dataclass
class ShellStats:
    dmax: float
    dmin: float
    n_obs: int
    n_uniq: int
    n_possible: int
    completeness: float
    multiplicity: float
    i_over_sigma: float
    rmerge: float
    rmeas: float
    rpim: float
    cc_half: float
    cc_ano: float
    sig_ano: float
    completeness_ano: float
    mean_i: float


def shell_edges(d, nbins, dmin, dmax):
    """Equal-volume shells in 1/d^3."""
    s3 = np.linspace(1.0 / dmax ** 3, 1.0 / dmin ** 3, nbins + 1)
    return 1.0 / np.cbrt(s3)


def _cc(a, b):
    if a.size < 3:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def merging_statistics(hasu, plus, I, sig, cell, sg, nbins=10, dmin=None, dmax=None, anomalous=False,
                       seed=0, shell_limits=None, split_key=None):
    """Compute per-shell and overall merging statistics. Returns (shells, overall, merged dict).
    shell_limits: high-resolution limits of the shells (RESOLUTION_SHELLS=), decreasing.
    split_key: per observation, e.g. its P1 reflection (Friedel mates together): CC1/2 half-datasets then split
    whole groups (all observations of one P1 reflection in the same half), so the halves of a unique reflection
    come from different symmetry mates and errors shared by repeated measurements do not correlate them."""
    sg = as_spacegroup(sg)
    d = d_spacing(hasu, cell)
    if dmin is None:
        dmin = d.min()
    if dmax is None:
        dmax = d.max() * 1.0001
    sel = (d >= dmin) & (d <= dmax)
    hasu, plus, I, sig, d = hasu[sel], plus[sel], I[sel], sig[sel], d[sel]
    if split_key is not None:
        split_key = np.asarray(split_key, np.int64)[sel]
    # non-anomalous merge (Friedel mates together)
    uid, uk = unique_ids(hasu)
    n = uk.size
    Im, sm, mult = weighted_merge(I, sig, uid, n)
    du = np.zeros(n)
    du[uid] = d
    # anomalous merge
    uida, uka = unique_ids(hasu, plus, anomalous=True)
    na = uka.size
    Ima, sma, multa = weighted_merge(I, sig, uida, na)
    # R factors
    absdev = np.abs(I - Im[uid])
    m_o = mult[uid]
    rm_num = np.bincount(uid, absdev, n)
    rmeas_num = np.bincount(uid, absdev * np.sqrt(m_o / np.maximum(m_o - 1, 1)), n)
    rpim_num = np.bincount(uid, absdev * np.sqrt(1.0 / np.maximum(m_o - 1, 1)), n)
    den = np.bincount(uid, I, n)
    multi = mult > 1
    # CC1/2 via random half-datasets
    rng = np.random.default_rng(seed)
    order = rng.permutation(I.size)
    rank = np.empty(I.size, np.int64)
    # random rank within each unique reflection
    srt = np.lexsort((order, uid))
    first = np.r_[0, np.flatnonzero(np.diff(uid[srt])) + 1]
    pos = np.arange(I.size) - np.repeat(first, np.diff(np.r_[first, I.size]))
    rank[srt] = pos
    half = (rank % 2).astype(bool)
    if split_key is not None:
        # random parity of the P1 reflections within each unique reflection
        gk = uid.astype(np.int64) * (np.int64(1) << np.int64(34)) + (split_key & np.int64((1 << 34) - 1))
        ug, ginv = np.unique(gk, return_inverse=True)
        g_uid = ug >> np.int64(34)
        srt_g = np.lexsort((rng.permutation(ug.size), g_uid))
        first_g = np.r_[0, np.flatnonzero(np.diff(g_uid[srt_g])) + 1]
        grank = np.empty(ug.size, np.int64)
        grank[srt_g] = np.arange(ug.size) - np.repeat(first_g, np.diff(np.r_[first_g, ug.size]))
        half = (grank[ginv] % 2).astype(bool)
    h1 = weighted_merge(I[~half], sig[~half], uid[~half], n)
    h2 = weighted_merge(I[half], sig[half], uid[half], n)
    both = (h1[2] > 0) & (h2[2] > 0)
    # anomalous differences
    uk_plain = uka >> 1
    is_minus = (uka & 1).astype(bool)
    centric_u, eps_u, _ = reflection_props(_unkey(uk), sg)
    # map each unique (plain) reflection to its + and - merged values
    ip = np.full(n, np.nan)
    im_ = np.full(n, np.nan)
    sp = np.full(n, np.nan)
    smn = np.full(n, np.nan)
    pos_plain = np.searchsorted(uk, uk_plain)
    ip_mask = ~is_minus
    ip[pos_plain[ip_mask]] = Ima[ip_mask]
    sp[pos_plain[ip_mask]] = sma[ip_mask]
    im_[pos_plain[~ip_mask]] = Ima[~ip_mask]
    smn[pos_plain[~ip_mask]] = sma[~ip_mask]
    # anomalous CC from half datasets: dano(half1) vs dano(half2)
    h1a = weighted_merge(I[~half], sig[~half], uida[~half], na)
    h2a = weighted_merge(I[half], sig[half], uida[half], na)
    d1p = np.full(n, np.nan); d1m = np.full(n, np.nan); d2p = np.full(n, np.nan); d2m = np.full(n, np.nan)
    v1 = h1a[2] > 0
    v2 = h2a[2] > 0
    d1p[pos_plain[ip_mask & v1]] = h1a[0][ip_mask & v1]
    d1m[pos_plain[~ip_mask & v1]] = h1a[0][~ip_mask & v1]
    d2p[pos_plain[ip_mask & v2]] = h2a[0][ip_mask & v2]
    d2m[pos_plain[~ip_mask & v2]] = h2a[0][~ip_mask & v2]
    dano1 = d1p - d1m
    dano2 = d2p - d2m
    possible, dposs, cposs = possible_unique(cell, sg, dmin, dmax)
    if shell_limits:
        lim = [x for x in sorted(shell_limits, reverse=True) if x < dmax]
        edges = np.array([dmax] + [x for x in lim if x > dmin] + [dmin])
        nbins = len(edges) - 1
    else:
        edges = shell_edges(du, nbins, dmin, dmax)
    shells = []

    def stats(selu, selo, selp, selcp):
        nu = int(selu.sum())
        no = int(selo.sum())
        npos = int(selp.sum())
        sm_mult = multi & selu
        dsum = den[sm_mult].sum()
        ok_r = dsum > 1e-6 * max(rm_num[sm_mult].sum(), 1e-30)
        rmerge = rm_num[sm_mult].sum() / dsum if ok_r else np.nan
        rmeas = rmeas_num[sm_mult].sum() / dsum if ok_r else np.nan
        rpim = rpim_num[sm_mult].sum() / dsum if ok_r else np.nan
        b = both & selu
        cc = _cc(h1[0][b], h2[0][b])
        ac = selu & ~centric_u & np.isfinite(dano1) & np.isfinite(dano2)
        ccano = _cc(dano1[ac], dano2[ac])
        dd = ip - im_
        sd = np.sqrt(sp ** 2 + smn ** 2)
        ok = selu & ~centric_u & np.isfinite(dd)
        sigano = float(np.mean(np.abs(dd[ok]) / sd[ok])) if ok.any() else np.nan
        # anomalous completeness: acentric reflections with both + and - measured
        nposs_ac = int((selp & ~selcp).sum())
        cano = float(ok.sum() / max(nposs_ac, 1)) if nposs_ac else np.nan
        return dict(n_obs=no, n_uniq=nu, n_possible=npos, completeness=nu / max(npos, 1),
                    multiplicity=no / max(nu, 1), i_over_sigma=float(np.mean(Im[selu] / sm[selu])) if nu else np.nan,
                    rmerge=rmerge, rmeas=rmeas, rpim=rpim, cc_half=cc, cc_ano=ccano, sig_ano=sigano,
                    completeness_ano=cano, mean_i=float(np.mean(Im[selu])) if nu else np.nan)

    for i in range(nbins):
        hi, lo = edges[i], edges[i + 1]
        selu = (du <= hi) & (du > lo) if i < nbins - 1 else (du <= hi) & (du >= lo * 0.9999)
        selo = selu[uid]
        selp = (dposs <= hi) & ((dposs > lo) if i < nbins - 1 else (dposs >= lo * 0.9999))
        shells.append(ShellStats(hi, lo, **stats(selu, selo, selp, cposs)))
    allu = np.ones(n, bool)
    overall = ShellStats(dmax, dmin, **stats(allu, allu[uid], np.ones(dposs.size, bool), cposs))
    merged = dict(hkl=_unkey(uk), I=Im, sig=sm, mult=mult, d=du, centric=centric_u, eps=eps_u,
                  Iplus=ip, sigIplus=sp, Iminus=im_, sigIminus=smn)
    return shells, overall, merged


def _unkey(uk):
    from .symops import key_hkl
    return key_hkl(uk)


def format_stats_table(shells, overall, anomalous=True):
    hdr = (" Resolution     Nobs   Nuniq  Compl%  Mult   I/sig   Rmerge   Rmeas    Rpim   CC1/2"
           + ("   CCano  SigAno" if anomalous else ""))
    lines = [hdr, "-" * len(hdr)]
    for s in list(shells) + [overall]:
        tag = f"{s.dmax:6.2f}-{s.dmin:5.2f}" if s is not overall else "     overall"
        line = (f"{tag} {s.n_obs:8d} {s.n_uniq:7d} {100 * s.completeness:7.1f} {s.multiplicity:5.1f} "
                f"{s.i_over_sigma:7.1f} {100 * s.rmerge:7.1f}% {100 * s.rmeas:6.1f}% {100 * s.rpim:6.1f}% "
                f"{100 * s.cc_half:6.1f}")
        if anomalous:
            line += f" {100 * s.cc_ano:7.1f} {s.sig_ano:7.2f}"
        lines.append(line)
    return "\n".join(lines)


def cc_half_cutoff(shells_fine, threshold=0.3):
    """Resolution where a tanh fit to CC1/2(s^2) drops below threshold."""
    from scipy.optimize import curve_fit
    s2 = np.array([1.0 / (0.5 * (s.dmax + s.dmin)) ** 2 for s in shells_fine])
    cc = np.array([s.cc_half for s in shells_fine])
    ok = np.isfinite(cc)
    if ok.sum() < 4:
        return None
    f = lambda x, x0, w: 0.5 * (1 - np.tanh((x - x0) / w))
    try:
        (x0, w), _ = curve_fit(f, s2[ok], np.clip(cc[ok], 0, 1), p0=[s2[ok].mean(), 0.1 * np.ptp(s2[ok]) + 1e-4],
                               maxfev=5000)
    except Exception:
        return None
    # solve f = threshold
    x = x0 + w * np.arctanh(1 - 2 * threshold)
    if not np.isfinite(x) or x <= 0:
        return None
    return float(1.0 / np.sqrt(x))


def delta_cc_half_groups(uid, I, group):
    """Delta-CC1/2 (sigma-tau method) of each group of observations (e.g. image ranges): CC1/2 of all data minus CC1/2
    without the group; negative = the group makes the data worse (Assmann et al. 2016; used per image group by
    DIALS scaling). The sigma-tau CC1/2 depends on four totals over the reflections (number with >= 2 observations,
    sum and sum of squares of their means, sum of the variances of the means): removing a group changes only the
    reflections it touches, so each group costs O(its observations), not O(all reflections)."""
    uid = np.asarray(uid)
    group = np.asarray(group)
    I = np.asarray(I, float)
    n_u = int(uid.max()) + 1
    cnt = np.bincount(uid, minlength=n_u).astype(float)
    s1 = np.bincount(uid, I, n_u)
    s2 = np.bincount(uid, I * I, n_u)

    def terms(c_, a_, b_):
        """Per-reflection (ok, mean, variance of the mean) of the sigma-tau estimate."""
        ok = c_ >= 2
        cs = np.where(ok, c_, 2.0)
        m1 = np.where(ok, a_ / cs, 0.0)
        var_h = (b_ / cs - m1 ** 2) * cs / (cs - 1)
        return ok, m1, np.where(ok, var_h / cs, 0.0)

    def cc_of(N, S1, S2, SE):
        if N <= 0:
            return np.nan
        vy = S2 / N - (S1 / N) ** 2
        ve = SE / N
        return (vy - ve) / (vy + ve) if vy + ve > 0 else np.nan

    ok0, m0, q0 = terms(cnt, s1, s2)
    N0, S10, S20, SE0 = float(ok0.sum()), float(m0[ok0].sum()), float((m0[ok0] ** 2).sum()), float(q0[ok0].sum())
    cc_all = cc_of(N0, S10, S20, SE0)
    groups = np.unique(group)
    out = np.full(groups.size, np.nan)
    order = np.argsort(group, kind="stable")
    gs = group[order]
    starts = np.searchsorted(gs, groups)
    ends = np.searchsorted(gs, groups, side="right")
    for i, (a, b) in enumerate(zip(starts, ends)):
        idx = order[a:b]
        u, inv = np.unique(uid[idx], return_inverse=True)
        k = np.bincount(inv, minlength=u.size).astype(float)
        dI = np.bincount(inv, I[idx], u.size)
        dI2 = np.bincount(inv, I[idx] ** 2, u.size)
        o1, m1, q1 = terms(cnt[u] - k, s1[u] - dI, s2[u] - dI2)
        oo, mo, qo = ok0[u], m0[u], q0[u]
        N = N0 - oo.sum() + o1.sum()
        S1 = S10 - mo[oo].sum() + m1[o1].sum()
        S2 = S20 - (mo[oo] ** 2).sum() + (m1[o1] ** 2).sum()
        SE = SE0 - qo[oo].sum() + q1[o1].sum()
        out[i] = cc_all - cc_of(N, S1, S2, SE)
    return groups, out, cc_all


def isa_empirical(uid, I, top_fraction=0.02, min_mult=3):
    """Model-free estimate of the systematic-error floor (ISa): 1 / robust sd of the relative deviations
    (I - <I>)/<I> between symmetry mates for the strongest reflections (top ``top_fraction`` by mean
    intensity, multiplicity >= ``min_mult``), where counting noise is negligible. Independent of how a
    program models its sigmas, so it compares programs fairly (error-model ISa values do not)."""
    uid = np.asarray(uid)
    n = np.bincount(uid)
    m = np.bincount(uid, I) / np.maximum(n, 1)
    ok = (n[uid] >= min_mult) & (m[uid] > 0)
    if ok.sum() < 50:
        return float("nan")
    sel = ok & (m[uid] >= np.quantile(m[uid][ok], 1 - top_fraction))
    rel = (I[sel] - m[uid][sel]) / m[uid][sel] * np.sqrt(n[uid][sel] / (n[uid][sel] - 1.0))
    sd = 1.4826 * np.median(np.abs(rel))
    return float(1.0 / sd) if sd > 0 else float("nan")

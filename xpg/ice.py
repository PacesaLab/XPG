"""Ice rings: detection from spot density and from E^2 excess bands of the integrated reflections."""
from __future__ import annotations

import numpy as np


ICE_RINGS = [3.897, 3.669, 3.441, 2.671, 2.249, 2.072, 1.948, 1.918, 1.883, 1.721]

def ice_spot_bands(s, I, sig, phi, hkl, bin_w=0.001, rel_window=0.015, e2_hot=10.0):
    """Resolution bands at ice spacings (within rel_window of 1/d) holding an excess of unique reflections whose mean
    E^2 over their repeats exceeds e2_hot (Wilson tail ~5e-5 acentric): reflections lying on a sharp ice ring,
    which the spot-density detection misses when the ring is narrow. Excess per 0.001 1/A bin against a
    running-median baseline of the rate (genuinely strong reflections are spread smoothly in resolution); a band
    needs >= 2 flagged bins and an excess of >= 10 reflections at >= 5 sigma. Returns merged (dmin, dmax) ranges."""
    from scipy.ndimage import median_filter
    from .symmetry import _friedel_canon, normalise_intensities
    ok = np.isfinite(I) & (sig > 0) & (s > 0)
    if ok.sum() < 2000:
        return []
    E2, _ = normalise_intensities(I[ok], sig[ok], s[ok], phi[ok])
    _, inv = np.unique(_friedel_canon(hkl[ok]), return_inverse=True)
    cnt = np.bincount(inv)
    Em = np.bincount(inv, E2) / cnt
    su = np.bincount(inv, s[ok]) / cnt
    hot = Em > e2_hot
    edges = np.arange(su.min(), su.max() + bin_w, bin_w)
    if edges.size < 3:
        return []
    n = np.histogram(su, edges)[0].astype(float)
    k = np.histogram(su[hot], edges)[0].astype(float)
    c = 0.5 * (edges[1:] + edges[:-1])
    ex = n * median_filter(np.where(n > 0, k / np.maximum(n, 1), 0.0), size=31, mode="nearest")
    flag = (k >= 3) & (k > 4 * ex) & (k > ex + 4 * np.sqrt(ex + 1))
    bands = []
    for dr in ICE_RINGS:
        sr = 1.0 / dr
        w = np.flatnonzero(flag & (np.abs(c - sr) < rel_window * sr))
        if not w.size:
            continue
        groups = np.split(w, np.flatnonzero(np.diff(w) > 2) + 1)
        g = max(groups, key=lambda g_: (k[g_[0]:g_[-1] + 1] - ex[g_[0]:g_[-1] + 1]).sum())
        i0, i1 = g[0], g[-1] + 1
        K, X = k[i0:i1].sum(), ex[i0:i1].sum()
        if K - X >= 10 and K - X >= 5 * np.sqrt(X + 1) and flag[i0:i1].sum() >= 2:
            bands.append((edges[i0], edges[i1]))
    out = []
    for a, b in sorted(bands):                       # merge overlapping bands (neighbouring ice spacings)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return [(float(1.0 / b), float(1.0 / a)) for a, b in out]

def detect_ice_rings(d, width=0.012, log=print, max_half_width=0.008):
    """Detect ice rings from spot density excess at hexagonal-ice d-spacings.

    Detection uses a +-``width`` window (1/A) around each ice d-spacing; the excluded range is then the
    measured extent of the excess spot density (fine histogram, contiguous bins above baseline + 3 sigma),
    at most +-``max_half_width``. Real rings are a few 1e-3 1/A wide (beam divergence, mosaicity, ice
    grain size); excluding a fixed wide band throws away good reflections. Returns (dmin, dmax) ranges."""
    if d.size < 500:
        return []
    s = 1.0 / d
    hist, edges = np.histogram(s, bins=400)
    centers = 0.5 * (edges[1:] + edges[:-1])
    from scipy.ndimage import median_filter
    base = median_filter(hist.astype(float), size=31, mode="nearest")
    out = []
    for dr in ICE_RINGS:
        sr = 1.0 / dr
        if sr < s.min() or sr > s.max():
            continue
        win = np.abs(centers - sr) < width
        if not win.any():
            continue
        obs = hist[win].sum()
        exp = base[win].sum()
        if not (obs > exp + 5 * np.sqrt(max(exp, 1)) and obs > 1.5 * exp):
            continue
        # measured ring extent: fine bins of 0.0005 1/A within +-2*width
        fe = np.arange(sr - 2 * width, sr + 2 * width + 1e-9, 0.0005)
        fh, _ = np.histogram(s, bins=fe)
        fc = 0.5 * (fe[1:] + fe[:-1])
        far = np.abs(fc - sr) > width
        bl = np.median(fh[far]) if far.any() else np.median(fh)
        hot = fh > bl + 3 * np.sqrt(max(bl, 1.0))
        k0 = int(np.argmin(np.abs(fc - sr)))
        if not hot[max(k0 - 2, 0):k0 + 3].any():
            lo_s, hi_s = sr - 0.002, sr + 0.002
        else:
            k = max(range(max(k0 - 2, 0), min(k0 + 3, fh.size)), key=lambda i: fh[i])
            i0 = i1 = k
            while i0 > 0 and hot[i0 - 1]:
                i0 -= 1
            while i1 < fh.size - 1 and hot[i1 + 1]:
                i1 += 1
            lo_s = max(fe[i0] - 0.0005, sr - max_half_width)
            hi_s = min(fe[i1 + 1] + 0.0005, sr + max_half_width)
        out.append((1.0 / hi_s, 1.0 / lo_s))
    if out:
        log("  ICE: rings detected and excluded: " + ", ".join(f"{a:.3f}-{b:.3f}" for a, b in out) + " A")
    return out

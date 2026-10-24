"""Detector setup: DETECTOR / BACKGROUND / MASK equivalents.

* static mask from the trusted-region settings (UNTRUSTED_RECTANGLE/ELLIPSE/QUADRILATERAL, TRUSTED_REGION)
* gain from temporal variance/mean of background frames
* mean background (BKGINIT) with spots removed by sigma-clipped temporal statistics
* automatic detection of shadowed regions (beamstop, arm, cryo nozzle): pixels whose background
  falls well below the azimuthal median at the same resolution are masked
  (VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS semantics)
* hot / dead / noisy pixel detection
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import Params
from .geometry import Geometry


@dataclass
class DetectorSetup:
    mask: np.ndarray          # True = trusted
    gain: float
    background: np.ndarray    # mean background per pixel (BKGINIT), masked = -1
    n_hot: int = 0
    n_dead: int = 0
    n_shadow: int = 0


def static_mask(p: Params, geom: Geometry) -> np.ndarray:
    ny, nx = geom.ny, geom.nx
    m = np.ones((ny, nx), bool)
    # pixel numbers (1-based) of array indices
    X = np.arange(nx) + 1
    Y = np.arange(ny) + 1
    for (x1, x2, y1, y2) in p.untrusted_rectangles:
        xs = (X > x1) & (X < x2)
        ys = (Y > y1) & (Y < y2)
        m[np.ix_(ys, xs)] = False
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1]
    for (x1, x2, y1, y2) in p.untrusted_ellipses:
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        ax, ay = max(0.5 * abs(x2 - x1), 0.5), max(0.5 * abs(y2 - y1), 0.5)
        m &= ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 > 1.0
    for q in p.untrusted_quadrilaterals:
        pts = np.array(q, float).reshape(4, 2)
        inside = np.ones_like(m)
        sgn = None
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            cr = (b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])
            s = cr >= 0
            inside &= s if sgn is None or sgn else ~s
            if sgn is None:
                sgn = True
        m &= ~inside
    if geom.segmented:      # pixels outside all SEGMENT= rectangles (module gaps)
        m &= geom.segmap() >= 0
    # TRUSTED_REGION: radii relative to half the (smaller) detector width, centred on ORGX/ORGY
    rmin, rmax = p.trusted_region
    R0 = 0.5 * min(nx * geom.qx, ny * geom.qy)
    r = np.hypot((xx - geom.orgx) * geom.qx, (yy - geom.orgy) * geom.qy) / R0
    m &= (r >= rmin) & (r <= rmax)
    return m


def _nanmedian_t(x, ok):
    """numpy.nanmedian along axis 0 (mean of the two middle values for even counts), NaN where no valid value."""
    import torch
    n = ok.sum(0)
    xs = torch.sort(torch.where(ok, x, torch.full_like(x, float("inf"))), dim=0).values
    lo = torch.clamp((n - 1) // 2, min=0)
    hi = torch.clamp(n // 2, min=0)
    a = torch.gather(xs, 0, lo[None]).squeeze(0)
    b = torch.gather(xs, 0, hi[None]).squeeze(0)
    return torch.where(n > 0, 0.5 * (a + b), torch.full_like(a, float("nan")))


def _temporal_stats(stack, valid):
    """Per-pixel median, 5-sigma-clipped mean / variance (ddof 1) and minimum of the valid values of a frame stack
    (numpy nan-function semantics; float64 on the compute device, processed in row bands)."""
    import torch
    from .backend import device, np_
    dev = device()
    n, ny, nx = stack.shape
    out = [np.full((ny, nx), np.nan) for _ in range(4)]
    band = max(1, int(4e7 // max(n * nx, 1)))
    for y0 in range(0, ny, band):
        x = torch.as_tensor(stack[:, y0:y0 + band], dtype=torch.float64, device=dev)
        ok = torch.as_tensor(valid[:, y0:y0 + band], device=dev)
        med = _nanmedian_t(x, ok)
        mad = _nanmedian_t(torch.abs(x - med[None]), ok)
        thr = med + 5.0 * torch.maximum(1.4826 * mad, torch.sqrt(torch.clamp(med, min=1.0)))
        okc = ok & (x <= thr[None])
        cnt = okc.sum(0).to(torch.float64)
        xs = torch.where(okc, x, 0.0)
        mean = xs.sum(0) / cnt
        var = (torch.where(okc, (x - mean[None]) ** 2, 0.0)).sum(0) / (cnt - 1)
        mean = torch.where(cnt > 0, mean, torch.nan)
        var = torch.where(cnt > 1, var, torch.nan)
        xmin = torch.where(ok, x, torch.inf).min(0).values
        xmin = torch.where(ok.any(0), xmin, torch.nan)
        for o, t in zip(out, (med, mean, var, xmin)):
            o[y0:y0 + band] = np_(t)
    return out


def _median_filter(img, size):
    """scipy.ndimage.median_filter(img, size) (mode 'reflect' = symmetric edges) on the compute device."""
    import torch
    from .backend import device, np_
    h = size // 2
    ny, nx = img.shape
    iy = np.abs(np.arange(-h, ny + h)); iy = np.where(iy >= ny, 2 * ny - 1 - iy, iy)
    iy = np.where(np.arange(-h, ny + h) < 0, -np.arange(-h, ny + h) - 1, iy)
    ix = np.abs(np.arange(-h, nx + h)); ix = np.where(ix >= nx, 2 * nx - 1 - ix, ix)
    ix = np.where(np.arange(-h, nx + h) < 0, -np.arange(-h, nx + h) - 1, ix)
    P = torch.as_tensor(np.asarray(img, np.float64)[np.ix_(iy, ix)], device=device())
    out = np.empty((ny, nx))
    band = max(1, int(2e7 // (nx * size * size)))
    for y0 in range(0, ny, band):
        y1 = min(ny, y0 + band)
        w = P[y0:y1 + 2 * h].unfold(0, size, 1).unfold(1, size, 1)      # (rows, nx, size, size)
        out[y0:y1] = np_(w.reshape(y1 - y0, nx, -1).median(dim=-1).values)
    return out


def analyse_background(images, frames, geom: Geometry, mask: np.ndarray, value_range=(6000.0, 30000.0),
                       detect_shadows=True, detect_bad_pixels=True, log=print, verbose=True):
    """Gain, mean background and refined mask from a set of (background) frames."""
    stack = images.stack(frames).astype(np.float32)         # (n, ny, nx); frames spread over sweep
    valid = (stack >= 0) & mask[None]
    n = stack.shape[0]
    # temporal sigma clipping removes Bragg spots passing through pixels (statistics on the compute device)
    med, mean, var, xmin = _temporal_stats(stack, valid)
    good = np.isfinite(mean) & np.isfinite(var) & mask & (mean > 0.5)
    # gain: median of temporal dispersion (photon counting detectors -> ~1)
    if good.sum() > 1000 and n >= 3:
        ratio = var[good] / mean[good]
        gain = float(np.clip(np.median(ratio), 0.1, 20.0))
    else:
        gain = 1.0
    newmask = mask & np.any(valid, axis=0)     # never-valid pixels (module gaps etc.)
    n_hot = n_dead = n_shadow = 0
    bkg = np.where(np.isfinite(mean), mean, -1.0).astype(np.float32)
    if detect_bad_pixels and n >= 3:
        # local reference: median filter of the background
        ref = _median_filter(np.where(good, mean, 0.0), 7)
        disp = np.where(good, var / np.maximum(mean, 1e-3), 0.0)
        # persistently high in *every* sampled frame (Bragg spots are transient)
        hot = mask & np.isfinite(xmin) & (xmin > 10 * np.maximum(ref, 1.0) + 20)
        frac_high = np.mean(np.where(valid, stack, 0) > (5 * np.maximum(ref, 0.5) + 10)[None], axis=0)
        noisy = mask & good & (disp > 20 * gain) & (frac_high >= 0.9)
        # dead: zero in all frames where the neighbourhood has counts
        allzero = np.all((stack <= 0) | ~valid, axis=0) & mask
        dead = allzero & (ref > 2.0)
        bad = hot | noisy | dead
        n_hot, n_dead = int((hot | noisy).sum()), int(dead.sum())
        newmask &= ~bad
    if detect_shadows and good.sum() > 1000:
        # azimuthal median of background as function of resolution
        dmap = geom.resolution_map()
        s = 1.0 / dmap
        ys, xs = np.mgrid[0:geom.ny, 0:geom.nx]
        _, pol = geom.lp_factor(geom.pixel_s_map())
        mean = np.where(np.isfinite(mean), mean / pol, mean)   # remove polarisation anisotropy
        nb = 200
        # radial reference from *all* trusted pixels (low-background data: median counts are 0, so a
        # median or a "mean > 0.5" selection is biased); upper-trimmed mean of each resolution bin
        okb = np.isfinite(mean) & newmask
        edges = np.quantile(s[okb], np.linspace(0, 1, nb + 1))
        idx = np.clip(np.searchsorted(edges, s) - 1, 0, nb - 1)
        med_r = np.full(nb, np.nan)
        for i in range(nb):
            v = mean[okb & (idx == i)]
            if v.size > 20:
                v = np.sort(v)[: max(int(0.9 * v.size), 1)]       # trim the top 10% (residual spots)
                med_r[i] = v.mean() / 0.97                         # bias of trimming ~3% for Poisson-like
        med_r = np.where(np.isfinite(med_r), med_r, np.nanmedian(med_r))
        expected = med_r[idx]
        # smooth the ratio to avoid masking individual noisy pixels; low counts need larger windows
        mean_s = ndimage.uniform_filter(np.where(np.isfinite(mean) & mask, mean, 0.0), 9)
        wsum = ndimage.uniform_filter((np.isfinite(mean) & mask).astype(float), 9)
        mean_s = mean_s / np.maximum(wsum, 1e-6)
        ratio = mean_s / np.maximum(expected, 1e-6) * 10000.0
        lo, hi = value_range
        # only judge where the expected background is statistically meaningful
        meaningful = (expected * n * 81 > 400) & (wsum > 0.5) & np.isfinite(mean)
        shadow = mask & meaningful & (ratio < lo)
        shadow = ndimage.binary_opening(shadow, iterations=2)
        shadow = ndimage.binary_dilation(shadow, iterations=2) & mask
        n_shadow = int(shadow.sum())
        newmask &= ~shadow
    bkg[~newmask] = -1.0
    if verbose:
        log(f"  BACKGROUND: gain {gain:.3f}; masked {n_hot} hot/noisy, {n_dead} dead, {n_shadow} shadowed pixels; "
            f"{100 * (1 - newmask.mean()):.2f}% of detector untrusted")
    return DetectorSetup(newmask, gain, bkg, n_hot, n_dead, n_shadow)


def background_template(geom: Geometry, background: np.ndarray, mask: np.ndarray, n_bins=1500, smooth=51):
    """Static spatial shape of the background, for integration (b_pixel = k * template, k fitted per shoebox).

    The background structure (ice and solvent rings, air/solvent scattering vs resolution, polarisation,
    sensor efficiency, shadow edges) is static during a sweep while its level changes, so a scaled template
    describes shoeboxes crossed by rings far better than a constant. Built from the BACKGROUND mean background:
    fine-binned radial profile (resolution) x polarisation x large-scale 2D correction (anisotropy),
    because raw per-pixel means from a few frames are far too noisy at low counts. Normalised to mean 1."""
    ok = mask & np.isfinite(background) & (background >= 0)
    if ok.sum() < 1000:
        return None
    ys, xs = np.mgrid[0:geom.ny, 0:geom.nx]
    S = geom.pixel_s_map()
    s = np.linalg.norm(S - geom.s0, axis=-1)
    _, pol = geom.lp_factor(S)
    v = np.where(ok, background / np.maximum(pol, 1e-3), np.nan)
    edges = np.quantile(s[ok], np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, s) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx[ok], minlength=n_bins)
    tot = np.bincount(idx[ok], v[ok], n_bins)
    prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    good = np.isfinite(prof)
    centres = 0.5 * (edges[1:] + edges[:-1])
    prof = np.interp(centres, centres[good], prof[good])
    model = prof[idx] * pol
    ratio = np.where(ok & (model > 0), background / np.maximum(model, 1e-6), 0.0)
    wsum = ndimage.uniform_filter(ok.astype(float), smooth)
    rs = ndimage.uniform_filter(ratio, smooth) / np.maximum(wsum, 1e-6)
    rs = np.clip(np.where(wsum > 0.05, rs, 1.0), 0.5, 2.0)
    t = model * rs
    t = t / np.mean(t[ok])
    t = np.where(np.isfinite(t) & (t > 0), t, 1.0)
    return t.astype(np.float32)

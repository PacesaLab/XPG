"""Forward-model integration of a sweep (or wedges of it): consecutive frame windows, optionally split into detector
tiles, each fitted jointly (intensities + background; options: local / per-pixel background, geometry, widths).
Used by the INTEGRATE stage (XPG_INTEGRATE_MODE=fm).

Each reflection is fitted in the first window that contains its whole support with its centre away from the window
edges (margin) and, with tiles, in the tile whose core contains its centre; reflections centred in the margins enter
the model at their current values (block Gauss-Seidel over consecutive windows / tiles).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .fit import WindowFit, fit_geometry, fit_widths, set_theta, theta_of
from .model import (GEOM, WindowModel, auto_window, make_window, memory_budget, plan_tiles, support,
                    tile_bounds)


def superpixel_side(refl, g, sigma_d, t, s_max=4, q=0.02):
    """Super-pixel side s for the forward model (one per sweep): the largest s <= s_max with s <= sigma / t for all
    but the q smallest spots (sigma: the detector footprint of sigma_D, in pixels, at distance and obliquity, as
    the integrator sizes its boxes). Returns (s, sigma quantile in px)."""
    Xv = g.pixel_to_lab(refl["x"], refl["y"])
    L = np.linalg.norm(Xv, axis=1)
    cosr = np.abs(Xv @ g.ed3) / L
    spx = L * np.radians(sigma_d) / min(g.qx, g.qy) / np.maximum(cosr, 0.4)
    sq = float(np.quantile(spx, q)) if spx.size else 0.0
    return int(np.clip(np.floor(sq / t), 1, s_max)) if t > 0 else 1, sq


def bin_pixels(a, s, dev=None, flags=True):
    """s x s sums of a 2D array (rows / columns beyond a whole super-pixel dropped). flags: a super-pixel with a
    negative pixel is flagged -2 (overloaded, first) or -1 (masked), as the integrator's super-pixels."""
    t = torch.as_tensor(np.asarray(a), dtype=torch.float32)
    if dev is not None:
        t = t.to(dev)
    t = t[None, None]
    out = F.avg_pool2d(t, s, divisor_override=1)
    if flags:
        mn = -F.max_pool2d(-t, s)
        out = torch.where(mn < 0, torch.where(mn <= -1.5, -2.0, -1.0), out)
    return out[0, 0].cpu().numpy()


def read_frames(images, f0, f1, ny, nx, mask=None, prev=None, pxs=1, dev=None):
    """Frames [f0, f1) (0-based) as (n, ny, nx): int16 when raw counts (-1 masked, -2 overloaded), else float32.
    prev (first frame, frames): frames read for the previous window; consecutive windows overlap by their margins,
    and the frames they share are copied instead of decoded again. New frames are decoded by the image set's
    worker pool (iter_frames: decode processes into shared memory), not one by one. pxs > 1: s x s super-pixel
    sums (bin_pixels, on dev), ny, nx the super-pixel detector's."""
    frames = None
    todo = []
    for k in range(f1 - f0):
        j = f0 + k - prev[0] if prev is not None else -1
        if prev is None or not (0 <= j < len(prev[1])):
            todo.append(k)

    def put(k, fr, encoded):
        nonlocal frames
        if not encoded and mask is not None:
            fr = np.where(mask, fr, -1)
        if not encoded and pxs > 1:
            fr = bin_pixels(fr, pxs, dev)
        # int16 when the frame holds integers in range: tested on its int16 conversion
        ci = fr.astype(np.int16) if fr.dtype != np.int16 else fr
        ints = bool(fr.max() < 32767 and fr.min() >= -32768 and np.array_equal(ci, fr))
        if frames is None:
            frames = np.empty((f1 - f0, ny, nx), np.int16 if ints else np.float32)
        elif frames.dtype == np.int16 and not ints:
            frames = frames.astype(np.float32)                  # a frame beyond int16: float32 for all
        if frames.dtype == np.int16:
            frames[k] = ci
            neg = frames[k] < 0
            if neg.any():
                frames[k][neg] = np.where(frames[k][neg] == -2, -2, -1)
        else:
            frames[k] = fr
    if prev is not None:
        for k in range(f1 - f0):
            j = f0 + k - prev[0]
            if 0 <= j < len(prev[1]):
                put(k, prev[1][j], True)                        # already masked and encoded
    if todo:
        nums = [f0 + k + 1 for k in todo]
        # copy=False: each frame is stored (converted) before the next one is requested
        it = images.iter_frames(nums, copy=False) if hasattr(images, "iter_frames") else ((n, images.get(n)) for n in nums)
        for n, fr in it:
            put(n - f0 - 1, np.asarray(fr), False)
    return frames


def tile_counts(frames, ty0, ty1, tx0, tx1, dev):
    """The window's counts on a detector tile as one flat tensor. int16 frames and a tile spanning the full width (row
    bands contiguous in memory) on a GPU: copied band by band straight into a device buffer, which WindowFit adopts
    as its store; otherwise a contiguous host copy as before."""
    dev = torch.device(dev)
    if frames.dtype == np.int16 and dev.type != "cpu" and tx0 == 0 and tx1 == frames.shape[2]:
        n = (ty1 - ty0) * frames.shape[2]
        out = torch.empty(frames.shape[0] * n, dtype=torch.int16, device=dev)
        for k in range(frames.shape[0]):
            out[k * n:(k + 1) * n].copy_(torch.from_numpy(frames[k, ty0:ty1]).reshape(-1))
        return out
    return torch.as_tensor(np.ascontiguousarray(frames[:, ty0:ty1, tx0:tx1]).reshape(-1))


def calibrate_widths(refl, model, blocks, geom, images, wedges, *, mask, bkg_template, profiles, box_n, block_params,
                     dev, budget, support_frac, tl_bounds, n_windows=2, kappa_min=1.3, w_min=0.02, log=print,
                     pxs=1, tmpl=None):
    """Widths and halo for the whole sweep (A7) by likelihood on a few short windows of the central detector tile:
    (sigma_D, sigma_M, halo w, kappa) fitted jointly (M2), all reflections of the windows in the model. The halo is
    used only if it is one (kappa >= kappa_min, w >= w_min); a narrow 'halo' (kappa < 1, a sharper core) is not.
    Returns [sigma_D, sigma_M, w, kappa] (degrees) or None (keep the integrator's widths)."""
    g = geom
    nx, ny = g.nx // pxs, g.ny // pxs
    tmpl = bkg_template if pxs == 1 else tmpl
    ix0, iy0, iz0, bx, dz = support(refl, support_frac, pxs)
    margin = int(np.ceil(np.median(dz) / 2))
    # central tile, frames: short windows (the width Jacobian holds ~8 window-size arrays)
    cx, cy = nx / 2, ny / 2
    (_, tile) = min(tl_bounds, key=lambda b: abs((b[0][0] + b[0][1]) / 2 - cx) + abs((b[0][2] + b[0][3]) / 2 - cy))
    tx0, tx1, ty0, ty1 = tile
    npx = (tx1 - tx0) * (ty1 - ty0)
    w_need = 2 * margin + 10
    W = int(np.clip(0.5 * budget / (40.0 * npx), w_need, 60))
    w0, w1 = int(wedges[0][0]) - 1, int(wedges[0][1])
    if w1 - w0 < W:
        return None
    starts = np.linspace(w0, w1 - W, n_windows + 2)[1:-1].astype(int)
    fits, Is = [], []
    I0 = np.nan_to_num(np.asarray(refl["iprf"], np.float32), nan=0.0)
    zc = refl["z"]
    for f0 in starts:
        f1 = f0 + W
        sel = (refl["fb"] == 1) & (iz0 >= f0) & (iz0 + dz <= f1) & (zc >= f0 + margin) & (zc < f1 - margin) \
            & (ix0 >= tx0) & (ix0 + bx <= tx1) & (iy0 >= ty0) & (iy0 + bx <= ty1)
        owned = np.flatnonzero(sel)
        if owned.size < 200:
            continue
        frames = read_frames(images, f0, f1, ny, nx, mask, pxs=pxs, dev=dev)
        win = make_window(refl, f0, f1, nx, ny, frac=support_frac, tile=tile, pxs=pxs)
        wm = WindowModel(refl, win, blocks, model, bkg_template, dev, profiles=profiles, n_box=box_n,
                         block_params=block_params, budget=budget)
        set_theta(wm, theta_of(wm))                             # halo on (w 0.05, kappa 2.5)
        ct = tile_counts(frames, ty0, ty1, tx0, tx1, wm.dev)
        gt = max(4, int(np.ceil(48 * max((tx1 - tx0) / nx, (ty1 - ty0) / ny))))
        fit = WindowFit(wm, ct, tmpl, f1 - f0, owned=owned, grid=gt, log=lambda *_: None)
        del ct, frames
        I, _ = fit.run(torch.as_tensor(I0, device=wm.dev), n_outer=2, log_every=False)
        fits.append(fit)
        Is.append(I)
    if not fits:
        return None
    th, _, _ = fit_widths(fits, Is, n_iter=5, n_free=4, log=lambda *_: None)
    sd, sm = float(np.degrees(np.exp(th[0]))), float(np.degrees(np.exp(th[1])))
    w, kap = float(1.0 / (1.0 + np.exp(-th[2]))), float(np.exp(th[3]))
    del fits, Is
    if dev is not None and torch.device(dev).type == "cuda":
        torch.cuda.empty_cache()
    if kap >= kappa_min and w >= w_min:
        log(f"  FM: halo calibrated on {len(starts)} x {W} frames: sigma_D {sd:.4f} sigma_M {sm:.4f} deg, halo w "
            f"{w:.3f} kappa {kap:.2f} -> used")
        return [sd, sm, w, kap]
    log(f"  FM: halo calibration: w {w:.3f} kappa {kap:.2f} (not a halo) -> integrator's widths kept")
    return None


def integrate_sweep(refl, model, blocks, geom, images, wedges, *, mask=None, bkg_template=None, profiles=None,
                    box_n=5.0, block_params=None, window="auto", mem_gb=None, tiles="auto", outer=2, device=None,
                    support_frac=0.8, widths=None, geometry=False, gamma0=None, geo_free=None, I_start=None,
                    grid=48, bg_block=None, outer_tol=0.0, max_deg=16.0, superpixel=0.0, log=print):
    """Forward-model intensities of the reflections in refl over the wedges [(first, last), ...] (1-based frames).

    Returns dict(I, sig (per reflection, NaN where not fitted; counts, before the LP factor), done, recovered
    (overloaded reflections fitted from their unsaturated pixels), geo_log, geo_names, window, tiles, seconds).
    superpixel t > 0: the fit runs on s x s super-pixel sums (Poisson counts still) with s the largest side
    <= sigma / t of all but the smallest 2% of spots (superpixel_side; s^2 fewer voxels)."""
    from ..integrate import _kernel_device
    g = geom
    dev = torch.device(device) if isinstance(device, str) else device
    kdev = dev if dev is not None else _kernel_device()
    budget = memory_budget(kdev, mem_gb)
    pxs = 1
    if superpixel and superpixel > 0 and len(refl["x"]):
        sd0 = widths[0] if isinstance(widths, (list, tuple)) else model.sigma_d
        pxs, sq = superpixel_side(refl, g, sd0, superpixel)
        log(f"  FM: super-pixels {pxs}x{pxs} (spot sigma {sq:.1f} px at the 2% quantile, t {superpixel:g})")
    nx, ny = g.nx // pxs, g.ny // pxs
    tmpl = bkg_template if pxs == 1 or bkg_template is None else bin_pixels(bkg_template, pxs, flags=False)
    ix0, iy0, iz0, bx, dz = support(refl, support_frac, pxs)
    zc = refl["z"]
    n = len(zc)
    I_cur = np.nan_to_num(np.asarray(refl["iprf"] if I_start is None else I_start, np.float32), nan=0.0)
    I_fm, s_fm = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    done, rec_fm = np.zeros(n, bool), np.zeros(n, bool)
    # detector tiles: margin = half the largest supports (99%), so that a reflection centred in a tile's core has its
    # whole support in the tile
    mxy = int(np.ceil(np.quantile(bx, 0.99) / 2)) + 2
    W_info = None
    if window == "auto":
        W, W_info = auto_window(refl, nx, ny, budget, kdev.type, frac=support_frac, pxs=pxs)
        # at most max_deg of rotation per window
        tile_for_angle = False
        if max_deg and abs(g.osc) > 0:
            W_deg = max(int(round(max_deg / abs(g.osc))), W_info["w_min"])
            if W > W_deg:
                W = W_deg
            elif W < W_deg and tiles == "auto" and W_info["fits"]:
                # memory limits full-detector windows below the angular target (large detectors): tile the detector so
                # that W_deg-frame windows fit
                nt = plan_tiles(nx, ny, W_deg, budget, kdev.type, mxy, max_tiles=16)
                if nt is not None and nt != (1, 1):
                    W, tile_for_angle = W_deg, True
        log(f"  FM: memory budget {W_info['budget_gb']:.1f} GiB on {kdev.type}: window {W} frames (memory allows "
            f"{W_info['w_mem']} over the whole detector, supports need >= {W_info['w_min']}, at most {max_deg:g} deg)")
    else:
        W = int(window)
    if tiles != "auto":
        ntx, nty = (int(v) for v in str(tiles).lower().split("x"))
    elif W_info is not None and (not W_info["fits"] or tile_for_angle):
        nt = plan_tiles(nx, ny, W, budget, kdev.type, mxy)
        if nt is None:
            raise RuntimeError(f"forward model: no detector tiling fits {budget / 2 ** 30:.1f} GiB with {W}-frame windows")
        ntx, nty = nt
    else:
        ntx, nty = 1, 1
    tl_bounds = tile_bounds(nx, ny, ntx, nty, mxy)
    if len(tl_bounds) > 1:
        tw = max(t[1][1] - t[1][0] for t in tl_bounds)
        th = max(t[1][3] - t[1][2] for t in tl_bounds)
        log(f"  FM: detector tiles {ntx} x {nty} (up to {tw} x {th} px incl. a {mxy}-px margin)")
    gamma = None if gamma0 is None else np.array(gamma0, float)
    geo_free = np.arange(6) if geo_free is None else np.asarray(geo_free)
    names, geo_log = list(GEOM), []
    margin = int(np.ceil(np.median(dz) / 2))
    step = W - 2 * margin
    if step < 10:
        raise RuntimeError(f"forward model: window {W} too short for supports of ~{np.median(dz):.0f} frames")
    t_all = time.time()
    widths_used = widths
    if isinstance(widths, str) and widths == "auto":
        widths = calibrate_widths(refl, model, blocks, g, images, wedges, mask=mask, bkg_template=bkg_template,
                                  profiles=profiles, box_n=box_n, block_params=block_params, dev=dev or kdev,
                                  budget=budget, support_frac=support_frac, tl_bounds=tl_bounds, log=log, pxs=pxs,
                                  tmpl=tmpl)
        widths_used = widths
    cxr, cyr = ix0 + bx // 2, iy0 + bx // 2                     # support centre (array column / row)
    prev = None
    for w0, w1 in wedges:
        f_lo, f_hi = int(w0) - 1, int(w1)                     # 0-based [f_lo, f_hi)
        s = f_lo
        while s < f_hi:
            f0, f1 = s, min(s + W, f_hi)
            inner_lo = f0 + (margin if f0 > f_lo else 0)
            inner_hi = f1 - (margin if f1 < f_hi else 0)
            sel_w = (~done) & (refl["fb"] == 1) & (iz0 >= f0) & (iz0 + dz <= f1) & (zc >= inner_lo) & (zc < inner_hi)
            frames = None
            if prev is not None and not (prev[0] < f1 and prev[0] + len(prev[1]) > f0):
                prev = None
            for (cx0, cx1, cy0, cy1), (tx0, tx1, ty0, ty1) in tl_bounds:
                sel = sel_w & (cxr >= cx0) & (cxr < cx1) & (cyr >= cy0) & (cyr < cy1) & (ix0 >= tx0) \
                    & (ix0 + bx <= tx1) & (iy0 >= ty0) & (iy0 + bx <= ty1)
                owned = np.flatnonzero(sel)
                if not owned.size:
                    continue
                t0 = time.time()
                if frames is None:                           # the window's frames, read once
                    frames = read_frames(images, f0, f1, ny, nx, mask, prev=prev, pxs=pxs, dev=kdev)
                    prev = None
                win = make_window(refl, f0, f1, nx, ny, frac=support_frac, tile=(tx0, tx1, ty0, ty1), pxs=pxs)
                wm = WindowModel(refl, win, blocks, model, bkg_template, dev, profiles=profiles, n_box=box_n,
                                 block_params=block_params, budget=budget)
                if widths:
                    wm.set_widths(sd=float(np.radians(widths[0])), sm=float(np.radians(widths[1])),
                                  halo=(widths[2], widths[3]) if len(widths) == 4 else None)
                ct = tile_counts(frames, ty0, ty1, tx0, tx1, wm.dev)
                # the smooth background grid (grid nodes per axis over the whole detector) keeps the full detector's
                # node spacing on a tile
                gt = max(4, int(np.ceil(grid * max((tx1 - tx0) / nx, (ty1 - ty0) / ny))))
                fit = WindowFit(wm, ct, tmpl, f1 - f0, owned=owned, grid=gt, log=lambda *_: None,
                                bg_block_frames=bg_block,
                                chunk_voxels=float(np.clip(min(0.05 * budget / 20.0, win.n_vox / 8.0), 1e7, 2e8)))
                del ct
                if geometry:
                    names = list(wm.geom_names)
                    gamma = wm.geom if gamma is None else gamma
                    wm.set_geometry(gamma)                    # warm start: the previous window's correction
                I, sig = fit.run(torch.as_tensor(I_cur, device=wm.dev), n_outer=outer, log_every=False,
                                 outer_tol=outer_tol, max_outer=4)
                if geometry:
                    gamma, gsd, _, I = fit_geometry(fit, I, n_iter=3, free=geo_free, log=lambda *_: None)
                    I, sig = fit.run(I, n_outer=1, log_every=False)
                    geo_log.append(dict(frames=[f0 + 1, f1], gamma=gamma.tolist(),
                                        sd=gsd.tolist() if gsd is not None else None))
                    log(f"    geometry: rot {np.degrees(gamma[:3]) * 1e3} mdeg, shift {gamma[3:6] * 1e3} um")
                I, sig = I.cpu().numpy(), sig.cpu().numpy()
                I_fm[owned], s_fm[owned] = I[owned], sig[owned]
                rec_fm[owned] = fit.recovered.cpu().numpy()[owned]
                I_cur[owned] = I[owned]
                done[owned] = True
                tl = f" tile x {tx0}-{tx1} y {ty0}-{ty1}" if len(tl_bounds) > 1 else ""
                ss_ = getattr(fit, "solve_stats", None)
                log(f"  frames {f0 + 1}-{f1}{tl}: {owned.size} reflections fitted ({win.rows.size} in the model), "
                    f"{time.time() - t0:.0f}s" + (f"; Newton steps {ss_['newton']}, no decrease {ss_['ls_fail']}, "
                                                 f"CG at cap {ss_['cg_cap']}" if ss_ else ""))
                del wm, fit
                if kdev.type == "cuda":
                    torch.cuda.empty_cache()
            # keep only the frames the next window shares (a copy, so that the window's array is freed)
            prev = (f0 + step, np.array(frames[step:])) if frames is not None and f1 < f_hi else None
            del frames
            if f1 >= f_hi:
                break
            s += step
    secs = time.time() - t_all
    log(f"FM: {done.sum()} reflections in {secs:.0f}s ({rec_fm.sum()} overloaded ones recovered from their "
        f"unsaturated pixels)")
    return dict(I=I_fm, sig=s_fm, done=done, recovered=rec_fm, geo_log=geo_log, geo_names=names,
                window=W, tiles=(ntx, nty), seconds=secs, widths=widths_used, superpixel=pxs)

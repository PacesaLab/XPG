"""Integration kernels: per-pixel detector tables, the Kabsch frame of each shoebox pixel, the rocking-curve basis,
and the profile-fitting kernels (pass 1, profile accumulation, pass 2) built per shoebox shape and cached."""
from __future__ import annotations

import numpy as np
import torch

from . import tnp as jnp
from .backend import np_
from .geometry import PIX0, Geometry
from .tnp import erf as jerf


N_FG = 3.0       # reference foreground radius (sigma units)
N_BOX = 5.0      # shoebox half-size in sigma units (detector plane)
FG_R = N_FG      # summation foreground radius (sigma units)
BG_R = N_FG + 0.5   # inner radius of the background region
PROFILE_GRID = 41   # fine grid: bilinear accumulate+sample smoothing biased weak intensities at 15
N_IRLS = 6       # Poisson-weighted profile-fit iterations
# IRLS stopping rule: after the second iteration, stop when no intensity of the batch moves by more than IRLS_TOL of
# its standard deviation (0: always N_IRLS iterations)
IRLS_TOL = 1e-3
# Per-pixel Poisson variance = expected counts; the floor only guards divisions. A floor of 0.5 counts dominated
# low-background photon-counting data, inflating the sigmas of weak reflections several-fold (error model then pinned
# at SdFac 0.3) and mis-weighting the profile fit.
VAR_FLOOR = 1e-3


_DET_TABLES = {}   # per-pixel detector tables on the kernel device: key -> (source object, tensor)


def _det_table(src, make, key_extra=()):
    """Device tensor of a per-pixel table (background template, segment map, geometric corrections), made once per
    source object and device: the forward model asked for the detector constants once per chunk, each time moving
    the 4150 x 4371 template to the GPU again (119 s of a 1612 s FM stage)."""
    dev = str(jnp.asarray(np.float32(0)).device)
    key = (id(src), dev) + tuple(key_extra)
    hit = _DET_TABLES.get(key)
    if hit is not None and hit[0] is src:
        return hit[1]
    t = make()
    if len(_DET_TABLES) > 32:
        _DET_TABLES.clear()
    _DET_TABLES[key] = (src, t)
    return t


def _detector_consts(g: Geometry, bkg_template=None):
    """Detector description for the kernels: per-segment affine maps (one segment for a flat
    single-panel detector), segment map and optional geometric-correction tables."""
    P0, Bx, By = g.panels()
    f32 = lambda a: jnp.asarray(np.asarray(a, float), jnp.float32)
    U1, U2 = Bx / g.qx, By / g.qy
    geo = g.geo_corr
    one16, one32 = np.zeros((1, 1), np.int16), np.zeros((1, 1))
    segmap = (_det_table(g.segments, lambda: jnp.asarray(g.segmap()), ("segmap", g.nx, g.ny)) if g.segmented
              else jnp.asarray(one16))
    return dict(orgx=g.orgx, orgy=g.orgy, P0=f32(P0), Bx=f32(Bx), By=f32(By), U1=f32(U1), U2=f32(U2),
                UN=f32(np.cross(U1, U2)), qx=g.qx, qy=g.qy, segmap=segmap,
                gx=_det_table(geo, lambda: f32(geo[0]), ("gx",)) if geo is not None else f32(one32),
                gy=_det_table(geo, lambda: f32(geo[1]), ("gy",)) if geo is not None else f32(one32),
                wl=g.wavelength, tsens=float(max(g.sensor_thickness, 1e-6)), mu=float(max(g.mu, 1e-6)),
                plx=float(g.has_parallax),
                bkg=(_det_table(bkg_template, lambda: jnp.asarray(bkg_template), ("bkg",)) if bkg_template is not None
                     else jnp.asarray(np.ones((1, 1), np.float32))))


def _bkg_ref(det, x, y):
    """Background template at shoebox pixel centres (x: (N,1,1,Dx), y: (N,1,Dy,1)); 1 if no template."""
    if det["bkg"].shape == (1, 1):
        return jnp.ones(jnp.broadcast_shapes(x.shape, y.shape), jnp.float32)
    ny, nx = det["bkg"].shape
    ix = jnp.clip(jnp.round(x - 1.0).astype(jnp.int32), 0, nx - 1)
    iy = jnp.clip(jnp.round(y - 1.0).astype(jnp.int32), 0, ny - 1)
    return det["bkg"][iy, ix]


def _pixel_S(det, x, y):
    from .detector import bilinear_table, lookup_segment
    x, y = jnp.broadcast_arrays(x, y)
    # shapes are static under jit: single-panel detectors skip the per-pixel gathers
    if det["segmap"].shape == (1, 1):
        P0, Bx, By, U1, U2, UN = (det[n][0] for n in ("P0", "Bx", "By", "U1", "U2", "UN"))
    else:
        k = jnp.maximum(lookup_segment(det["segmap"], x, y, xp=jnp), 0).astype(jnp.int64)   # torch indexes with int64
        P0, Bx, By = det["P0"][k], det["Bx"][k], det["By"][k]
        U1, U2, UN = det["U1"][k], det["U2"][k], det["UN"][k]
    if det["gx"].shape == (1, 1):
        xc, yc = x, y
    else:
        xc = x + bilinear_table(det["gx"], x, y, xp=jnp)
        yc = y + bilinear_table(det["gy"], x, y, xp=jnp)

    def plane(xx, yy):
        return P0 + (xx - det["orgx"])[..., None] * Bx + (yy - det["orgy"])[..., None] * By
    X = plane(xc, yc)
    for _ in range(2 if det["plx"] else 0):   # invert thick-sensor parallax (none without a sensor model: CCD)
        e = X / jnp.linalg.norm(X, axis=-1, keepdims=True)
        cosa = jnp.clip(jnp.sum(e * UN, -1), 1e-3, 1.0)
        L = det["tsens"] / cosa
        muL = det["mu"] * L
        lbar = 1.0 / det["mu"] - L * jnp.exp(-muL) / jnp.maximum(-jnp.expm1(-muL), 1e-12)
        d1, d2 = lbar * jnp.sum(e * U1, -1), lbar * jnp.sum(e * U2, -1)
        X = plane(xc - det["plx"] * d1 / det["qx"], yc - det["plx"] * d2 / det["qy"])
    return X / (jnp.linalg.norm(X, axis=-1, keepdims=True) * det["wl"])


def _eps12(det, S, e1, e2, x, y):
    """eps1, eps2 (radians) of detector points (x, y) for reflections (S, e1, e2).

    x: (N, Dx), y: (N, Dy) -> (N, Dy, Dx)."""
    Sp = _pixel_S(det, x[:, None, :], y[:, :, None])                       # (N, Dy, Dx, 3)
    d = Sp - S[:, None, None, :]
    invS = 1.0 / jnp.linalg.norm(S, axis=-1)[:, None, None]
    return jnp.einsum("nyxk,nk->nyx", d, e1) * invS, jnp.einsum("nyxk,nk->nyx", d, e2) * invS


# Rocking curve: non-negative mixture of K Gaussians (centres RC_C, width RC_W, in units of sigma_M)
# whose weights are learnt per phi block. With the default weights the mixture is a unit Gaussian;
# frame integration stays exact (erf differences) for any slicing.
RC_C = np.linspace(-3.5, 3.5, 21)
RC_W = 0.25


def rc_default_weights():
    w = np.exp(-0.5 * RC_C ** 2 / (1 - RC_W ** 2))
    return (w / w.sum()).astype(np.float32)


def _phi_rc(phi_lo, phi_hi, phi_c, zeta, sm, rcw=None):
    """Fraction of the rocking curve (scale sm/|zeta|) recorded between phi_lo and phi_hi."""
    s = sm / jnp.maximum(jnp.abs(zeta), 1e-3)
    if rcw is None:
        a = (phi_hi - phi_c) / (s * jnp.sqrt(2.0))
        b = (phi_lo - phi_c) / (s * jnp.sqrt(2.0))
        return 0.5 * (jerf(a) - jerf(b))
    c = jnp.asarray(RC_C, jnp.float32)
    sw = (s * RC_W * jnp.sqrt(2.0))[..., None]
    a = (phi_hi[..., None] - phi_c[..., None] - c * s[..., None]) / sw
    b = (phi_lo[..., None] - phi_c[..., None] - c * s[..., None]) / sw
    return jnp.sum(rcw * 0.5 * (jerf(a) - jerf(b)), axis=-1)


def rc_basis_np(phi_lo, phi_hi, phi_c, zeta, sm):
    """Frame integrals of each mixture component (N, Dz, K), numpy (for fitting weights)."""
    from scipy.special import erf
    s = (sm / np.maximum(np.abs(zeta), 1e-3))[:, None, None]
    sw = s * RC_W * np.sqrt(2.0)
    a = (phi_hi[..., None] - phi_c[:, None, None] - RC_C * s) / sw
    b = (phi_lo[..., None] - phi_c[:, None, None] - RC_C * s) / sw
    return 0.5 * (erf(a) - erf(b))


def _bilinear(grid, u1, u2, umax):
    """Sample a (R, G, G) grid at normalised coords u in [-umax, umax]; region index r."""
    G = grid.shape[-1]
    f1 = (u1 + umax) / (2 * umax) * (G - 1)
    f2 = (u2 + umax) / (2 * umax) * (G - 1)
    inside = (f1 >= 0) & (f1 <= G - 1) & (f2 >= 0) & (f2 <= G - 1)
    f1 = jnp.clip(f1, 0, G - 1 - 1e-4)
    f2 = jnp.clip(f2, 0, G - 1 - 1e-4)
    i1 = jnp.floor(f1).astype(jnp.int32)
    i2 = jnp.floor(f2).astype(jnp.int32)
    t1 = f1 - i1
    t2 = f2 - i2
    return inside, i1, i2, t1, t2


def H(r):
    """Host copies of a kernel result's per-reflection (1-D) outputs, fetched in ONE transfer and cached in the dict
    (one device -> host copy per output synchronised the device ~20 times per chunk)."""
    h = r.get("_h")
    if h is None:
        keys = [k for k, v in r.items() if torch.is_tensor(v) and v.dim() == 1]
        h = {}
        if keys:
            dev = r[keys[0]].device
            ft = torch.float32 if dev.type == "mps" else torch.float64
            st = np_(torch.stack([r[k].to(ft) for k in keys]))
            dt = {k: np.dtype(str(r[k].dtype).replace("torch.", "")) for k in keys}
            h = {k: st[i].astype(dt[k]) for i, k in enumerate(keys)}
        r["_h"] = h
    return h


def _neighbour_mask(xx, yy, zz, nb, shape, skip_first=None):
    """Voxels inside any neighbour's foreground ellipsoid (shared by pass 1, pass 2 and the 3D accumulation).
    xx (N,1,1,Dx), yy (N,1,Dy,1), zz (N,Dz,1,1); nb: per-reflection neighbour centres / widths (N, K)."""
    nbm = jnp.zeros(shape, bool)
    # smallest normalised distance of each neighbour's centre to each box (separable): a neighbour column whose
    # ellipsoid misses every box of the chunk adds no voxel and is skipped (exact; most far neighbours)
    lo_x, hi_x = xx[:, 0, 0, 0], xx[:, 0, 0, -1]
    lo_y, hi_y = yy[:, 0, 0, 0], yy[:, 0, -1, 0]
    lo_z, hi_z = zz[:, 0, 0, 0], zz[:, -1, 0, 0]
    for kk in range(nb["x"].shape[1]):
        nx_, ny_, nz_ = nb["x"][:, kk], nb["y"][:, kk], nb["z"][:, kk]
        mx = (jnp.clip(nx_, lo_x, hi_x) - nx_) / nb["sx"][:, kk]
        my = (jnp.clip(ny_, lo_y, hi_y) - ny_) / nb["sx"][:, kk]
        mz = (jnp.clip(nz_, lo_z, hi_z) - nz_) / nb["sz"][:, kk]
        hit = mx * mx + my * my + mz * mz < 1.0
        n_hit = int(hit.sum())
        if n_hit == 0:
            continue
        # only the boxes this neighbour column reaches (exact: the others add no voxel); uncrowded data: a few
        rows = torch.nonzero(hit).squeeze(1) if n_hit < hit.numel() // 2 else slice(None)
        dx = (xx[rows] - nb["x"][rows, kk, None, None, None]) / nb["sx"][rows, kk, None, None, None]
        dy = (yy[rows] - nb["y"][rows, kk, None, None, None]) / nb["sx"][rows, kk, None, None, None]
        dz = (zz[rows] - nb["z"][rows, kk, None, None, None]) / nb["sz"][rows, kk, None, None, None]
        m_k = (dx * dx + dy * dy + dz * dz) < 1.0
        if kk == 0 and skip_first is not None:
            m_k = m_k & ~skip_first[rows, None, None, None]
        nbm[rows] = nbm[rows] | m_k
    return nbm


def make_kernels(Dz, Dy, Dx, n_regions, G=PROFILE_GRID):
    umax = N_BOX * 0.9

    def geometry_terms(det, axis, s0, rc, frames_phi0, osc, sd, sm, rcw, pix=None):
        """Per-reflection pixel mapping. rc: dict of per-reflection arrays. pix: the per-pixel terms of an earlier
        call on the same reflections with the same sigma_D (pass 1's zeta, J, u1, u2, seff), which depend on neither
        sigma_M nor the rocking curve; only the per-frame terms are recomputed."""
        S = rc["S"]
        if "pxs" in rc:         # s x s super-pixels: box index -> centre of the super-pixel in detector pixels
            s_ = rc["pxs"][:, None]
            x = (rc["ix0"][:, None].astype(jnp.float32) + jnp.arange(Dx, dtype=jnp.float32)[None]) * s_ + (s_ - 1) * 0.5 + PIX0
            y = (rc["iy0"][:, None].astype(jnp.float32) + jnp.arange(Dy, dtype=jnp.float32)[None]) * s_ + (s_ - 1) * 0.5 + PIX0
        else:
            x = rc["ix0"][:, None].astype(jnp.float32) + jnp.arange(Dx, dtype=jnp.float32)[None] + PIX0
            y = rc["iy0"][:, None].astype(jnp.float32) + jnp.arange(Dy, dtype=jnp.float32)[None] + PIX0
        if pix is not None:
            zeta = pix["zeta"]
        else:
            c1 = jnp.cross(S, s0)
            e1 = c1 / jnp.linalg.norm(c1, axis=-1, keepdims=True)
            c2 = jnp.cross(S, e1)
            e2 = c2 / jnp.linalg.norm(c2, axis=-1, keepdims=True)
            zeta = e1 @ axis
            eps1, eps2 = _eps12(det, S, e1, e2, x, y)
        # pixel area in eps space: derivatives of the (smooth) eps maps by second-order differences of the pixel-
        # centre values (one-sided at the box edges); boxes under 3 pixels: central differences at the pixel edges
        # (four more per-pixel geometry evaluations, the largest cost of the kernels)
        if pix is not None:
            J = pix["J"]
        elif Dx >= 3 and Dy >= 3:
            g1x, g1y = torch.gradient(eps1, dim=(2, 1), edge_order=2)
            g2x, g2y = torch.gradient(eps2, dim=(2, 1), edge_order=2)
            J = jnp.abs(g1x * g2y - g2x * g1y)                                    # rad^2 per pixel
        else:
            a1, a2 = _eps12(det, S, e1, e2, x + 0.5, y)
            b1, b2 = _eps12(det, S, e1, e2, x - 0.5, y)
            c1_, c2_ = _eps12(det, S, e1, e2, x, y + 0.5)
            d1, d2 = _eps12(det, S, e1, e2, x, y - 0.5)
            J = jnp.abs((a1 - b1) * (c2_ - d2) - (a2 - b2) * (c1_ - d1))     # rad^2 per pixel
        # frames
        fb = rc["fb"][:, None]                                                   # frames per phi bin
        k = rc["iz0"][:, None].astype(jnp.float32) + fb * jnp.arange(Dz, dtype=jnp.float32)[None]   # first frame (0-based)
        phi_lo = frames_phi0 + osc * k
        phi_hi = phi_lo + osc * fb
        lo, hi = jnp.minimum(phi_lo, phi_hi), jnp.maximum(phi_lo, phi_hi)
        R = _phi_rc(lo, hi, jnp.broadcast_to(rc["phi"][:, None], lo.shape),
                    jnp.broadcast_to(zeta[:, None], lo.shape), sm, rcw)                 # (N, Dz)
        # observed spot = true profile convolved with the pixel footprint: work in units of
        # sigma_eff = sqrt(sigma_D^2 + pixel^2/12) so masks and the Gaussian prior describe the
        # pixel-integrated profile (essential when spots are undersampled, sigma_D ~ pixel)
        if pix is not None:
            seff, u1c, u2c = pix["seff"], pix["u1"], pix["u2"]
        else:
            psz = jnp.sqrt(J[:, Dy // 2, Dx // 2])                                # pixel size (rad)
            seff = sd * jnp.sqrt(1.0 + (psz / sd) ** 2 / 12.0)                   # (N,)
            u1c = eps1 / seff[:, None, None]
            u2c = eps2 / seff[:, None, None]
        phi_c = 0.5 * (lo + hi)
        # spot coordinates (N,1,Dy,Dx), broadcast over frames
        u1, u2 = u1c[:, None], u2c[:, None]
        return dict(x=x, y=y, u1=u1, u2=u2, u1c=u1c, u2c=u2c, J=J, R=R, zeta=zeta, phi_c=phi_c, seff=seff)

    def profile_values(prof, region, gt, sd):
        """p2 = P2(u) * J (fraction of the 2D spot per pixel, per frame (N,Dz,Dy,Dx): the spot centre drifts);
        prof is density in u-units."""
        inside, i1, i2, t1, t2 = _bilinear(prof, gt["u1"], gt["u2"], umax)
        r = region[:, None, None, None]
        v = ((1 - t1) * (1 - t2) * prof[r, i2, i1] + t1 * (1 - t2) * prof[r, i2, i1 + 1]
             + (1 - t1) * t2 * prof[r, i2 + 1, i1] + t1 * t2 * prof[r, i2 + 1, i1 + 1])
        return jnp.where(inside, v, 0.0) * (gt["J"] / (gt["seff"] ** 2)[:, None, None])[:, None]

    @torch.no_grad()
    def pass1(sb, rc, det, axis, s0, frames_phi0, osc, sd, sm, gain, prof, nb, rcw):
        """Background, summation, centroids, 2D projected spot images for profile learning."""
        gt = geometry_terms(det, axis, s0, rc, frames_phi0, osc, sd, sm, rcw)
        valid = sb >= 0                                                         # (N,Dz,Dy,Dx)
        recorded = sb > -2.5                                                   # frames inside sweep
        over = sb == -2
        r2 = gt["u1"] ** 2 + gt["u2"] ** 2                                     # (N,Dz,Dy,Dx)
        fg2 = r2 <= FG_R ** 2
        # frames in foreground: rocking-curve fraction significant
        Rmax = jnp.max(gt["R"], axis=1, keepdims=True)
        fgz = gt["R"] >= 1e-3 * Rmax + 1e-6
        fg = fg2 & fgz[:, :, None, None]
        # neighbour foreground exclusion (ellipsoids in detector/frame coordinates)
        xx = gt["x"][:, None, None, :]
        yy = gt["y"][:, None, :, None]
        zz = (rc["iz0"][:, None].astype(jnp.float32) + rc["fb"][:, None] * (jnp.arange(Dz, dtype=jnp.float32)[None] + 0.5))[:, :, None, None]
        nbm = _neighbour_mask(xx, yy, zz, nb, sb.shape)
        bgm = valid & ~fg & ~nbm & (r2 >= BG_R ** 2)
        c = jnp.where(valid, sb, 0.0)
        # background = k * static template (rings, polarisation, ...); robust (Poisson-aware upper clipping)
        # ratio estimate of k. With a flat template this is the plain clipped mean.
        bref = _bkg_ref(det, xx, yy)                                           # (N,1,Dy,Dx)
        w = bgm.astype(jnp.float32)
        b = jnp.sum(w * c, axis=(1, 2, 3)) / jnp.maximum(jnp.sum(w * bref, axis=(1, 2, 3)), 1e-6)
        for _ in range(4):
            bp = b[:, None, None, None] * bref
            thr = bp + 3.0 * jnp.sqrt(gain * jnp.maximum(bp, 0.5)) + 1.0
            w = (bgm & (c <= thr)).astype(jnp.float32)
            b = jnp.sum(w * c, axis=(1, 2, 3)) / jnp.maximum(jnp.sum(w * bref, axis=(1, 2, 3)), 1e-6)
        nbg = jnp.sum(w * bref, axis=(1, 2, 3))            # template-weighted background "pixel count"
        fgv = fg & valid & ~nbm
        fgw = fgv.astype(jnp.float32)
        bpix = b[:, None, None, None] * bref
        net = c - bpix
        nfg = jnp.sum(fgw, axis=(1, 2, 3))
        sfb = jnp.sum(fgw * bref, axis=(1, 2, 3))
        isum = jnp.sum(fgw * net, axis=(1, 2, 3))
        vsum = gain * jnp.sum(fgw * c, axis=(1, 2, 3)) + sfb ** 2 * gain * jnp.maximum(b, VAR_FLOOR) / jnp.maximum(nbg, 1.0)
        # variance for zero intensity, from the background MODEL (not the observed counts): the observed-count
        # variance correlates with each reflection's own fluctuation (I/sigma of weak data biased negative, and
        # 1/sigma^2-weighted merging biased low); SCALE adds the Poisson term of the expected intensity
        vsum0 = gain * jnp.sum(fgw * bpix, axis=(1, 2, 3)) + sfb ** 2 * gain * jnp.maximum(b, VAR_FLOOR) / jnp.maximum(nbg, 1.0)
        # observed centroid
        wpos = jnp.maximum(net, 0.0) * fgw
        sw = jnp.maximum(jnp.sum(wpos, axis=(1, 2, 3)), 1e-6)
        xo = jnp.sum(wpos * xx, axis=(1, 2, 3)) / sw
        yo = jnp.sum(wpos * yy, axis=(1, 2, 3)) / sw
        zo = jnp.sum(wpos * zz, axis=(1, 2, 3)) / sw
        # second moments in eps space (for sigma_D / sigma_M estimation); signed weights so
        # background noise averages out instead of inflating the moments
        eps1 = gt["u1"] * gt["seff"][:, None, None, None]
        eps2 = gt["u2"] * gt["seff"][:, None, None, None]
        wsg = net * fgw
        sws = jnp.maximum(jnp.sum(wsg, axis=(1, 2, 3)), 1e-6)
        # central moments (about the observed centroid): prediction errors (orientation drift before the
        # per-block refinement) would otherwise count as spot width and feed back through the box size
        mu1 = jnp.sum(wsg * eps1, axis=(1, 2, 3)) / sws
        mu2 = jnp.sum(wsg * eps2, axis=(1, 2, 3)) / sws
        m_d = (jnp.sum(wsg * (eps1 ** 2 + eps2 ** 2), axis=(1, 2, 3)) / sws - mu1 ** 2 - mu2 ** 2) / 2.0
        e3 = gt["zeta"][:, None] * (gt["phi_c"] - rc["phi"][:, None])
        mu3 = jnp.sum(wsg * e3[:, :, None, None], axis=(1, 2, 3)) / sws
        m_m = jnp.sum(wsg * (e3 ** 2)[:, :, None, None], axis=(1, 2, 3)) / sws - mu3 ** 2
        # projected 2D spot (sum over frames) for profile learning, in u space density
        # neighbour pixels must not enter the reference profile (crowded data would broaden it)
        clean = valid & ~nbm
        use_net = jnp.where(fgz[:, :, None, None] & clean, net, 0.0)
        okv = (clean | ~fgz[:, :, None, None]).astype(jnp.float32)
        img2 = jnp.sum(use_net, axis=1)                                          # (N,Dy,Dx)
        all_valid2 = jnp.all(okv > 0.999, axis=1)
        frac_rec = jnp.sum(jnp.where(jnp.any(recorded, axis=(2, 3)), gt["R"], 0.0), axis=1)
        n_over = jnp.sum(over & fg, axis=(1, 2, 3))
        n_fg_masked = jnp.sum(fg & ~valid & recorded, axis=(1, 2, 3))
        n_fg_nb = jnp.sum(fg & nbm, axis=(1, 2, 3))          # foreground pixels claimed by neighbours
        fzp = jnp.sum(jnp.where((fg2 & valid & ~nbm), net, 0.0), axis=(2, 3))   # (N, Dz)
        return dict(bg=b, nbg=nbg, isum=isum, vsum=vsum, vsum0=vsum0, xo=xo, yo=yo, zo=zo, m_d=m_d, m_m=m_m, fz=fzp,
                    img2=img2, all_valid2=all_valid2, frac_rec=frac_rec, n_over=n_over,
                    n_fg_masked=n_fg_masked, nfg=nfg, zeta=gt["zeta"], J=gt["J"], u1=gt["u1c"], u2=gt["u2c"],
                    seff=gt["seff"], n_fg_nb=n_fg_nb)

    @torch.no_grad()
    def accumulate_profiles(img2, ok2, J, u1, u2, region, isum, sel, seff):
        """Scatter normalised projected spots into per-region 2D grids (u units)."""
        dens = img2 / jnp.maximum(isum, 1e-6)[:, None, None] / jnp.maximum(J / (seff ** 2)[:, None, None], 1e-12)
        w = (sel[:, None, None] & ok2).astype(jnp.float32)
        inside, i1, i2, t1, t2 = _bilinear(jnp.zeros((1, G, G)), u1, u2, umax)
        w = w * inside
        r = jnp.broadcast_to(region[:, None, None], u1.shape)
        num = jnp.zeros((n_regions, G, G))
        den = jnp.zeros((n_regions, G, G))
        for di, dj, ww in ((0, 0, (1 - t1) * (1 - t2)), (1, 0, t1 * (1 - t2)), (0, 1, (1 - t1) * t2), (1, 1, t1 * t2)):
            num = jnp.index_add(num, (r, i2 + dj, i1 + di), w * ww * dens)
            den = jnp.index_add(den, (r, i2 + dj, i1 + di), w * ww)
        return num, den

    @torch.no_grad()
    def pass2(sb, rc, det, axis, s0, frames_phi0, osc, sd, sm, gain, prof, bg, nbm_x, nb, rcw, pix=None):
        """Profile fitting with fixed background. Reflections with an overlapping neighbour (same or
        other lattice) are deconvoluted: both intensities are fitted jointly in the shared pixels, each
        with its own predicted profile (2x2 Poisson-weighted least squares), instead of masking.
        pix: pass 1's per-pixel geometry of these reflections (same sigma_D), not recomputed."""
        gt = geometry_terms(det, axis, s0, rc, frames_phi0, osc, sd, sm, rcw, pix)
        valid = sb >= 0
        p = profile_values(prof, rc["region"], gt, sd) * gt["R"][:, :, None, None]   # (N,Dz,Dy,Dx)
        # partner (overlapping neighbour) profile on this reflection's pixel grid
        has_p = rc["has_p"]
        rc_p = dict(S=rc["pS"], phi=rc["pphi"], ix0=rc["ix0"], iy0=rc["iy0"], iz0=rc["iz0"], region=rc["region"], fb=rc["fb"])
        if "pxs" in rc:
            rc_p["pxs"] = rc["pxs"]
        anyp = bool(jnp.any(has_p > 0.5))
        # partner geometry only when this chunk has partners (most chunks have none)
        gtp = geometry_terms(det, axis, s0, rc_p, frames_phi0, osc, sd, sm, rcw) if anyp else None
        pp = ((profile_values(prof, rc["region"], gtp, sd) * gtp["R"][:, :, None, None])
              * has_p[:, None, None, None]) if anyp else jnp.zeros_like(p)
        xx = gt["x"][:, None, None, :]
        yy = gt["y"][:, None, :, None]
        zz = (rc["iz0"][:, None].astype(jnp.float32) + rc["fb"][:, None] * (jnp.arange(Dz, dtype=jnp.float32)[None] + 0.5))[:, :, None, None]
        nbm = _neighbour_mask(xx, yy, zz, nb, sb.shape, skip_first=has_p > 0.5)   # the partner is modelled, not masked
        def _fit_tail(p, pp, two, wuse, wp, wpp_, I, Ip, a11, a12, a22, det2, iv):
            """Variances, profile correlation and recorded fractions after the profile fit."""
            C = jnp.sum(wp * bref * iv, axis=(1, 2, 3))
            var_b = gain * jnp.maximum(bg, VAR_FLOOR) / jnp.maximum(nbm_x, 1.0)
            a11i = a22 / det2                                    # (A^-1)_11 includes the overlap covariance
            varI = a11i + (C * a11i) ** 2 * var_b
            # variance for zero intensity (background model only), scaled by the overlap inflation of this fit
            v0 = gain * jnp.maximum(b, VAR_FLOOR)
            iv0 = 1.0 / v0
            a110 = jnp.maximum(jnp.sum(wpp_ * iv0, axis=(1, 2, 3)), 1e-12)
            C0 = jnp.sum(wp * bref * iv0, axis=(1, 2, 3))
            infl = a11i * a11                    # (A^-1)_11 / (1/a11): >= 1 with an overlapping partner
            var0 = (1.0 / a110 + (C0 / a110) ** 2 * var_b) * jnp.maximum(infl, 1.0)
            pm = jnp.sum(wp, axis=(1, 2, 3)) / jnp.maximum(jnp.sum(wuse, axis=(1, 2, 3)), 1)
            resid = (net - jnp.maximum(Ip, 0.0)[:, None, None, None] * pp) if two else net
            nm = jnp.sum(wuse * resid, axis=(1, 2, 3)) / jnp.maximum(jnp.sum(wuse, axis=(1, 2, 3)), 1)
            dp = (p - pm[:, None, None, None]) * wuse
            dn = (resid - nm[:, None, None, None]) * wuse
            corr = jnp.sum(dp * dn, axis=(1, 2, 3)) / jnp.sqrt(jnp.maximum(jnp.sum(dp * dp, axis=(1, 2, 3)) * jnp.sum(dn * dn, axis=(1, 2, 3)), 1e-20))
            p_valid = jnp.sum(jnp.where(valid & ~nbm, p, 0.0), axis=(1, 2, 3))
            p_total = jnp.sum(p, axis=(1, 2, 3))
            # conditioning of the joint fit with an overlapping partner: profiles that nearly coincide cannot
            # be separated (the 2x2 solve then amplifies noise); such reflections are flagged
            rho = jnp.where(has_p > 0.5, a12 / jnp.sqrt(jnp.maximum(a11 * a22, 1e-30)), 0.0)
            return dict(iprf=I, vprf=varI, vprf0=var0, corr=corr, p_valid=p_valid, p_total=p_total, ipartner=Ip, rho=rho)

        def _fit(p, pp, two):
            pmax = jnp.max(p, axis=(1, 2, 3), keepdims=True)
            ppmax = jnp.max(pp, axis=(1, 2, 3), keepdims=True)
            use = valid & ~nbm & ((p > 0.01 * pmax) | (pp > 0.01 * jnp.maximum(ppmax, 1e-12)))
            wuse = use.astype(jnp.float32)
            wp = wuse * p
            I = jnp.sum(wuse * net, axis=(1, 2, 3)) / jnp.maximum(jnp.sum(wp + wuse * pp, axis=(1, 2, 3)), 1e-6)
            Ip = I * has_p
            # loop-invariant products hoisted, one reciprocal of the variance per iteration; without partners
            # (pp = 0: a12 = b2 = 0, a22 = 1) the 2x2 solve is I = b1 / a11 and the partner terms are skipped
            wpp_, wpn = wp * p, wp * net
            if two:
                wq = wuse * pp
                wqq, wqn, wpq = wq * pp, wq * net, wp * pp
            for it_ in range(N_IRLS):
                I_old = I
                mu = b + jnp.maximum(I, 0.0)[:, None, None, None] * p
                if two:
                    mu = mu + jnp.maximum(Ip, 0.0)[:, None, None, None] * pp
                iv = 1.0 / (gain * jnp.maximum(mu, VAR_FLOOR))
                a11 = jnp.sum(wpp_ * iv, axis=(1, 2, 3))
                b1 = jnp.sum(wpn * iv, axis=(1, 2, 3))
                if two:
                    a12 = jnp.sum(wpq * iv, axis=(1, 2, 3))
                    a22 = jnp.sum(wqq * iv, axis=(1, 2, 3)) + (1.0 - has_p) * 1.0
                    b2 = jnp.sum(wqn * iv, axis=(1, 2, 3))
                else:
                    a12 = jnp.zeros_like(a11)
                    a22 = jnp.ones_like(a11)
                    b2 = jnp.zeros_like(a11)
                det2 = jnp.maximum(a11 * a22 - a12 * a12, 1e-30)
                I = (a22 * b1 - a12 * b2) / det2
                Ip = (a11 * b2 - a12 * b1) / det2 * has_p
                if IRLS_TOL > 0 and it_ >= 1 and it_ < N_IRLS - 1:
                    dz = jnp.abs(I - I_old) * jnp.sqrt(jnp.maximum(det2 / jnp.maximum(a22, 1e-30), 0.0))
                    if float(jnp.max(dz)) < IRLS_TOL:
                        break
            return _fit_tail(p, pp, two, wuse, wp, wpp_, I, Ip, a11, a12, a22, det2, iv)
        c = jnp.where(valid, sb, 0.0)
        bref = _bkg_ref(det, xx, yy)
        b = bg[:, None, None, None] * bref
        net = c - b
        res = _fit(p, pp, anyp)
        # fraction of the reflection inside the summation foreground (pass1: N_FG ellipse x frames with a significant
        # rocking-curve fraction, valid, not claimed by neighbours): summation is divided by it, so broad / split
        # spots whose learnt profile extends beyond the foreground are not biased low
        r2 = gt["u1"] ** 2 + gt["u2"] ** 2
        Rmax = jnp.max(gt["R"], axis=1, keepdims=True)
        fgz = gt["R"] >= 1e-3 * Rmax + 1e-6
        fgm = (r2 <= FG_R ** 2) & fgz[:, :, None, None]
        # shape term only (validity / masking is in PEAK): profile mass inside the foreground / mass in the box
        res["p_fg"] = jnp.sum(jnp.where(fgm, p, 0.0), axis=(1, 2, 3)) / jnp.maximum(jnp.sum(p, axis=(1, 2, 3)), 1e-9)
        return res

    return pass1, accumulate_profiles, pass2


_KERNEL_CACHE: dict = {}


def gaussian_profile(G=PROFILE_GRID):
    umax = N_BOX * 0.9
    u = np.linspace(-umax, umax, G)
    U1, U2 = np.meshgrid(u, u)
    return (np.exp(-0.5 * (U1 ** 2 + U2 ** 2)) / (2 * np.pi)).astype(np.float32)

"""3D profile-fitting integration, batched on GPU with PyTorch.

Model (Kabsch 1988/2010): each pixel of a reflection's shoebox is mapped to the local
reflection frame

    eps1 = e1.(S'-S)/|S|,   eps2 = e2.(S'-S)/|S|,   eps3 = zeta (phi' - phi)

with e1 = S x S0/|S x S0|, e2 = S x e1/|S x e1|, zeta = m2.e1. In this frame the spot is
(nearly) invariant over the detector: width sigma_D in eps1/eps2 and sigma_M along eps3.

The expected fraction of the reflection recorded in pixel (i, j) of frame k is

    p_ijk = P2(eps1, eps2) * J_ij * R_k

where P2 is an empirical 2D reference profile (per detector region and phi block, learnt
from strong reflections and regularised towards a Gaussian), J_ij the pixel area in eps
units, and R_k the exact frame-integrated Gaussian rocking curve. This separable form deals
exactly with fine and coarse phi slicing and with partially recorded reflections.

Intensities are obtained by background subtraction + summation and by weighted profile
fitting (Poisson variances, iterated), with neighbour-overlap masking and robust background.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import tnp as jnp
from .backend import image_device, np_
from .geometry import PIX0, Geometry, predict_reflections
from .integrate_kernels import (FG_R, N_BOX, N_FG, PROFILE_GRID, RC_C, _KERNEL_CACHE, H, _detector_consts,
                                gaussian_profile, make_kernels, rc_basis_np, rc_default_weights)
from .shoebox import (GATHER_CHUNK, KERNEL_RESERVE, SB_OFF, _decode_bins, _gather_process_ok, _HostChunk,
                      _integrate_budget, _job_memory, _map_fd, _plan_passes, _recv_fds, _sb_arg, _send_fds,
                      _shared_zeros, _Stager)


N_BOX_Z = 4.0    # shoebox half-size along phi in sigma units
DZ_BUCKETS = (3, 5, 9, 17, 33, 65)
BIN_SLOTS = 33   # very wide reflections: phi bins of several frames, about this many bins per shoebox
BIN_MAX = 8      # largest number of frames per bin (memory: blocks stay open over the whole box)
XY_BUCKETS = (9, 13, 17, 23, 31, 41, 55, 73, 97)   # shoebox sides (px): per-reflection footprint
# oversampled spots (spot sigma of many pixels: 75 um pixels at long distance): with SUPERPIXEL = t > 0 a reflection
# whose footprint sigma spans sigma_px >= 2 t pixels is integrated on s x s super-pixels, s = floor(sigma_px / t) <=
# SUPERPIXEL_MAX (count sums, so still Poisson; masked / overloaded if any of their pixels is): s^2 fewer voxels to
# gather and fit. Default t = 2.5.
SUPERPIXEL = 2.5
SUPERPIXEL_MAX = 4
QUEUE_MAX_BLOCKS = 6   # blocks the gather process may run ahead, memory permitting
CHUNK_ELEMENTS = int(float(__import__("os").environ.get("XPG_CHUNK_ELEMENTS", 1.5e7)))   # voxels per kernel call
_CHUNK_SET = "XPG_CHUNK_ELEMENTS" in __import__("os").environ


@dataclass
class IntegrationModel:
    sigma_d: float      # beam divergence e.s.d. (degrees)
    sigma_m: float      # reflecting range (mosaicity) e.s.d. (degrees)
    gain: float = 1.0
    box_xy: int = 0     # shoebox side (pixels), odd

    def box(self, geom: Geometry, dmin=None):
        if self.box_xy:
            return self.box_xy
        # largest |X| on detector -> largest pixel footprint of sigma_d
        corners = geom.pixel_to_lab(np.array([0.5, geom.nx + .5, 0.5, geom.nx + .5]),
                                    np.array([0.5, 0.5, geom.ny + .5, geom.ny + .5]))
        L = np.linalg.norm(corners, axis=1).max()
        # oblique incidence stretches spots by up to 1/cos(2theta)
        cos2t = geom.distance / L
        spx = L * np.radians(self.sigma_d) / min(geom.qx, geom.qy) / max(cos2t, 0.4)
        spx = math.sqrt(spx ** 2 + 1.0 / 12.0)          # pixel footprint
        h = int(math.ceil(N_BOX * spx)) + 1
        return 2 * max(h, 2) + 1


# ---------------------------------------------------------------------------
# Reflection prediction for integration with block-wise refined models
# ---------------------------------------------------------------------------

def lattice_allowed(hkl, space_group):
    """True for reflections allowed by the lattice centring (h.t integral for all centring vectors)."""
    import gemmi
    try:
        cen = gemmi.SpaceGroup(space_group).operations().cen_ops
    except Exception:
        return np.ones(len(hkl), bool)
    ok = np.ones(len(hkl), bool)
    for t in cen:
        tv = np.array(t, float) / gemmi.Op.DEN
        ph = hkl @ tv
        ok &= np.abs(ph - np.round(ph)) < 1e-6
    return ok


def predict_for_integration(exps_by_block, block_edges, dmin, dmax, z_lo, z_hi, minimum_zeta=0.05, lattice_of=None):
    """Predictions for each phi block; ``block_edges[b], block_edges[b+1]`` bound block b. With several
    lattices, ``exps_by_block`` is the flat list over lattices and ``lattice_of[b]`` the lattice id."""
    out = []
    nb0 = len(block_edges) - 1
    for b, exp in enumerate(exps_by_block):
        za, zb = block_edges[b % nb0], block_edges[b % nb0 + 1]
        from .geometry import generate_hkl
        hkl = generate_hkl(exp.crystal.Astar, dmin, dmax)
        hkl = hkl[lattice_allowed(hkl, exp.crystal.space_group)]   # centred cells: skip absent lattice rows
        pr = predict_reflections(exp.geom, exp.crystal.Astar, dmin, dmax, z_range=(za, zb), hkl=hkl)
        pr["block"] = np.full(pr["x"].size, b, np.int32)
        pr["lattice"] = np.full(pr["x"].size, 0 if lattice_of is None else lattice_of[b], np.int32)
        out.append(pr)
    res = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    g = exps_by_block[0].geom
    e1 = np.cross(res["S"], g.s0)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    res["zeta_k"] = e1 @ g.axis          # Kabsch zeta = m2.e1 (rocking width); res["zeta"] is Lorentz
    ok = (np.abs(res["zeta_k"]) >= minimum_zeta) & (res["x"] >= 0.5) & (res["x"] < g.nx + 0.5) & \
         (res["y"] >= 0.5) & (res["y"] < g.ny + 0.5) & (res["z"] >= z_lo) & (res["z"] < z_hi)
    return {k: v[ok] for k, v in res.items()}


@dataclass
class IntegrationResult:
    refl: dict
    model: IntegrationModel
    profiles: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)


def _chunk_target(dev):
    """Voxels per kernel call: XPG_CHUNK_ELEMENTS, else on CUDA a quarter of the free memory at ~100 bytes of
    temporaries per voxel, within 1.5e7-1e8, else 1.5e7."""
    if _CHUNK_SET or getattr(dev, "type", "cpu") != "cuda":
        return CHUNK_ELEMENTS
    try:
        free = torch.cuda.mem_get_info(dev)[0]
    except Exception:  # noqa: BLE001
        return CHUNK_ELEMENTS
    return int(np.clip(0.25 * free / 100.0, 1.5e7, 1e8))


def _chunk_size(Dz, box, target=CHUNK_ELEMENTS):
    """Reflections per kernel call so that one (N, Dz, box, box) array has ~target elements."""
    n = max(64, min(8192, target // (Dz * box * box)))
    return int(2 ** np.floor(np.log2(n)))


def _padded_rows(m, chunk):
    """Leading (reflection) dimension of a kernel call. JAX needed a few fixed sizes (each shape was a separate XLA
    compilation); eager PyTorch has no such cost, so no padding: padded duplicate rows were pure extra work."""
    return m


def _bucket(dz):
    for b in DZ_BUCKETS:
        if dz <= b:
            return b
    return None


def _neighbours(x, y, z, sx, sz, k=8):
    pts = np.c_[x / sx, y / sx, z / sz]
    tree = cKDTree(pts)
    d, j = tree.query(pts, k=k + 1, distance_upper_bound=4.0 * N_FG)
    j = j[:, 1:]
    d = d[:, 1:]
    has = np.isfinite(d)
    j = np.where(has, j, 0)
    return j, has, np.where(has, d, np.inf)


def _kernel_device():
    """Device of the (float32) integration kernels: the compute device (CUDA or CPU)."""
    want = __import__("os").environ.get("XPG_INTEGRATE_DEVICE", "").strip()
    if want:
        return torch.device(want)
    dev = image_device()
    return torch.device("cpu") if dev.type == "mps" else dev


def _sb_device_budget(dev):
    """Shoebox bytes of one block kept on the kernel device (XPG_SB_DEVICE_GB; default min(8 GB, 1/4 of the free
    CUDA memory); other devices: none)."""
    gb = __import__("os").environ.get("XPG_SB_DEVICE_GB")
    if gb is not None:
        return float(gb) * 2 ** 30
    try:
        free = torch.cuda.mem_get_info(dev)[0] if dev.type == "cuda" else 0
    except Exception:  # noqa: BLE001
        free = 0
    return min(8 * 2 ** 30, 0.25 * free)


def integrate(*args, **kw):
    """3D profile-fitting integration of the predicted reflections (see _integrate)."""
    with jnp.on(_kernel_device()):
        return _integrate(*args, **kw)


def estimate_sigmas(*args, **kw):
    """sigma_D / sigma_M from strong spots (see _estimate_sigmas)."""
    with jnp.on(_kernel_device()):
        return _estimate_sigmas(*args, **kw)


def _integrate(images, exp_blocks, block_edges, model: IntegrationModel, dmin, dmax=1e9, frames=None,
              mask=None, minimum_zeta=0.05, verbose=True, log=print, n_regions_xy=3, learn_profiles=True,
              exclude_frames=(), adaptive=True, deconvolve_overlaps=True, extra_lattices=None, bkg_template=None,
              foreign_spots=None, n_foreign=4, moments_only=False):
    """Integrate a sweep. ``exp_blocks``: one Experiment per phi block, block_edges in frame z units.
    ``extra_lattices``: further lattices (lists of per-block Experiments on the same edges), integrated
    simultaneously so that overlaps between lattices are deconvoluted or masked; refl['lattice'] = id."""
    lattice_of = [0] * len(exp_blocks)
    if extra_lattices:
        exp_blocks = list(exp_blocks)
        for li, lat in enumerate(extra_lattices, start=1):
            exp_blocks += list(lat)
            lattice_of += [li] * len(lat)
    t0 = time.time()
    g0 = exp_blocks[0].geom
    if frames is None:
        frames = (images.first, images.last)
    f_first, f_last = frames
    z_lo, z_hi = f_first - 1.0, float(f_last)
    refl = predict_for_integration(exp_blocks, block_edges, dmin, dmax, z_lo, z_hi, minimum_zeta, lattice_of)
    n = refl["x"].size
    box = (IntegrationModel(model.sigma_d * (1.3 if adaptive else 1.0), model.sigma_m).box(g0)
           if not model.box_xy else model.box_xy)
    half = box // 2
    sd = np.radians(model.sigma_d)
    sm = np.radians(model.sigma_m)
    sz_frames = model.sigma_m / np.abs(refl["zeta_k"]) / abs(g0.osc)
    hz = np.ceil(N_BOX_Z * sz_frames + 0.5).astype(int)
    # very wide reflections (high mosaicity, near the rotation axis): frames are summed into phi bins of fb
    # frames (about BIN_SLOTS bins per box) so the box covers the whole rocking curve; beyond BIN_MAX frames per
    # bin the box is truncated instead of rejecting the reflection, and the recorded rocking-curve fraction
    # (coverage) enters PEAK, so MINPK still removes reflections that are mostly outside the box
    hmax = (max(DZ_BUCKETS) - 1) // 2
    wide = hz > hmax
    fb = np.where(wide, np.clip(np.ceil((2 * hz + 1) / BIN_SLOTS), 2, BIN_MAX), 1).astype(int)
    hz = np.where(wide, np.minimum(np.ceil(N_BOX_Z * sz_frames / fb + 0.5).astype(int), (BIN_SLOTS - 1) // 2),
                  hz).astype(int)
    trunc = wide & (np.ceil(N_BOX_Z * sz_frames / fb + 0.5) > (BIN_SLOTS - 1) // 2)
    iz_c = np.floor(refl["z"] - (fb - 1) / 2.0).astype(int)     # bin hz holds the predicted centre
    iz0 = iz_c - hz * fb
    dz = 2 * hz + 1
    bucket = np.array([_bucket(d) or -1 for d in dz])
    from scipy.special import erf as _erf
    zc = refl["z"]
    cov = 0.5 * (_erf((iz0 + dz * fb - zc) / (np.sqrt(2) * sz_frames)) - _erf((iz0 - zc) / (np.sqrt(2) * sz_frames)))
    refl["coverage"] = np.where(trunc, np.clip(cov, 0, 1), 1.0).astype(np.float32)
    keep = bucket > 0
    if verbose:
        log(f"  INTEGRATE: {n} predicted reflections, box {box}x{box} px, sigma_D {model.sigma_d:.4f} "
            f"sigma_M {model.sigma_m:.4f} deg; {int(wide.sum())} wide in phi (> {max(DZ_BUCKETS)} frames): integrated "
            f"in bins of 2-{BIN_MAX} frames, {int(trunc.sum())} of them still truncated")
    refl = {k: v[keep] for k, v in refl.items()}
    iz0, dz, bucket, fb = iz0[keep], dz[keep], bucket[keep], fb[keep]
    n = refl["x"].size
    # per-reflection shoebox side from the local footprint of sigma_D (distance, obliquity), bucketed: one
    # global box sized for the most oblique detector corner wastes most of the compute near the beam
    if model.box_xy:
        bxs = np.full(n, box, int)
    else:
        Xv = g0.pixel_to_lab(refl["x"], refl["y"])
        Lr = np.linalg.norm(Xv, axis=1)
        cosr = np.abs(Xv @ g0.ed3) / Lr
        spx = Lr * np.radians(model.sigma_d * (1.3 if adaptive else 1.0)) / min(g0.qx, g0.qy) / np.maximum(cosr, 0.4)
        spx = np.sqrt(spx ** 2 + 1.0 / 12.0)
        need = 2 * np.maximum(np.ceil(N_BOX * spx).astype(int) + 1, 2) + 1
        xyb = np.array([b_ for b_ in XY_BUCKETS if b_ < box] + [box])
        bxs = xyb[np.minimum(np.searchsorted(xyb, need), len(xyb) - 1)]
    pxs = np.ones(n, int)            # super-pixel side per reflection (SUPERPIXEL)
    if SUPERPIXEL > 0 and not model.box_xy and n:
        pxs = np.clip(np.floor(spx / SUPERPIXEL), 1, SUPERPIXEL_MAX).astype(int)
        need_s = 2 * np.maximum(np.ceil(N_BOX * spx / pxs).astype(int) + 1, 2) + 1
        bxs = np.where(pxs > 1, xyb[np.minimum(np.searchsorted(xyb, need_s), len(xyb) - 1)], bxs)
        if verbose and (pxs > 1).any():
            log(f"  INTEGRATE: {int((pxs > 1).sum())} reflections on super-pixels (spot sigma >= {2 * SUPERPIXEL:g} px): "
                + ", ".join(f"{int((pxs == k).sum())} of {k}x{k}" for k in np.unique(pxs[pxs > 1])))
    halfs = bxs // 2
    # box origin in the reflection's own (super-)pixel units: detector pixel i lies in super-pixel i // s
    ix0 = np.floor(refl["x"] - PIX0 + 0.5).astype(int) // pxs - halfs
    iy0 = np.floor(refl["y"] - PIX0 + 0.5).astype(int) // pxs - halfs
    # neighbour lists for overlap masking
    Xlen = np.linalg.norm(g0.pixel_to_lab(refl["x"], refl["y"]), axis=1)
    sx_px = np.maximum(Xlen * sd / g0.qx, 0.5) * N_FG
    sz_f = np.sqrt((model.sigma_m / np.abs(refl["zeta_k"])) ** 2 + (g0.osc ** 2) / 12.0) / abs(g0.osc) * N_FG
    nbj, nbh, nbd = _neighbours(refl["x"], refl["y"], refl["z"], sx_px.mean(), sz_f.mean())
    # overlapping partner: nearest neighbour whose foreground ellipsoid intersects ours -> joint fit
    has_partner = (nbh[:, 0] & (nbd[:, 0] < 2.0)) if deconvolve_overlaps else np.zeros(len(nbd), bool)
    out_overlap = has_partner.copy()
    nbx = np.where(nbh, refl["x"][nbj], -1e6).astype(np.float32)
    nby = np.where(nbh, refl["y"][nbj], -1e6).astype(np.float32)
    nbz = np.where(nbh, refl["z"][nbj], -1e6).astype(np.float32)
    nbsx = np.where(nbh, sx_px[nbj], 1.0).astype(np.float32)
    nbsz = np.where(nbh, sz_f[nbj], 1.0).astype(np.float32)
    # spots of other crystals (unindexed SPOTS spots far from every predicted reflection of the integrated
    # lattices) are masked like non-modelled neighbours: excluded from background, summation and profile fit
    if foreign_spots is not None and len(foreign_spots[0]) and n:
        # spots of other crystals are masked like non-modelled neighbours (excluded from background, summation
        # and profile fit), decided per reflection: a foreign spot within 2.5 sigma of the reflection's own predicted
        # centre is never masked for it (it may be the reflection itself: centroids scatter by 1-2 sigma), and the
        # mask radius is 2 sigma so it cannot reach the reflection's core
        fx, fy, fz = (np.asarray(a, float) for a in foreign_spots)
        sxm, szm = float(sx_px.mean()), float(sz_f.mean())          # N_FG-sigma units
        ft = cKDTree(np.c_[fx / sxm, fy / sxm, fz / szm])
        d_f, j_f = ft.query(np.c_[refl["x"] / sxm, refl["y"] / sxm, refl["z"] / szm], k=n_foreign + 2,
                            distance_upper_bound=4.0 * N_FG)
        d_f, j_f = np.atleast_2d(d_f), np.atleast_2d(j_f)
        hf = np.isfinite(d_f) & (d_f > 2.5 / N_FG)
        # keep the n_foreign nearest eligible ones
        order_f = np.argsort(np.where(hf, d_f, np.inf), axis=1)[:, :n_foreign]
        rows_f = np.arange(d_f.shape[0])[:, None]
        hf, j_f = hf[rows_f, order_f], np.where(np.isfinite(d_f), j_f, 0)[rows_f, order_f]
        rad = 2.0 / N_FG
        nbx = np.concatenate([nbx, np.where(hf, fx[j_f], -1e6).astype(np.float32)], 1)
        nby = np.concatenate([nby, np.where(hf, fy[j_f], -1e6).astype(np.float32)], 1)
        nbz = np.concatenate([nbz, np.where(hf, fz[j_f], -1e6).astype(np.float32)], 1)
        nbsx = np.concatenate([nbsx, np.full(hf.shape, sxm * rad, np.float32)], 1)
        nbsz = np.concatenate([nbsz, np.full(hf.shape, szm * rad, np.float32)], 1)
        if verbose:
            log(f"  INTEGRATE: {len(fx)} unindexed spots of other crystals; masked (2 sigma) where farther than 2.5 sigma "
                f"from a reflection's own centre: {100 * hf.any(axis=1).mean():.0f}% of the reflections have one")
    # detector regions for reference profiles
    rx = np.clip((refl["x"] / g0.nx * n_regions_xy).astype(int), 0, n_regions_xy - 1)
    ry = np.clip((refl["y"] / g0.ny * n_regions_xy).astype(int), 0, n_regions_xy - 1)
    region = (ry * n_regions_xy + rx).astype(np.int32)
    nreg = n_regions_xy * n_regions_xy
    nblocks = len(exp_blocks)
    # frame ranges needed per block
    blk = refl["block"]
    zstart = np.maximum(iz0, f_first - 1)
    zend = np.minimum(iz0 + dz * fb - 1, f_last - 1)
    order = np.argsort(blk, kind="stable")
    # outputs
    out = {k: np.full(n, np.nan, np.float64) for k in
           ("bg", "isum", "vsum", "vsum0", "iprf", "vprf", "vprf0", "corr", "xo", "yo", "zo", "p_valid", "p_total",
            "frac_rec", "rho", "m_d", "m_m", "p_fg")}
    out["n_over"] = np.zeros(n, np.int32)
    out["n_fg_masked"] = np.zeros(n, np.int32)
    keep_fz = not learn_profiles
    fz_out = np.zeros((n, max(DZ_BUCKETS)), np.float32) if keep_fz else None
    def get_kernels(Dz, bx):
        key = (Dz, bx, nreg)
        if key not in _KERNEL_CACHE:   # compiled once per shape for the whole session
            _KERNEL_CACHE[key] = make_kernels(Dz, bx, bx, nreg)
        return _KERNEL_CACHE[key]

    gauss = gaussian_profile()
    prof_by_block = {}
    blocks_refl = [order[blk[order] == b] for b in range(nblocks)]
    need_lo = [int(zstart[r].min()) if r.size else 10 ** 9 for r in blocks_refl]
    need_hi = [int(zend[r].max()) if r.size else -1 for r in blocks_refl]
    open_bufs = {}
    sched = {}
    shared_fd = {}       # gather process: (block, B, bx, 0 / 1) -> descriptor of the shoebox / count buffer
    gproc_child = [False]

    def _zeros(shape, dtype, key):
        if not gproc_child[0]:
            return np.zeros(shape, dtype)
        a, shared_fd[key] = _shared_zeros(shape, dtype)
        return a
    pad = int(halfs[pxs == 1].max()) + 1 if (pxs == 1).any() else half + 1
    pads = {int(k): int(halfs[pxs == k].max()) + 1 for k in np.unique(pxs) if k > 1}    # super-pixel frames
    t_read = t_gpu = 0.0

    def open_block(b):
        rows = blocks_refl[b]
        bufs = {}
        for B in DZ_BUCKETS:
            for bx in np.unique(bxs[rows]):
                for sp in np.unique(pxs[rows]):
                    sel = rows[(bucket[rows] == B) & (bxs[rows] == bx) & (pxs[rows] == sp)]
                    # reflections with an overlapping partner first: the joint-fit branch then runs only in their chunks
                    sel = sel[np.argsort(~has_partner[sel], kind="stable")]
                    if not sel.size:
                        continue
                    gk = (B, int(bx), int(sp))        # depth, box side, super-pixel side
                    # stored as value + SB_OFF so that "unrecorded" (-3) is 0: np.zeros maps pages lazily, so rows of
                    # reflections whose frames have not been read yet cost no memory
                    arr = _zeros((sel.size, B, int(bx), int(bx)), np.float32, (b,) + gk + (0,))
                    # binned reflections: number of frames summed into each bin (complete bins only are used)
                    cnt = _zeros((sel.size, B), np.int16, (b,) + gk + (1,)) if (fb[sel] > 1).any() else None
                    bufs[gk] = (sel, arr, cnt)
                    # rows by first frame: a frame's rows are found by two binary searches, and the box corners in the
                    # padded frame are precomputed
                    o_g = np.argsort(iz0[sel], kind="stable")
                    r_ = sel[o_g]
                    Pk, pk = frames_by_px[int(sp)]
                    sched[(b,) + gk] = (
                        iz0[r_], o_g, np.clip(iy0[r_] + pk, 0, Pk.shape[0] - int(bx)),
                        np.clip(ix0[r_] + pk, 0, Pk.shape[1] - int(bx)), fb[r_], int((B * fb[r_]).max()))
        open_bufs[b] = bufs

    adapt = dict(sd=sd, sm=sm, rcw=rc_default_weights())
    stagers = {}            # pinned upload buffers of the compute thread (process_block), per device
    chunk_targets = {}      # voxels per kernel call, per device (from the free memory at the first block)

    def _adapt_block(b, res1_all, sd_b, sm_b, rcw_b):
        """per-batch update: refit sigma_M and the rocking-curve shape (NNLS on per-frame
        fractions of strong reflections) for this block's profile fit, and sigma_D (second moments)
        for the next block."""
        g = exp_blocks[b].geom
        F, PH, ZK, IZ, MD, XL = [], [], [], [], [], []
        for ((B, _bx), s_, m, idx, sb, rc, nb, r1) in res1_all:
            isum = H(r1)["isum"][:m]
            vs = H(r1)["vsum"][:m]
            st = (isum > 0) & (isum / np.sqrt(np.maximum(vs, 1e-6)) > 15) & (H(r1)["n_over"][:m] == 0) \
                & (H(r1)["frac_rec"][:m] > 0.98) & (fb[s_] == 1)
            if not st.any():
                continue
            fz = np.zeros((st.sum(), max(DZ_BUCKETS)))
            fz[:, :B] = np_(r1["fz"])[:m][st]
            F.append(fz)
            PH.append(refl["phi"][s_][st])
            ZK.append(refl["zeta_k"][s_][st])
            IZ.append(iz0[s_][st])
            MD.append(H(r1)["m_d"][:m][st])
            XL.append(np.c_[refl["x"][s_][st], refl["y"][s_][st]])
        n = sum(len(f) for f in F)
        if n < 100:
            block_params[b] = dict(sigma_d=float(np.degrees(sd_b)), sigma_m=float(np.degrees(sm_b)), n=n,
                                   rcw=np.asarray(rcw_b).round(3).tolist())
            return sm_b, rcw_b
        F, PH, ZK, IZ, MD, XL = map(np.concatenate, (F, PH, ZK, IZ, MD, XL))
        # anchored to the global estimate (sm, sd of this integration run): searched within 0.5-2x of it and
        # shrunk towards it with a prior worth 300 strong reflections, so blocks cannot drift (each block
        # used to start from the previous block's value)
        sm_fit = np.radians(_fit_rocking_width([(F, IZ, PH, ZK)], g, np.degrees(sm), 0.5, 2.0))
        sm_new = float(np.exp((n * np.log(sm_fit) + 300.0 * np.log(sm)) / (n + 300.0)))
        k = IZ[:, None] + np.arange(F.shape[1])[None]
        lo = g.z_to_phi(k.astype(float))
        hi = g.z_to_phi(k + 1.0)
        lo, hi = np.radians(np.minimum(lo, hi)), np.radians(np.maximum(lo, hi))
        Bm = rc_basis_np(lo, hi, np.radians(PH), ZK, sm_new)               # (n, Dz, K)
        tot = np.maximum(F.sum(1, keepdims=True), 1e-9)
        A = (Bm * np.sqrt(tot)[:, :, None]).reshape(-1, len(RC_C))
        y = (F / tot * np.sqrt(tot)).reshape(-1)
        w = _nnls_gram(A, y)
        # regularise towards the Gaussian shape (prior weight equivalent to ~200 reflections)
        lam = 200.0 / n
        w = (w / max(w.sum(), 1e-9) + lam * rc_default_weights()) / (1 + lam)
        # sigma_D from truncation-corrected second moments minus pixel footprint (as estimate_sigmas)
        Xv = g.pixel_to_lab(XL[:, 0], XL[:, 1])
        Xlen = np.linalg.norm(Xv, axis=1)
        cosa = np.abs(Xv @ g.ed3) / Xlen
        t2 = 1.0 - (FG_R ** 2 / 2) * np.exp(-FG_R ** 2 / 2) / (1 - np.exp(-FG_R ** 2 / 2))
        md = MD / t2 - (g.qx / Xlen) ** 2 * (1 + cosa ** 2) / 24.0
        sd_fit = float(np.clip(np.sqrt(max(np.median(md), 1e-12)), 0.5 * sd, 2.0 * sd))
        sd_new = float(np.exp((n * np.log(sd_fit) + 300.0 * np.log(sd)) / (n + 300.0)))
        adapt.update(sd=sd_new, sm=float(sm_new), rcw=w.astype(np.float32))
        block_params[b] = dict(sigma_d=float(np.degrees(sd_b)), sigma_m=float(np.degrees(sm_new)), n=int(n),
                               rcw=w.round(3).tolist())
        return float(sm_new), w.astype(np.float32)
    block_params = {}
    def process_block(b, bufs):
        nonlocal t_gpu
        tg = time.time()
        sd_b = adapt["sd"] if adaptive else sd          # sigma_D used for this block (fixed within block)
        sm_b, rcw_b = (adapt["sm"], adapt["rcw"]) if adaptive else (sm, rc_default_weights())
        exp = exp_blocks[b]
        g = exp.geom
        det = _detector_consts(g, bkg_template)
        axis = jnp.asarray(g.axis, jnp.float32)
        s0 = jnp.asarray(g.s0, jnp.float32)
        frames_phi0 = np.float32(np.radians(g.z_to_phi(0.0)))
        osc = np.float32(np.radians(g.osc))
        prof0 = jnp.asarray(np.repeat(gauss[None], nreg, 0))
        res1_all = []
        # shoebox chunks on the (GPU) kernel device for the whole block when they fit: one transfer per chunk
        # instead of one per pass (pass 1, profile accumulation, pass 2), padding and offset applied there
        kdev = jnp.asarray(np.float32(0)).device
        sb_dev = kdev.type != "cpu" and sum(a.nbytes for (_, a, _) in bufs.values()) <= _sb_device_budget(kdev)
        stager = stagers.setdefault(str(kdev), _Stager(kdev))
        chunk_target = chunk_targets.setdefault(str(kdev), _chunk_target(kdev))
        for (B, bx, sp), (sel, arr, cnt) in bufs.items():
            if cnt is not None:
                _decode_bins(arr, cnt)
                # bins missing frames (sweep ends, excluded frames): unrecorded
                inc = cnt < fb[sel][:, None]
                arr[inc] = 0.0                    # (-3 encoded)
            p1, acc, p2 = get_kernels(B, bx)
            chunk = _chunk_size(B, bx, chunk_target)
            for c0 in range(0, sel.size, chunk):
                s = sel[c0:c0 + chunk]
                m = s.size
                padn = _padded_rows(m, chunk) - m
                idx = np.concatenate([s, np.repeat(s[-1:], padn)])
                if sb_dev:
                    t_ = stager.upload(arr[c0:c0 + m])
                    if padn:
                        t_ = torch.cat([t_, t_[-1:].expand(padn, *t_.shape[1:])])
                    sb = t_.sub_(float(SB_OFF))
                elif kdev.type != "cpu":
                    # block larger than the device budget: moved per pass (pass 1, pass 2) from the block buffer,
                    # padded and offset on the device
                    sb = _HostChunk(arr, c0, m, padn, kdev, stager)
                else:
                    sb = np.concatenate([arr[c0:c0 + m], np.repeat(arr[c0 + m - 1:c0 + m], padn, 0)])
                    sb -= SB_OFF
                pj = nbj[idx, 0]
                rc = dict(S=jnp.asarray(refl["S"][idx], jnp.float32), phi=jnp.asarray(np.radians(refl["phi"][idx]), jnp.float32),
                          ix0=jnp.asarray(ix0[idx]), iy0=jnp.asarray(iy0[idx]), iz0=jnp.asarray(iz0[idx]),
                          fb=jnp.asarray(fb[idx], jnp.float32),
                          region=jnp.asarray(region[idx]), has_p=jnp.asarray(has_partner[idx], jnp.float32),
                          pS=jnp.asarray(refl["S"][pj], jnp.float32), pphi=jnp.asarray(np.radians(refl["phi"][pj]), jnp.float32))
                if sp > 1:
                    rc["pxs"] = jnp.asarray(np.full(idx.size, sp, np.float32))
                nb = dict(x=jnp.asarray(nbx[idx]), y=jnp.asarray(nby[idx]), z=jnp.asarray(nbz[idx]),
                          sx=jnp.asarray(nbsx[idx]), sz=jnp.asarray(nbsz[idx]))
                r1 = p1(_sb_arg(sb), rc, det, axis, s0, frames_phi0, osc, np.float32(sd_b), np.float32(sm_b),
                        np.float32(model.gain), prof0, nb, jnp.asarray(rcw_b))
                res1_all.append(((B, bx), s, m, idx, sb, rc, nb, r1))
        # learn reference profiles from strong, fully recorded, unobstructed reflections
        num = np.zeros((nreg, PROFILE_GRID, PROFILE_GRID))
        den = np.zeros((nreg, PROFILE_GRID, PROFILE_GRID))
        if learn_profiles:
            for (Bk, s, m, idx, sb, rc, nb, r1) in res1_all:
                p1, acc, p2 = get_kernels(*Bk)
                isum = H(r1)["isum"]
                vs = H(r1)["vsum"]
                # reference profiles only from strong, fully recorded, unobstructed spots (no foreground
                # pixel claimed by a neighbour): partly masked spots would distort the learnt shape
                strong = (isum > 0) & (isum / np.sqrt(np.maximum(vs, 1e-6)) > 10.0) & \
                         (H(r1)["n_over"] == 0) & (H(r1)["frac_rec"] > 0.95) & \
                         (H(r1)["n_fg_nb"] == 0) & (H(r1)["n_fg_masked"] == 0)
                strong[m:] = False
                if not strong.any():
                    continue
                # only the strong rows carry weight: scatter just those (the zero-weight rows' atomic adds into the
                # few profile bins dominated the accumulation on crowded data; same sums)
                ks = jnp.asarray(np.flatnonzero(strong))
                nn, dd = acc(r1["img2"][ks], r1["all_valid2"][ks], r1["J"][ks], r1["u1"][ks], r1["u2"][ks],
                             rc["region"][ks], r1["isum"][ks], jnp.asarray(np.ones(int(ks.shape[0]), bool)), r1["seff"][ks])
                num += np_(nn)
                den += np_(dd)
        if adaptive:
            sm_b, rcw_b = _adapt_block(b, res1_all, sd_b, sm_b, rcw_b)
        dens = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        prof = np.empty((nreg, PROFILE_GRID, PROFILE_GRID), np.float32)
        du = (2 * N_BOX * 0.9 / (PROFILE_GRID - 1)) ** 2
        for r in range(nreg):
            k = 3.0  # prior weight (pseudo-counts per grid point)
            emp = np.clip(dens[r], 0, None)
            pr = (den[r] * emp + k * gauss) / (den[r] + k)
            pr = pr / max(pr.sum() * du, 1e-12)
            prof[r] = pr
        prof_by_block[b] = prof
        profj = jnp.asarray(prof)
        for (Bk, s, m, idx, sb, rc, nb, r1) in res1_all:
            B = Bk[0]
            p1, acc, p2 = get_kernels(*Bk)
            r2 = None if moments_only else \
                p2(_sb_arg(sb), rc, det, axis, s0, frames_phi0, osc, np.float32(sd_b), np.float32(sm_b),
                   np.float32(model.gain), profj, r1["bg"], r1["nbg"], nb, jnp.asarray(rcw_b),
                   {k_: r1[k_] for k_ in ("zeta", "J", "u1", "u2", "seff")})
            for k_ in ("bg", "isum", "vsum", "vsum0", "xo", "yo", "zo", "frac_rec", "m_d", "m_m"):
                out[k_][s] = H(r1)[k_][:m]
            out["n_over"][s] = H(r1)["n_over"][:m]
            if fz_out is not None:
                fz_out[s, :B] = np_(r1["fz"])[:m]
            out["n_fg_masked"][s] = H(r1)["n_fg_masked"][:m]
            if r2 is None:              # moments only (spot-width estimation): no profile fit
                continue
            for k_ in ("iprf", "vprf", "vprf0", "corr", "p_valid", "p_total", "rho", "p_fg"):
                out[k_][s] = H(r2)[k_][:m]
        t_gpu += time.time() - tg

    pending = sorted((b for b in range(nblocks) if blocks_refl[b].size), key=lambda b: need_lo[b])
    # EXCLUDE_DATA_RANGE: excluded frames are never read; their shoebox slices stay 'unrecorded'
    # (-3), so profile fitting uses the remaining frames and MINPK rejects heavily affected spots
    excl = set(exclude_frames)
    all_frames = [f for f in range(f_first, f_last + 1) if f not in excl]
    # completed blocks are integrated by one worker thread, in order (the adaptive sigma_D / sigma_M carry over from
    # block to block), while this thread keeps decoding and gathering frames: device compute and host gathering
    # overlap. At most two finished blocks wait (memory).
    from concurrent.futures import ThreadPoolExecutor
    _os = __import__("os")
    gproc = len(all_frames) >= 100 and _gather_process_ok()
    worker = ThreadPoolExecutor(1) if not gproc else None
    block_jobs = []
    job_bytes = {}
    queue_bytes = float(_os.environ.get("XPG_INTEGRATE_QUEUE_GB", 8)) * 2 ** 30
    t_wait = 0.0
    # host memory: shoebox bytes per reflection (depth x side^2 float32, + the bin counts) from its first frame until
    # its block is integrated; blocks split over several passes over the frames when one pass would not fit, and a
    # block waits in the queue only within what is left
    n_fr = f_last - f_first + 1
    w_row = bucket * bxs.astype(np.float64) ** 2 * 4.0 + np.where(fb > 1, 2.0 * bucket, 0.0)
    t_row = np.clip(iz0 - (f_first - 1), 0, n_fr)
    mem_budget = _integrate_budget() if pending else None
    passes, mp_ = _plan_passes(pending, blocks_refl, lattice_of, [h_ - (f_first - 1) for h_ in need_hi], t_row,
                               w_row, n_fr, mem_budget) if pending else ([pending], None)
    if mem_budget is not None and mp_ is not None:
        queue_bytes = min(queue_bytes, max(mem_budget - mp_["open"] - mp_["block"] - KERNEL_RESERVE, 0.0))
        if verbose and len(passes) > 1:
            log(f"  INTEGRATE: shoebox memory in one pass over the frames ~{mp_['one'] / 2 ** 30:.1f} GB > budget "
                f"{mem_budget / 2 ** 30:.1f} GB (XPG_INTEGRATE_MEM_GB): {mp_['plan']}, peak ~{mp_['peak'] / 2 ** 30:.1f} GB")
    pending = list(passes[0])

    def submit_block(b):
        nonlocal t_wait
        bufs = open_bufs.pop(b)
        # shoebox memory of the queue is bounded: at most one block waits behind the one being integrated, and only
        # while the queue holds < XPG_INTEGRATE_QUEUE_GB (default 8)
        nbytes = sum(arr.nbytes + (cnt.nbytes if cnt is not None else 0) for (_, arr, cnt) in bufs.values())
        block_jobs.append(worker.submit(process_block, b, bufs))
        job_bytes[block_jobs[-1]] = nbytes
        tw = time.time()
        while True:
            live = [j for j in block_jobs if not j.done()]
            if len(live) <= 1 or (len(live) <= 2 and sum(job_bytes[j] for j in live) <= queue_bytes):
                break
            live[0].result()
        for j in [j for j in block_jobs if j.done()]:
            j.result()                                   # surface worker errors here
            block_jobs.remove(j)
            job_bytes.pop(j, None)
        t_wait += time.time() - tw

    tr = time.time()
    # padded decode: the decode workers write each frame inside the border with the mask ceiling applied, and the
    # frame loop gathers straight from their ring slot. The pad is rounded up so that successive calls share one pool
    # layout.
    padded_dec = hasattr(images, "padded_ok") and images.padded_ok(len(all_frames))
    if padded_dec:
        pad = max(64, -(-pad // 32) * 32)
    # one padded frame buffer, filled in place (np.pad / np.where made two 4-byte-per-pixel copies per frame)
    P = np.full((images.ny + 2 * pad, images.nx + 2 * pad), -1.0, np.float32)
    inner = P[pad:pad + images.ny, pad:pad + images.nx]
    cur_inner = [inner]                  # the current frame's interior (P's, or the decode ring slot's)
    # super-pixel frames (SUPERPIXEL): s x s sums of the prepared frame; incomplete super-pixels at the far edges
    # stay masked
    frames_by_px = {1: (P, pad)}
    sp_inner = {}
    for sp, pk in pads.items():
        Pk = np.full((images.ny // sp + 2 * pk, images.nx // sp + 2 * pk), -1.0, np.float32)
        frames_by_px[sp] = (Pk, pk)
        sp_inner[sp] = Pk[pk:pk + images.ny // sp, pk:pk + images.nx // sp]
    # the mask as a per-pixel ceiling (-1 masked, +inf elsewhere): copy and mask in one pass, min(value, ceiling)
    # (a masked overload keeps -2), instead of a copy plus a gather / scatter of the masked pixels
    mval = (np.where(np.asarray(mask), np.float32(np.inf), np.float32(-1.0)).astype(np.float32)
            if mask is not None and not np.asarray(mask).all() else None)
    n_gth = int(_os.environ.get("XPG_GATHER_THREADS", min(8, _os.cpu_count() or 1)))
    # the frame copy runs in row bands on the gather threads (a 16M-pixel frame: ~40 ms on one cluster core)
    bands = [slice(int(a[0]), int(a[-1]) + 1) for a in np.array_split(np.arange(images.ny), max(n_gth, 1)) if a.size]

    def _prep_band(args):
        img_, sl = args
        if mval is None:
            np.copyto(inner[sl], img_[sl])
        else:
            np.minimum(img_[sl], mval[sl], out=inner[sl], dtype=np.float32)

    def _sp_band(args):
        """Super-pixel sums of rows sl (super-pixel units) of the prepared frame; a super-pixel with a masked (-1) or
        overloaded (-2) pixel is flagged (overload first)."""
        sp, sl = args
        dst = sp_inner[sp][sl]
        a_ = cur_inner[0][sl.start * sp:sl.stop * sp, :dst.shape[1] * sp]
        # s^2 strided slices added / min-ed in place
        np.copyto(dst, a_[0::sp, 0::sp])
        mn = a_[0::sp, 0::sp].copy()
        for i_ in range(sp):
            for j_ in range(sp):
                if i_ or j_:
                    v_ = a_[i_::sp, j_::sp]
                    np.add(dst, v_, out=dst)
                    np.minimum(mn, v_, out=mn)
        neg = mn < 0
        if neg.any():
            dst[neg] = np.where(mn[neg] <= -1.5, np.float32(-2.0), np.float32(-1.0))

    sp_bands = [(sp, slice(int(a[0]), int(a[-1]) + 1)) for sp, si in sp_inner.items()
                for a in np.array_split(np.arange(si.shape[0]), max(n_gth, 1)) if a.size]

    def _lean_task(group):
        """A list of (V, arr, cnt, rows, slot, iy, ix, B, f) gathers (one per worker and frame): box corners and
        slots precomputed, three NumPy calls per unbinned gather."""
        for V, arr, cnt, rows, slot, iy, ix in group:
            v = V[iy, ix]
            if cnt is None:
                arr[rows, slot] = v + SB_OFF
                continue
            neg = v < 0
            if neg.any():
                v[neg] = np.where(v[neg] == -2.0, np.float32(np.nan), np.float32(np.inf))
            arr[rows, slot] += v
            cnt[rows, slot] += 1

    def frame_loop(submit, say, queued, poll=None):
        """Decode and gather all frames; submit(b) hands each completed block over. Returns the timing stats."""
        t_dec = t_gat = 0.0                          # waiting for decoded frames / copying shoebox pixels
        gpool = ThreadPoolExecutor(n_gth) if n_gth > 1 else None
        tl = time.perf_counter()
        it = images.iter_frames_padded(all_frames, pad, mask) if padded_dec else None
        from_ring = it is not None
        if not from_ring:
            it = images.iter_frames(all_frames, copy=False)
        for f, img in it:
            tg0 = time.perf_counter()
            t_dec += tg0 - tl
            fi = f - 1  # 0-based frame index
            while pending and need_lo[pending[0]] <= fi:
                open_block(pending.pop(0))
            if from_ring:                    # the ring slot is the padded, masked frame
                frames_by_px[1] = (img, pad)
                cur_inner[0] = img[pad:pad + images.ny, pad:pad + images.nx]
                if gpool is not None:
                    for _ in gpool.map(_sp_band, sp_bands):
                        pass
                else:
                    for a_ in sp_bands:
                        _sp_band(a_)
            elif gpool is not None:
                for _ in gpool.map(_prep_band, [(img, sl) for sl in bands]):
                    pass
                for _ in gpool.map(_sp_band, sp_bands):
                    pass
            else:
                _prep_band((img, slice(None)))
                for a_ in sp_bands:
                    _sp_band(a_)
            views = {}
            work, n_rows = [], 0
            for b, bufs in list(open_bufs.items()):
                for (B, bx, sp), (sel, arr, cnt) in bufs.items():
                    zs, o_g, iyc, ixc, f_s, maxlen = sched[(b, B, bx, sp)]
                    hi = int(zs.searchsorted(fi, "right"))
                    lo = int(zs.searchsorted(fi - maxlen + 1, "left"))
                    if hi <= lo:
                        continue
                    kk = fi - zs[lo:hi]
                    rows, iy, ix = o_g[lo:hi], iyc[lo:hi], ixc[lo:hi]
                    if cnt is not None:
                        kk = kk // f_s[lo:hi]
                        ok_ = kk < B
                        if not ok_.all():
                            rows, iy, ix, kk = rows[ok_], iy[ok_], ix[ok_], kk[ok_]
                    V = views.get((bx, sp))
                    if V is None:
                        V = views[(bx, sp)] = np.lib.stride_tricks.sliding_window_view(frames_by_px[sp][0], (bx, bx))
                    for c0 in range(0, rows.size, GATHER_CHUNK):
                        work.append((V, arr, cnt, rows[c0:c0 + GATHER_CHUNK], kk[c0:c0 + GATHER_CHUNK],
                                     iy[c0:c0 + GATHER_CHUNK], ix[c0:c0 + GATHER_CHUNK]))
                        n_rows += min(GATHER_CHUNK, rows.size - c0)
            # one task per worker and frame, balanced by rows (disjoint rows of each buffer)
            if gpool is not None and len(work) > 1:
                parts, cur, acc = [], [], 0
                for w_ in work:
                    cur.append(w_)
                    acc += w_[3].size
                    if acc >= n_rows / n_gth and len(parts) < n_gth - 1:
                        parts.append(cur)
                        cur, acc = [], 0
                parts.append(cur)
                for _ in gpool.map(_lean_task, parts):
                    pass
            else:
                _lean_task(work)
            t_gat += time.perf_counter() - tg0
            for b in list(open_bufs.keys()):
                if need_hi[b] <= fi:
                    for k_ in [k_ for k_ in sched if k_[0] == b]:
                        del sched[k_]
                    submit(b)
            if poll is not None:
                poll()
            tl = time.perf_counter()
            if verbose and (f - f_first + 1) % 500 == 0:
                say(f"  INTEGRATE: frame {f}, {time.time() - t0:.1f}s")
        for b in list(open_bufs.keys()):
            submit(b)
        if gpool is not None:
            gpool.shutdown(wait=True)
        return dict(t_dec=t_dec, t_gat=t_gat)

    def gather_child(mconn, fsock):
        """Forked gather process: the frame loop with shared-memory buffers; each completed block goes to the parent
        as (block, groups) plus the buffers' descriptors, which are closed once the parent reports it done."""
        import gc
        import traceback
        gc.freeze()             # never collect (or touch) the parent's objects here: some hold CUDA memory
        gproc_child[0] = True
        sent = {}                                    # block -> (bytes, descriptors) not yet reported done
        qstat = dict(deep=0)                         # deepest queue allowed by the memory headroom

        def handle(msg):
            if msg[0] != "done":
                raise RuntimeError(f"unexpected message {msg[0]!r}")
            for fd in sent.pop(msg[1])[1]:
                _os.close(fd)

        def poll():
            while mconn.poll():
                handle(mconn.recv())

        def send_block(b):
            meta, fds, nbytes = [], [], 0
            for gk, (sel, arr, cnt) in open_bufs.pop(b).items():   # unmapped here once the frame's work lists go;
                meta.append((gk, sel, arr.shape, None if cnt is None else cnt.shape))   # the descriptors keep the
                fds.append(shared_fd.pop((b,) + gk + (0,)))                            # pages for the parent
                if cnt is not None:
                    fds.append(shared_fd.pop((b,) + gk + (1,)))
                nbytes += arr.nbytes + (cnt.nbytes if cnt is not None else 0)
            mconn.send(("block", b, meta, len(fds)))
            _send_fds(fsock, fds)
            sent[b] = (nbytes, fds)
            # as the worker queue: at most one block waits behind the one being integrated, within the queue limit;
            # deeper (up to QUEUE_MAX_BLOCKS) while the job keeps > 30% of its memory free and the queue holds < 1/4
            # of it: with very wide reflections blocks complete in bursts and the parent waited
            while True:
                n_out, b_out = len(sent), sum(v[0] for v in sent.values())
                if n_out <= 1 or (n_out == 2 and b_out <= queue_bytes):
                    break
                if n_out <= QUEUE_MAX_BLOCKS:
                    lim, used = _job_memory()
                    if lim is not None and lim - used > 0.3 * lim and b_out < 0.25 * lim:
                        qstat["deep"] = max(qstat["deep"], n_out)
                        break
                handle(mconn.recv())

        try:
            stats = frame_loop(send_block, lambda m: mconn.send(("log", m)),
                               lambda: sum(v[0] for v in sent.values()), poll)
            stats["queue_deep"] = qstat["deep"]
            mconn.send(("end", stats))
            while sent:
                handle(mconn.recv())
        except BaseException:  # noqa: BLE001
            try:
                mconn.send(("error", traceback.format_exc()))
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                from .io.images import _close_pool
                _close_pool()
            finally:
                _os._exit(0)

    def run_pass(t_start):
        """One loop over the frames for the blocks in `pending` (gather here or in the forked gather process);
        returns the frame loop's stats and the gathering time not hidden behind block compute."""
        nonlocal t_wait
        if not gproc:
            tw0 = t_wait
            st = frame_loop(submit_block, log, lambda: sum(job_bytes.get(j, 0) for j in block_jobs if not j.done()))
            tw = time.time()
            for fut in block_jobs:
                fut.result()
            block_jobs.clear()
            job_bytes.clear()
            t_wait += time.time() - tw
            t_read = time.time() - t_start - (t_wait - tw0)    # gathering (and decode waits) not hidden behind block compute
        else:
            import multiprocessing as mp
            import socket
            import sys
            from .io.images import _close_pool
            _close_pool()           # the decode pool's threads do not survive fork: the child starts its own workers
            for s_ in (sys.stdout, sys.stderr):
                s_.flush()
            pconn, cconn = mp.Pipe(duplex=True)
            pfs, cfs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            child = mp.get_context("fork").Process(target=gather_child, args=(cconn, cfs), name="xpg-gather")
            import warnings
            with warnings.catch_warnings():
                # fork of a multi-threaded process: the child uses only NumPy and the threads / decode processes it
                # starts itself, never a lock or a library state (torch, CUDA) of this process's other threads
                warnings.simplefilter("ignore", DeprecationWarning)
                child.start()
            cconn.close()
            cfs.close()
            st = None
            t_idle = 0.0                                 # waiting for the gather process (gather not hidden)

            def receive(meta, nfd):
                fds = _recv_fds(pfs, nfd)
                it = iter(fds)
                bufs = {}
                for key, sel, ashape, cshape in meta:
                    arr = _map_fd(next(it), ashape, np.float32)
                    bufs[key] = (sel, arr, _map_fd(next(it), cshape, np.int16) if cshape is not None else None)
                for fd in fds:
                    _os.close(fd)
                return bufs
            try:
                while st is None:
                    tw = time.time()
                    try:
                        msg = pconn.recv()
                    except EOFError:
                        child.join(5)
                        raise RuntimeError(f"shoebox gather process died (exit code {child.exitcode}; out of memory?); "
                                           "XPG_GATHER_PROCESS=0 gathers in this process") from None
                    t_idle += time.time() - tw
                    if msg[0] == "block":
                        process_block(msg[1], receive(msg[2], msg[3]))
                        pconn.send(("done", msg[1]))
                    elif msg[0] == "log":
                        log(msg[1])
                    elif msg[0] == "end":
                        st = msg[1]
                    else:
                        raise RuntimeError("shoebox gather process failed:\n" + msg[1])
            except BaseException:
                child.terminate()
                raise
            finally:
                child.join(60)
                if child.is_alive():
                    child.kill()
                    child.join()
                pconn.close()
                pfs.close()
            t_read = t_idle
            if verbose and st.get("queue_deep"):
                log(f"  INTEGRATE: gather process ran up to {st['queue_deep']} blocks ahead (memory headroom)")
        return st, t_read

    all_frames_full = all_frames
    t_read = t_dec = t_gat = 0.0
    for pi, pblocks in enumerate(passes):
        t_start = tr if pi == 0 else time.time()
        if len(passes) > 1:
            if pi:
                # each pass starts like a sweep: the adaptive sigma_D / sigma_M carry over only between the blocks of
                # one pass
                adapt.update(sd=sd, sm=sm, rcw=rc_default_weights())
            pending = list(pblocks)
            lo_, hi_ = min(need_lo[b] for b in pblocks), max(need_hi[b] for b in pblocks)
            all_frames = [f for f in all_frames_full if lo_ <= f - 1 <= hi_]
            if verbose and all_frames:
                log(f"  INTEGRATE: pass {pi + 1}/{len(passes)}: {len(pblocks)} blocks, frames {all_frames[0]}-{all_frames[-1]}")
        st, t_r = run_pass(t_start)
        t_read += t_r
        t_dec += st["t_dec"]
        t_gat += st["t_gat"]
    if worker is not None:
        worker.shutdown(wait=True)
    refl.update(out)
    refl["overlap"] = out_overlap
    if fz_out is not None:
        refl["fz"] = fz_out
    refl["iz0"], refl["dz"], refl["fb"] = iz0, dz, fb
    refl["region"] = region
    refl["ix0"], refl["iy0"] = ix0 * pxs, iy0 * pxs        # detector pixels (first pixel of the first super-pixel)
    if (pxs > 1).any():
        refl["pxs"] = pxs
    if verbose:
        ok = np.isfinite(refl["iprf"])
        log(f"  INTEGRATE: {ok.sum()} reflections integrated in {time.time() - t0:.1f}s "
            f"(io+gather {t_read:.1f}s [frame decode wait {t_dec:.1f}s, shoebox gather {t_gat:.1f}s], compute "
            f"{t_gpu:.1f}s{', gather process' if gproc else ', overlapped'})")
    res = IntegrationResult(refl, model, prof_by_block, dict(total=time.time() - t0, gpu=t_gpu))
    res.block_params = block_params
    if verbose and block_params and adaptive:
        sms = [v["sigma_m"] for v in block_params.values()]
        sds = [v["sigma_d"] for v in block_params.values()]
        log(f"  INTEGRATE: per-block sigma_M {min(sms):.4f}-{max(sms):.4f} deg, sigma_D {min(sds):.4f}-{max(sds):.4f} deg")
    return res


def _estimate_sigmas(images, exp, model0: IntegrationModel, dmin, frames, n_sample_frames=30, verbose=True, log=print,
                    extra_exps=(), bkg_template=None, foreign_spots=None, max_iter=None, guard=None):
    """Estimate beam divergence and mosaicity (e.s.d.s) from second moments of strong spots.

    Corrections: pixel footprint (q^2/12 per axis) and frame width (osc^2 zeta^2/12)."""
    g = exp.geom
    f0, f1 = frames
    nf = f1 - f0 + 1
    # sample several short ranges across the sweep
    ranges = []
    nr = 5
    # each range covers >= 5 degrees (fine slicing: 10 frames of 0.2 deg held too few strong spots)
    span = max(int(round(n_sample_frames / 3)), int(np.ceil(5.0 / max(abs(g.osc), 1e-6))), 3)
    span = min(span, max(nf // nr, 3))
    for i in range(nr):
        a = f0 + int(i * (nf - span) / max(nr - 1, 1))
        ranges.append((a, min(a + span - 1, f1)))
    model = prev = model0
    n_iter = max_iter or 3
    history = []                    # (sigma_D, sigma_M, n strong) per iteration: run-away growth = box-driven
    for it in range(n_iter):
        sel_md, sel_mm, sel_fit, per_range = [], [], [], []
        for a, b in ranges:
            edges = np.array([a - 1.0, float(b)])
            res = integrate(images, [exp], edges, model, dmin, frames=(max(f0, a - 5), min(f1, b + 5)),
                            verbose=False, learn_profiles=False, adaptive=False, bkg_template=bkg_template,
                            extra_lattices=[[e] for e in extra_exps] or None, foreign_spots=foreign_spots,
                            moments_only=True)
            r = res.refl
            z_ok = (r["z"] >= a - 1) & (r["z"] < b) & (r.get("lattice", 0) == 0) & ~r["overlap"]
            isg = r["isum"] / np.sqrt(np.maximum(r["vsum"], 1e-6))
            base = z_ok & np.isfinite(r["isum"]) & (r["isum"] > 0) & (r["n_over"] == 0) & (r["frac_rec"] > 0.98)
            base &= _rocking_resolved(r)
            per_range.append((r, base, isg))
        # per range: I/sigma > 20, or > 10 when that leaves < 150; ranges with < 20 such spots are skipped. Weak data
        # with no usable range: all ranges pooled at I/sigma > 5 (>= 30 spots)
        sels = []
        for r, base, isg in per_range:
            strong = base & (isg > 20)
            if strong.sum() < 150:
                strong = base & (isg > 10)
            sels.append(strong if strong.sum() >= 20 else None)
        if all(x is None for x in sels) and sum(int((bs & (ig > 5.0)).sum()) for _, bs, ig in per_range) >= 30:
            sels = [bs & (ig > 5.0) for _, bs, ig in per_range]
        for (r, base, isg), strong in zip(per_range, sels):
            if strong is None or not strong.any():
                continue
            Xv = g.pixel_to_lab(r["x"][strong], r["y"][strong])
            Xlen = np.linalg.norm(Xv, axis=1)
            cosa = np.abs(Xv @ g.ed3) / Xlen
            # pixel-footprint variance per axis in eps units (tangential q/|X|, radial q cos(a)/|X|)
            pix = (g.qx / Xlen) ** 2 * (1 + cosa ** 2) / 24.0
            if "pxs" in r:                                    # super-pixels: s x s pixel footprint
                pix = pix * r["pxs"][strong] ** 2
            sel_md.append(np.c_[r["m_d"][strong], pix])
            fbs = r["fb"][strong] if "fb" in r else 1
            sel_mm.append(r["m_m"][strong] - (np.radians(g.osc) * fbs * r["zeta_k"][strong]) ** 2 / 12.0)   # bin width
            one = strong & (r["fb"] == 1) if "fb" in r else strong          # per-frame rocking-curve fit
            sel_fit.append((r["fz"][one], r["iz0"][one], r["phi"][one], r["zeta_k"][one]))
        if not sel_md:
            break
        md2 = np.concatenate(sel_md)
        mm = np.concatenate(sel_mm)
        # moments inside the N_FG-sigma foreground are truncated: correct analytically
        # (2D Gaussian within radius n: <r^2/2>/sigma^2 = 1 - n^2/2 e^{-n^2/2}/(1-e^{-n^2/2}))
        t2 = 1.0 - (FG_R ** 2 / 2) * np.exp(-FG_R ** 2 / 2) / (1 - np.exp(-FG_R ** 2 / 2))
        md = md2[:, 0] / t2 - md2[:, 1]        # truncation-corrected moment minus pixel footprint
        sd_new = np.degrees(np.sqrt(max(np.median(md), 1e-12)))
        # grid centre: the moment estimate, but not below half the current width. A rocking curve much narrower than a
        # frame leaves almost no second moment after the frame-width correction (5MQ6, 0.5 deg frames: 0.084 / 0.068 /
        # 0.065 deg, then a ~0 guess whose 0.2-3x grid cannot reach 0.065 -> the 0.005 floor)
        sm_new = _fit_rocking_width(sel_fit, g, max(np.degrees(np.sqrt(max(np.median(mm), 1e-12))), 0.5 * model.sigma_m))
        new = IntegrationModel(sigma_d=float(np.clip(sd_new, 0.005, 1.0)),
                               sigma_m=float(np.clip(sm_new, 0.005, 5.0)), gain=model.gain)
        history.append((float(new.sigma_d), float(new.sigma_m), int(md.size)))
        if verbose:
            log(f"  INTEGRATE: estimated sigma_D {new.sigma_d:.4f} deg, sigma_M {new.sigma_m:.4f} deg "
                f"from {md.size} strong spots (iteration {it + 1})")
        # truncation-corrected widths of a Gaussian spot do not depend on the box: continued growth with the
        # box means contamination (neighbouring / foreign-crystal spots) -> keep the previous estimate
        if (guard is None or guard) and it >= 1 and (new.sigma_d > 1.15 * model.sigma_d or new.sigma_m > 1.15 * model.sigma_m) and \
                (model.sigma_d > 1.15 * prev.sigma_d or model.sigma_m > 1.15 * prev.sigma_m):
            if verbose:
                log("  INTEGRATE: spot-width estimate keeps growing with the box (contamination): keeping iteration "
                    f"{it} values")
            break
        prev, model = model, new
        if max_iter and it >= 2 and abs(model.sigma_d / prev.sigma_d - 1) < 0.03 and abs(model.sigma_m / prev.sigma_m - 1) < 0.03:
            break
    model.history = history
    return model


def _nnls_gram(A, y):
    """argmin ||A w - y||, w >= 0, for tall A (n x K, K small) through the Cholesky factor of A^T A: the same
    minimiser from a K x K problem (scipy's NNLS on the full A took ~1 s per integration block on the cluster)."""
    from scipy.linalg import LinAlgError, cholesky, solve_triangular
    from scipy.optimize import nnls
    try:
        R = cholesky(A.T @ A)
        return nnls(R, solve_triangular(R.T, A.T @ y, lower=True))[0]
    except (LinAlgError, ValueError):
        return nnls(A, y)[0]


def _rocking_resolved(r, max_edge=0.5):
    """Reflections whose net counts show a rocking curve within the box: the first and last frames (or bins) well
    below the frame of the predicted centre. A feature present on every frame has none (8TCA: under-subtracted high
    background near the beamstop, 68% of the 'strong' spots, net counts flat over the box) and drives the
    spot-width estimate to the box size. A Gaussian rocking curve up to ~3x wider than the box was sized for still
    passes (edge / centre 0.41)."""
    fz = r.get("fz")
    if fz is None:
        return np.ones(len(r["x"]), bool)
    n = len(fz)
    dz = np.minimum(np.asarray(r["dz"], int), fz.shape[1])
    rows = np.arange(n)
    c = np.clip(np.floor((r["z"] - r["iz0"]) / np.maximum(r["fb"], 1)).astype(int), 0, np.maximum(dz - 1, 0))
    centre = fz[rows, c]
    # outermost RECORDED frames on each side (frames beyond the sweep are exactly 0)
    k = np.arange(fz.shape[1])[None]
    rec = (fz != 0) & (k < dz[:, None])
    first = np.where(rec.any(1), np.argmax(rec, 1), 0)
    last = np.where(rec.any(1), fz.shape[1] - 1 - np.argmax(rec[:, ::-1], 1), 0)
    edge = 0.5 * (fz[rows, first] + fz[rows, last])
    return (dz < 3) | ((centre > 0) & (edge < max_edge * centre) & (last - first >= 2))


def _fit_rocking_width(sel_fit, g: Geometry, sm_guess, lo_fac=0.2, hi_fac=3.0):
    """Least-squares fit of the frame-integrated Gaussian rocking curve to the per-frame
    intensity fractions of strong reflections (grid search in log sigma_M)."""
    from scipy.special import erf
    fz = np.concatenate([s[0] for s in sel_fit])
    iz0 = np.concatenate([s[1] for s in sel_fit])
    phi = np.concatenate([s[2] for s in sel_fit])
    zeta = np.abs(np.concatenate([s[3] for s in sel_fit]))
    tot = fz.sum(1, keepdims=True)
    ok = tot[:, 0] > 0
    fz, iz0, phi, zeta, tot = fz[ok], iz0[ok], phi[ok], zeta[ok], tot[ok]
    frac = fz / tot
    k = iz0[:, None] + np.arange(fz.shape[1])[None]
    lo = g.z_to_phi(k.astype(float))
    hi = g.z_to_phi(k + 1.0)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    best = (np.inf, sm_guess)
    for sm in sm_guess * np.exp(np.linspace(np.log(lo_fac), np.log(hi_fac), 61)):
        s = sm / zeta[:, None] * np.sqrt(2)
        R = 0.5 * (erf((hi - phi[:, None]) / s) - erf((lo - phi[:, None]) / s))
        R = R / np.maximum(R.sum(1, keepdims=True), 1e-9)
        cost = np.sum(tot * (frac - R) ** 2)
        if cost < best[0]:
            best = (cost, sm)
    return float(best[1])

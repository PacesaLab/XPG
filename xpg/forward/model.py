"""Forward model: profile kernel, voxel index of a window of frames, and the likelihood passes.

Profile (the shoebox integrator's model profile): f_h(p, k) = G(u1, u2) J / seff^2 x R_k with u = eps / seff in the
Kabsch frame (eps1, eps2), G the unit 2D Gaussian (or the learnt profile), J the pixel area in eps units and R_k the
frame-integrated rocking curve (erf differences). Mirrors make_kernels.geometry_terms in integrate_kernels.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .. import tnp as jnp
from ..geometry import PIX0, cell_to_B
from ..models import CELL_PARAMS, cell_from_free_np, free_from_cell
from ..integrate import _kernel_device
from ..integrate_kernels import RC_C, _detector_consts, _eps12, _phi_rc


# peak memory per window voxel (frame x detector pixel) of a window fit: mu and the CG product (float32), the int16
# counts store, chunk temporaries and the per-chunk kernel temporaries.
BYTES_PER_VOXEL = {"cuda": 25.0, "xpu": 25.0, "cpu": 26.0, "mps": 26.0}


def host_memory_available():
    """Bytes of host memory available to this process: SLURM job limit / cgroup limit and MemAvailable (Linux),
    free + inactive + purgeable pages (macOS); None if unknown."""
    import os
    import re
    import subprocess
    avail = None
    try:
        if os.path.exists("/proc/meminfo"):
            m = re.search(r"MemAvailable:\s+(\d+) kB", open("/proc/meminfo").read())
            avail = int(m.group(1)) * 1024 if m else None
        else:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+) bytes", out).group(1))
            pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1)) for k in
                        ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
                        if re.search(rf"{k}:\s+(\d+)", out))
            avail = pages * page
    except Exception:
        avail = None
    limits = []
    if os.environ.get("SLURM_MEM_PER_NODE"):
        limits.append(int(os.environ["SLURM_MEM_PER_NODE"]) * 2 ** 20)
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 2 ** 60:
                limits.append(int(v))
        except Exception:
            pass
    if limits:
        avail = min([avail] + limits) if avail else min(limits)
    return avail


def memory_budget(dev, gb=None, frac=None):
    """Bytes the window fits may use on dev: gb if given, else frac (0.6 GPU, 0.5 host) of the free memory."""
    if gb:
        return float(gb) * 2 ** 30
    if dev.type in ("cuda", "xpu"):
        # capped at 40 GiB; XPG_FM_MAX_GPU_GB raises it
        import os
        try:
            free, _ = (torch.cuda if dev.type == "cuda" else torch.xpu).mem_get_info(dev)
        except Exception:  # pragma: no cover (XPU builds without mem_get_info)
            free = 16 * 2 ** 30
        cap = float(os.environ.get("XPG_FM_MAX_GPU_GB", 40.0)) * 2 ** 30
        return min((0.6 if frac is None else frac) * free, cap)
    avail = host_memory_available()
    if avail is None:
        avail = 8 * 2 ** 30
    return (0.5 if frac is None else frac) * avail


def auto_window(refl, nx, ny, budget, dev_type="cpu", frac=0.8, w_max=200, pxs=1):
    """Window length (frames) for a memory budget (bytes): as long as the budget allows (longer windows: less
    overlap, fewer re-fitted margins), at least long enough for 95% of the supports to be owned by some window (the
    supports' frame extent plus a 10-frame step). nx, ny: the (super-pixel) detector. Returns (frames, info dict)."""
    _, _, _, _, dz = support(refl, frac, pxs)
    margin = int(np.ceil(np.median(dz) / 2))
    w_min = int(max(2 * margin + 10, np.quantile(dz, 0.95) + 2))
    bpv = BYTES_PER_VOXEL.get(dev_type, 36.0)
    w_mem = int((budget - 2 * 2 ** 30) // (bpv * nx * ny))
    w = int(np.clip(w_mem, w_min, w_max))
    return w, dict(w_min=w_min, w_mem=w_mem, budget_gb=budget / 2 ** 30, fits=w_mem >= w_min,
                   gb_needed_min=(w_min * bpv * nx * ny + 2 * 2 ** 30) / 2 ** 30)


def box_sides(refl):
    """Shoebox side (px) of each predicted reflection: the integrator stores the corner (ix0) of a box centred on
    the pixel nearest the predicted centre, so half = round(x - PIX0) - ix0."""
    half = np.floor(refl["x"] - PIX0 + 0.5).astype(int) - refl["ix0"]
    return 2 * half + 1


@dataclass
class Window:
    """Reflections whose support (shoebox) overlaps frames [f0, f1) (0-based frame indices) and the detector tile
    [x0, x0 + nx) x [y0, y0 + ny) (array columns / rows; the whole detector by default). Voxels are indexed densely
    over the window (frame - f0, row - y0, column - x0): near-reflection supports cover 35-65% of all pixel-frames,
    so a sparse index buys little."""
    f0: int
    f1: int
    nx: int                   # tile width / height (the detector's without tiling)
    ny: int
    rows: np.ndarray          # reflection indices (into refl)
    groups: list              # (row indices into rows, Dz, bx): reflections sharing a support shape
    voxel_ids: list
    x0: int = 0               # tile origin on the detector (array column / row)
    y0: int = 0
    pxs: int = 1              # super-pixel side: x0, y0, nx, ny and the supports in s x s super-pixels

    @property
    def n_vox(self):
        return (self.f1 - self.f0) * self.ny * self.nx


def support(refl, frac=0.8, pxs=1):
    """Support of each reflection: the central ``frac`` of the shoebox in each direction (the shoebox spans +-5
    sigma; 0.8 -> +-4 sigma, beyond which a Gaussian holds < 1e-4 per axis). Returns (ix0, iy0, iz0, bx, dz).
    pxs > 1: ix0, iy0, bx in s x s super-pixels (detector pixel i in super-pixel i // s), the half-side rounded up
    around the super-pixel holding the centre."""
    half = (box_sides(refl) - 1) // 2
    hz = (refl["dz"] - 1) // 2
    hz2 = np.maximum(np.ceil(frac * hz).astype(int), 1)
    if pxs > 1:
        h2 = np.maximum(np.ceil(frac * half / pxs).astype(int), 1)
        cx = np.floor(refl["x"] - PIX0 + 0.5).astype(int) // pxs
        cy = np.floor(refl["y"] - PIX0 + 0.5).astype(int) // pxs
        return cx - h2, cy - h2, refl["iz0"] + hz - hz2, 2 * h2 + 1, 2 * hz2 + 1
    h2 = np.maximum(np.ceil(frac * half).astype(int), 1)
    return (refl["ix0"] + half - h2, refl["iy0"] + half - h2, refl["iz0"] + hz - hz2, 2 * h2 + 1, 2 * hz2 + 1)


def make_window(refl, f0, f1, nx, ny, frac=0.8, tile=None, pxs=1):
    """Reflections with fb == 1 (no phi binning) whose support (see ``support``) overlaps [f0, f1) and, with
    tile = (x0, x1, y0, y1) (array columns / rows), the detector tile; nx, ny: the detector (pxs > 1: all in s x s
    super-pixels)."""
    ix0, iy0, iz0, bx, dz = support(refl, frac, pxs)
    z0, z1 = iz0, iz0 + dz
    x0, x1, y0, y1 = tile if tile is not None else (0, nx, 0, ny)
    sel = np.flatnonzero((refl["fb"] == 1) & (z1 > f0) & (z0 < f1) & (ix0 < x1) & (ix0 + bx > x0)
                         & (iy0 < y1) & (iy0 + bx > y0))
    groups, vox = [], []
    keys = np.stack([dz[sel], bx[sel]], 1)
    for key in np.unique(keys, axis=0):
        Dz, b = int(key[0]), int(key[1])
        gi = np.flatnonzero((keys[:, 0] == Dz) & (keys[:, 1] == b))
        groups.append((gi, Dz, b))
        vox.append(None)            # voxel indices are computed on the device per chunk (WindowModel.vid)
    w = Window(f0, f1, x1 - x0, y1 - y0, sel, groups, vox, x0=x0, y0=y0, pxs=pxs)
    w.sup = (ix0, iy0, iz0)
    return w


def plan_tiles(nx, ny, n_frames, budget, dev_type, margin, max_tiles=64):
    """Detector tiling (ntx, nty) for windows of n_frames frames within a memory budget (bytes): the fewest tiles
    (squarest first) whose tile area including the margins (pixels on interior borders) fits. (1, 1) if the whole
    detector fits."""
    bpv = BYTES_PER_VOXEL.get(dev_type, 26.0)
    px = (budget - 2 * 2 ** 30) / (bpv * n_frames)
    best = None
    for n in range(1, max_tiles + 1):
        for ntx in range(1, n + 1):
            if n % ntx:
                continue
            nty = n // ntx
            tw = -(-nx // ntx) + (2 * margin if ntx > 1 else 0)
            th = -(-ny // nty) + (2 * margin if nty > 1 else 0)
            if tw * th <= px:
                cand = (abs(np.log(tw / th)), ntx, nty)
                best = cand if best is None or cand < best else best
        if best is not None:
            return best[1], best[2]
    return None


def tile_bounds(nx, ny, ntx, nty, margin):
    """[(core x0, x1, y0, y1), (tile x0, x1, y0, y1)] of an ntx x nty tiling: the cores partition the detector, the
    tiles extend them by margin pixels (clipped to the detector)."""
    xs = np.linspace(0, nx, ntx + 1).round().astype(int)
    ys = np.linspace(0, ny, nty + 1).round().astype(int)
    out = []
    for j in range(nty):
        for i in range(ntx):
            core = (xs[i], xs[i + 1], ys[j], ys[j + 1])
            tile = (max(0, xs[i] - margin), min(nx, xs[i + 1] + margin), max(0, ys[j] - margin), min(ny, ys[j + 1] + margin))
            out.append((core, tile))
    return out


def expand_terms(terms):
    """sum_t R_t (n, Dz) x P_t (n, Dy, Dx) -> (n, Dz, Dy, Dx)."""
    f = None
    for R, P in terms:
        g = R[:, :, None, None] * P[:, None]
        f = g if f is None else f + g
    return f


def fraction_terms(det, axis, s0, frames_phi0, osc, sd, sm, S, phi, ix0, iy0, iz0, Dz, Dx, prof=None, region=None,
                   umax=None, halo=None, rcw=None, pxs=1):
    """Separable factors [(R (n, Dz), P (n, Dx, Dx)), ...] of the fraction of each reflection recorded in each
    pixel / frame of its shoebox (one term; two with the halo).
    prof (R, G, G): learnt 2D reference profiles (density in sigma_eff units, the shoebox integrator's), sampled
    bilinearly in the reflection's region; None: parametric. halo (w, kappa): two-component scale mixture of the
    profile, (1 - w) core + w Gaussian core broadened by kappa in all three directions (M2 / A7; spots have heavier
    tails than a Gaussian or than the learnt profile's domain); with learnt profiles the core is the learnt profile.
    sd, sm may be tensors (width fits). pxs > 1: ix0, iy0, Dx in s x s super-pixels, the profile sampled at their
    centres (midpoint rule; J the super-pixel's solid angle; s <= sigma / 2.5 by construction, see driver)."""
    Dy = Dx
    ar = torch.arange(Dx, dtype=torch.float32, device=S.device)[None]
    if pxs > 1:
        x = (ix0[:, None].float() + ar) * pxs + ((pxs - 1) * 0.5 + PIX0)
        y = (iy0[:, None].float() + ar) * pxs + ((pxs - 1) * 0.5 + PIX0)
    else:
        x = ix0[:, None].float() + ar + PIX0
        y = iy0[:, None].float() + ar + PIX0
    c1 = torch.linalg.cross(S, s0[None].expand_as(S))
    e1 = c1 / torch.linalg.norm(c1, dim=-1, keepdim=True)
    c2 = torch.linalg.cross(S, e1)
    e2 = c2 / torch.linalg.norm(c2, dim=-1, keepdim=True)
    zeta = e1 @ axis
    eps1, eps2 = _eps12(det, S, e1, e2, x, y)                                   # (n, Dy, Dx) rad
    if Dx >= 3:
        g1x, g1y = torch.gradient(eps1, dim=(2, 1), edge_order=2)
        g2x, g2y = torch.gradient(eps2, dim=(2, 1), edge_order=2)
        J = torch.abs(g1x * g2y - g2x * g1y)
    else:
        J = torch.full_like(eps1, float(det["qx"] * pxs) ** 2)
    k = iz0[:, None].float() + torch.arange(Dz, dtype=torch.float32, device=S.device)[None]
    lo = frames_phi0 + osc * k
    hi = lo + osc
    lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
    phb, zb = phi[:, None].expand_as(lo), zeta[:, None].expand_as(lo)
    R = _phi_rc(lo, hi, phb, zb, sm, rcw)                                           # (n, Dz)
    pix = torch.sqrt(J[:, Dy // 2, Dx // 2])
    seff = sd * torch.sqrt(1.0 + (pix / sd) ** 2 / 12.0)
    r2 = eps1 ** 2 + eps2 ** 2

    def gauss2(se):
        return torch.exp(-0.5 * r2 / se[:, None, None] ** 2) / (2 * np.pi) * J / se[:, None, None] ** 2

    def with_halo(core):
        if halo is None:
            return [core]
        w, kap = halo
        sh = torch.sqrt((kap * sd) ** 2 + pix ** 2 / 12.0)                          # broadened core, pixel-convolved
        Rh = _phi_rc(lo, hi, phb, zb, kap * sm)
        return [((1 - w) * core[0], core[1]), (w * Rh, gauss2(sh))]
    if prof is None:
        return with_halo((R, gauss2(seff)))
    else:
        G = prof.shape[-1]
        u1 = eps1 / seff[:, None, None]
        u2 = eps2 / seff[:, None, None]
        f1 = (u1 + umax) / (2 * umax) * (G - 1)
        f2 = (u2 + umax) / (2 * umax) * (G - 1)
        inside = (f1 >= 0) & (f1 <= G - 1) & (f2 >= 0) & (f2 <= G - 1)
        f1 = torch.clamp(f1, 0, G - 1 - 1e-4)
        f2 = torch.clamp(f2, 0, G - 1 - 1e-4)
        i1, i2 = f1.floor().long(), f2.floor().long()
        t1, t2 = f1 - i1, f2 - i2
        r = region[:, None, None]
        v = ((1 - t1) * (1 - t2) * prof[r, i2, i1] + t1 * (1 - t2) * prof[r, i2, i1 + 1]
             + (1 - t1) * t2 * prof[r, i2 + 1, i1] + t1 * t2 * prof[r, i2 + 1, i1 + 1])
        P = torch.where(inside, v, torch.zeros((), device=v.device)) * J / seff[:, None, None] ** 2
    # learnt core: the halo (A7) adds the tails beyond the learnt profile's domain (strong spots of crowded data
    # have them out to ~8 sigma: TDS / mosaic tails)
    return with_halo((R, P))


GEOM = ("crystal rot x", "crystal rot y", "crystal rot z", "detector shift x", "detector shift y",
        "detector shift z", "beam tilt 1", "beam tilt 2")
N_GEOM = len(GEOM)


def _rotate(v, axis, ang):
    """Rodrigues rotation of v (n, 3) about the unit axis (3,) by angles ang (n,) or a scalar (radians)."""
    ang = np.broadcast_to(np.asarray(ang, float), v.shape[:1])[:, None]
    c, s_ = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s_ + np.outer(v @ axis, axis) * (1 - c)


def _rotvec(v, rv):
    """Rotation of v (n, 3) by the rotation vector rv (3,)."""
    a = float(np.linalg.norm(rv))
    return v if a == 0 else _rotate(v, rv / a, a)


def perturb_reflections(S, phi, s0, axis, omega, beam, dr0=None):
    """Diffracted beam S (n, 3) and spindle angle phi (n,, rad) after a change dr0 (n, 3) of the reciprocal vectors
    at phi = 0 (cell), a crystal rotation omega (rotation vector, lab frame at phi = 0) and a beam tilt (2 angles about
    two axes perpendicular to s0), by re-solving the diffraction condition s0' . R(phi') r0' = -|r0'|^2 / 2 exactly
    (root nearest the old phi; float64)."""
    m = axis / np.linalg.norm(axis)
    r0 = _rotate(S - s0, m, -phi)                                       # reciprocal vector at phi = 0
    if dr0 is not None:
        r0 = r0 + dr0
    r0 = _rotvec(r0, np.asarray(omega, float))
    ea = np.cross(s0, m)
    ea /= np.linalg.norm(ea)
    eb = np.cross(s0 / np.linalg.norm(s0), ea)
    s0n = _rotvec(s0[None], beam[0] * ea + beam[1] * eb)[0]
    par = (r0 @ m)[:, None] * m
    a = (r0 - par) @ s0n
    b = np.cross(m, r0) @ s0n
    c = -0.5 * np.einsum("ij,ij->i", r0, r0) - par @ s0n
    base = np.arctan2(b, a)
    dlt = np.arccos(np.clip(c / np.maximum(np.hypot(a, b), 1e-30), -1, 1))
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    d1, d2 = wrap(base + dlt - phi), wrap(base - dlt - phi)
    phn = phi + np.where(np.abs(d1) <= np.abs(d2), d1, d2)
    return _rotate(r0, m, phn) + s0n, phn


class WindowModel:
    """Per-group kernel inputs of a window on one device, and the three likelihood passes."""

    def __init__(self, refl, win, exp_blocks, model, bkg_template, dev=None, profiles=None, n_box=5.0, max_vox=4e7,
                 block_params=None, budget=None):
        """profiles: {block: (R, G, G)} learnt reference profiles (results['integration_profiles']); None: Gaussian.
        block_params: {block: dict(sigma_d, sigma_m (deg), rcw)} the integrator's per-block widths and rocking-curve
        shape (results['integration_block_params']; the learnt profiles are in units of the block's sigma_D). The
        global widths (set_widths, M2) scale all blocks."""
        self.dev = dev or _kernel_device()
        if budget is not None:          # kernel chunk temporaries (~40 B / reflection-voxel) within 5% of the budget
            max_vox = float(np.clip(0.05 * budget / 40.0, 2e6, max_vox))
        self.umax = 0.9 * n_box
        self.win = win
        self.refl = refl
        self.items = []
        sd, sm = np.radians(model.sigma_d), np.radians(model.sigma_m)
        # profile widths shared by all reflections (radians); halo (w, kappa) or None
        self.width = dict(sd=float(sd), sm=float(sm), halo=None)
        # geometry correction shared by the window (M3): crystal rotation (3, rad, lab frame at phi = 0), detector
        # translation (3, mm, lab), beam tilt (2, rad), cell (the lattice system's free parameters: relative change of
        # the lengths, change of the angles in rad); zero = XPG's per-block geometry
        self.system = exp_blocks[int(refl["block"][win.rows[0]])].crystal.system if win.rows.size else "a"
        self.geom_names = GEOM + tuple("cell " + p for p in CELL_PARAMS[self.system])
        self.geom = np.zeros(len(self.geom_names))
        self.beta = None            # flat local background level per reflection (diagnostics), else None
        self.cache = True           # cache the separable profile factors per chunk (see _f)
        # scatter: boolean compaction on CUDA, zero-masked full index_add on the CPU (0.85 vs 1.03 s)
        self.compact = None
        with jnp.on(self.dev):
            ix0s, iy0s, iz0s = win.sup
            for (gi, Dz, b), vid in zip(win.groups, win.voxel_ids):
                r = win.rows[gi]
                blk = refl["block"][r]
                per = max(1, int(max_vox // (Dz * b * b)))       # reflections per chunk (bounded kernel memory)
                for bk in np.unique(blk):
                    idx_b = np.flatnonzero(blk == bk)
                    for c0 in range(0, idx_b.size, per):
                        m = idx_b[c0:c0 + per]
                        g = exp_blocks[int(bk)].geom
                        det = _detector_consts(g, bkg_template)
                        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
                        pr = None
                        if profiles is not None and int(bk) in profiles:
                            pr = t(profiles[int(bk)])
                        cr = exp_blocks[int(bk)].crystal
                        bp = (block_params or {}).get(int(bk))
                        rcw = None
                        if bp and len(bp.get("rcw") or []) == len(RC_C):
                            rcw = t(np.asarray(bp["rcw"], np.float32))
                        self.items.append(dict(
                            sd_rel=float(np.radians(bp["sigma_d"]) / sd) if bp else 1.0,
                            sm_rel=float(np.radians(bp["sigma_m"]) / sm) if bp else 1.0, rcw=rcw,
                            rows=r[m], Dz=Dz, b=b, prof=pr, s0_np=np.asarray(g.s0, float),
                            axis_np=np.asarray(g.axis, float), U=np.asarray(cr.U, float),
                            cell=np.asarray(cr.cell, float),
                            region=t(refl["region"][r[m]], torch.int64),
                            det=det, axis=t(g.axis), s0=t(g.s0), phi0=float(np.radians(g.z_to_phi(0.0))),
                            osc=float(np.radians(g.osc)), sd=float(sd), sm=float(sm),
                            S=t(refl["S"][r[m]]), phi=t(np.radians(refl["phi"][r[m]])),
                            ix0=t(ix0s[r[m]], torch.int64), iy0=t(iy0s[r[m]], torch.int64),
                            iz0=t(iz0s[r[m]], torch.int64)))

    def vid(self, it):
        """Dense voxel index (n, Dz, b, b) of a chunk, -1 outside the window / detector (computed, not stored)."""
        w = self.win
        ar_z = torch.arange(it["Dz"], device=self.dev)
        ar_b = torch.arange(it["b"], device=self.dev)
        kz = it["iz0"][:, None] + ar_z[None] - w.f0
        ky = it["iy0"][:, None] + ar_b[None] - w.y0
        kx = it["ix0"][:, None] + ar_b[None] - w.x0
        ok = ((kz >= 0) & (kz < w.f1 - w.f0))[:, :, None, None] & ((ky >= 0) & (ky < w.ny))[:, None, :, None] \
            & ((kx >= 0) & (kx < w.nx))[:, None, None, :]
        lin = (kz[:, :, None, None] * w.ny + ky[:, None, :, None]) * w.nx + kx[:, None, None, :]
        return torch.where(ok, lin, torch.full((), -1, dtype=lin.dtype, device=self.dev))

    def set_widths(self, sd=None, sm=None, halo="keep"):
        """Profile widths (radians) and halo (w, kappa) or None."""
        if sd is not None:
            self.width["sd"] = sd
        if sm is not None:
            self.width["sm"] = sm
        if halo != "keep":
            self.width["halo"] = halo

    def set_geometry(self, gamma):
        """Geometry correction (N_GEOM,): see GEOM."""
        self.geom = np.asarray(gamma, float).copy()

    def _geometry(self, it):
        """det, S, phi of a chunk under the current geometry correction (increments from the stored values, so a zero
        correction reproduces them exactly)."""
        if not np.any(self.geom):
            return it["det"], it["S"], it["phi"]
        rows = it["rows"]
        S, ph = np.asarray(self.refl["S"][rows], float), np.radians(np.asarray(self.refl["phi"][rows], float))
        dr0 = None
        if np.any(self.geom[8:]):
            names = CELL_PARAMS[self.system]
            fc0 = free_from_cell(self.system, it["cell"])
            ang = np.array([p in ("al", "be", "ga") for p in names])
            fc = np.where(ang, fc0 + np.degrees(self.geom[8:]), fc0 * (1.0 + self.geom[8:]))
            dA = it["U"] @ (cell_to_B(cell_from_free_np(self.system, fc)) - cell_to_B(it["cell"]))
            dr0 = np.asarray(self.refl["hkl"][rows], float) @ dA.T
        S1, ph1 = perturb_reflections(S, ph, it["s0_np"], it["axis_np"], self.geom[0:3], self.geom[6:8], dr0)
        S0, ph0 = perturb_reflections(S, ph, it["s0_np"], it["axis_np"], np.zeros(3), np.zeros(2))
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.dev)
        det = dict(it["det"])
        det["P0"] = it["det"]["P0"] + t(self.geom[3:6])[None]
        return det, t(S + (S1 - S0)), t(ph + (ph1 - ph0))

    def _key(self):
        w = self.width
        return (float(w["sd"]), float(w["sm"]), None if w["halo"] is None else tuple(map(float, w["halo"])),
                tuple(np.asarray(self.geom, float).round(15)))

    def _f(self, it):
        """Fractions of a chunk (n, Dz, b, b). The separable factors (rocking curve x detector profile, ~1/30 of the
        voxels) are cached per chunk and recomputed only when the widths or the geometry change: computing them
        (Kabsch frame of every pixel, profile lookup) was half of every likelihood pass."""
        return expand_terms(self._terms(it))

    def _terms(self, it):
        """The separable factors [(R, P), ...] of a chunk (cached per widths / geometry, see _f)."""
        key = self._key()
        if self.cache and it.get("_key") == key:
            return it["_terms"]
        w = self.width
        det, S, phi = self._geometry(it)
        with jnp.on(self.dev):
            terms = fraction_terms(det, it["axis"], it["s0"], it["phi0"], it["osc"], w["sd"] * it["sd_rel"],
                                   w["sm"] * it["sm_rel"], S, phi, it["ix0"], it["iy0"], it["iz0"], it["Dz"], it["b"],
                                   prof=it["prof"], region=it["region"], umax=self.umax, halo=w["halo"],
                                   rcw=it["rcw"], pxs=self.win.pxs)
        if self.cache:
            it["_key"], it["_terms"] = key, terms
        return terms

    def _rp(self, it):
        """The separable factors of a chunk stacked for the fused kernels: R (T, n, Dz), P (T, n, b, b) (cached with
        the factors), and the chunk's reflection rows on the device."""
        terms = self._terms(it)
        key = it.get("_key")
        if it.get("_rp_key") != key or "_rp" not in it:
            it["_rp"] = (torch.stack([R.float() for R, _ in terms]).contiguous(),
                         torch.stack([P.float() for _, P in terms]).contiguous())
            it["_rp_key"] = key
        if "_rows_t" not in it:
            it["_rows_t"] = torch.as_tensor(it["rows"], device=self.dev)
        return it["_rp"][0], it["_rp"][1], it["_rows_t"]

    def _fused(self):
        from . import kernels
        return self.cache and kernels.available(self.dev)

    def grad_ratio(self, A, M, NS):
        """Per reflection sum_voxels f A / M over the voxels with NS >= 0 (the Fisher-vector product of the intensity
        solve: A = mu of the direction, M = mu); fused on CUDA."""
        n = len(self.refl["x"])
        g = torch.zeros(n, dtype=torch.float32, device=self.dev)
        if self._fused():
            from . import kernels
            for it in self.items:
                Rs, Ps, rows = self._rp(it)
                kernels.gather(g, None, Rs, Ps, rows, it["iz0"], it["iy0"], it["ix0"], self.win, A, M, NS, 0)
            return g
        zero = torch.zeros((), device=self.dev)
        return self.grad(lambda vc, ok: torch.where(ok & (NS[vc] >= 0), A[vc] / M[vc], zero))[0]

    def grad_newton(self, M, NS):
        """Gradient and Fisher diagonal of the Poisson likelihood in I: per reflection sum f (1 - n / mu) and
        sum f^2 / mu over the voxels with counts (NS >= 0); fused on CUDA."""
        n = len(self.refl["x"])
        if self._fused():
            from . import kernels
            g = torch.zeros(n, dtype=torch.float32, device=self.dev)
            fd = torch.zeros(n, dtype=torch.float32, device=self.dev)
            for it in self.items:
                Rs, Ps, rows = self._rp(it)
                kernels.gather(g, fd, Rs, Ps, rows, it["iz0"], it["iy0"], it["ix0"], self.win, None, M, NS, 1)
            return g, fd
        zero = torch.zeros((), device=self.dev)
        return self.grad(lambda vc, ok: torch.where(ok & (NS[vc] >= 0), 1.0 - torch.clamp(NS[vc], min=0).float() / M[vc],
                                                    zero), M, valid=lambda vc: NS[vc] >= 0)

    def valid_fraction(self, valid):
        """Per reflection: fraction of its profile on trusted voxels (the shoebox integrator's PEAK criterion)."""
        n = len(self.refl["x"])
        num = torch.zeros(n, dtype=torch.float32, device=self.dev)
        den = torch.zeros(n, dtype=torch.float32, device=self.dev)
        for it in self.items:
            f = self._f(it)
            v = self.vid(it)
            ok = v >= 0
            rows = torch.as_tensor(it["rows"], device=self.dev)
            inside = (ok & valid[torch.clamp(v, min=0)]).float()
            num.index_add_(0, rows, (f * inside).sum(dim=(1, 2, 3)))
            den.index_add_(0, rows, (f * ok.float()).sum(dim=(1, 2, 3)))
        return num / torch.clamp(den, min=1e-12)

    def n_voxel_evals(self):
        return sum(len(it["rows"]) * it["Dz"] * it["b"] * it["b"] for it in self.items)

    def mu(self, I, B, beta="model"):
        """Phase A: mu = B + sum_h (I_h f_h + beta_h 1_h) over the ROI (scatter-add). beta: local background level
        of each reflection, flat over its support (counts per pixel-frame); "model": self.beta (None: no local
        terms)."""
        b_ = self.beta if isinstance(beta, str) else beta
        mu = B.clone() if B is not None else torch.zeros(self.win.n_vox, device=self.dev)
        zero = torch.zeros((), device=self.dev)
        if self._fused():
            from . import kernels
            for it in self.items:
                Rs, Ps, rows = self._rp(it)
                kernels.scatter(mu, Rs, Ps, I[rows].contiguous(), None if b_ is None else b_[rows].contiguous(),
                                it["iz0"], it["iy0"], it["ix0"], self.win)
            return mu
        for it in self.items:
            f = self._f(it)
            v = self.vid(it)
            ok = v >= 0
            rows = torch.as_tensor(it["rows"], device=self.dev)
            contrib = f * I[rows][:, None, None, None]
            if b_ is not None:
                contrib = contrib + b_[rows][:, None, None, None]
            if self.compact or (self.compact is None and self.dev.type == "cuda"):
                mu.index_add_(0, v[ok], contrib[ok])
            else:       # no boolean compaction (a device sync and two copies per chunk): invalid voxels add 0 at 0
                mu.index_add_(0, torch.clamp(v, min=0).reshape(-1), torch.where(ok, contrib, zero).reshape(-1))
        return mu

    def grad(self, w, mu=None, valid=None):
        """Phase C: per-reflection gradient sum_voxels f_h w (and, with mu, the Fisher diagonal sum f_h^2 / mu) by
        gather; tensors over all reflections of refl. w: a voxel tensor, or a function (vc, ok) -> weights of the
        gathered voxels (computed on the fly: no full-window weight array). mu: a voxel tensor or a function vc ->
        mu of the gathered voxels. valid: voxels with a likelihood term (the Fisher diagonal counts only those)."""
        n = len(self.refl["x"])
        g = torch.zeros(n, dtype=torch.float32, device=self.dev)
        fd = torch.zeros(n, dtype=torch.float32, device=self.dev) if mu is not None else None
        zero = torch.zeros((), device=self.dev)
        for it in self.items:
            f = self._f(it)
            v = self.vid(it)
            vc = torch.clamp(v, min=0)
            ok = v >= 0
            wv = w(vc, ok) if callable(w) else torch.where(ok, w[vc], zero)
            rows = torch.as_tensor(it["rows"], device=self.dev)
            g.index_add_(0, rows, (f * wv).sum(dim=(1, 2, 3)))
            if fd is not None:
                okm = ok if valid is None else ok & (valid(vc) if callable(valid) else valid[vc])
                mv = mu(vc) if callable(mu) else mu[vc]
                im = torch.where(okm, 1.0 / torch.clamp(mv, min=1e-6), zero)
                fd.index_add_(0, rows, (f * f * im).sum(dim=(1, 2, 3)))
        return g, fd



"""Fused scatter / gather passes of the window model on CUDA.

The eager passes (model.WindowModel.mu / grad) materialise, per chunk of reflections, the fractions f (n, Dz, b, b),
the int64 voxel index, the validity mask and the compacted copies before one index_add / gather: ~60 bytes of memory
traffic per reflection-voxel, which makes them memory-bound. Here one program per (reflection, frame) forms f = sum_t
R_t[z] P_t[y, x] in registers from the separable factors and does only the atomic add (scatter) or the masked loads
and the reduction (gather).

    scatter(mu, R, P, I, beta, iz0, iy0, ix0, win)            mu[v] += I f + beta over the in-window voxels
    gather_ratio(g, R, P, rows, iz0, iy0, ix0, win, A, M, NS)  g[row] += sum f A / M        (voxels with NS >= 0)
    gather_newton(g, fd, R, P, rows, ..., M, NS)              g[row] += sum f (1 - NS / M), fd[row] += sum f^2 / M

R: (T, n, Dz), P: (T, n, b, b) float32 contiguous (T = 1, or 2 with the halo). Used only when available() (CUDA and
Triton); XPG_FM_TRITON=0 keeps the eager passes."""
from __future__ import annotations

import os

try:                                    # Triton ships with the CUDA builds of PyTorch (not on macOS)
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None

BLOCK = 256


def available(dev) -> bool:
    return (triton is not None and getattr(dev, "type", str(dev)) == "cuda"
            and os.environ.get("XPG_FM_TRITON", "1") != "0")


if triton is not None:
    @triton.jit
    def _scatter_kernel(mu_ptr, R_ptr, P_ptr, I_ptr, beta_ptr, iz0_ptr, iy0_ptr, ix0_ptr,
                        n, Dz, b, f0, nf, y0, ny, x0, nx,
                        NT: tl.constexpr, HAS_BETA: tl.constexpr, BLK: tl.constexpr):
        row = tl.program_id(0)
        z = tl.program_id(1)
        kz = tl.load(iz0_ptr + row) + z - f0
        ky0 = tl.load(iy0_ptr + row) - y0
        kx0 = tl.load(ix0_ptr + row) - x0
        zok = (kz >= 0) & (kz < nf)
        I = tl.load(I_ptr + row)
        bb = b * b
        for p0 in range(0, bb, BLK):
            offs = p0 + tl.arange(0, BLK)
            m = offs < bb
            yy = offs // b
            xx = offs - yy * b
            ky = ky0 + yy
            kx = kx0 + xx
            ok = m & zok & (ky >= 0) & (ky < ny) & (kx >= 0) & (kx < nx)
            f = tl.zeros([BLK], dtype=tl.float32)
            for t in tl.static_range(NT):
                r = tl.load(R_ptr + (t * n + row) * Dz + z)
                f += r * tl.load(P_ptr + (t * n + row).to(tl.int64) * bb + offs, mask=m, other=0.0)
            val = f * I
            if HAS_BETA:
                val += tl.load(beta_ptr + row)
            lin = (kz * ny + ky) * nx + kx
            tl.atomic_add(mu_ptr + lin, val, mask=ok)

    @triton.jit
    def _gather_kernel(g_ptr, fd_ptr, R_ptr, P_ptr, rows_ptr, iz0_ptr, iy0_ptr, ix0_ptr, A_ptr, M_ptr, NS_ptr,
                       n, Dz, b, f0, nf, y0, ny, x0, nx,
                       NT: tl.constexpr, MODE: tl.constexpr, BLK: tl.constexpr):
        # MODE 0: g += f A / M; MODE 1: g += f (1 - max(NS, 0) / M), fd += f^2 / max(M, 1e-6); voxels with NS >= 0
        row = tl.program_id(0)
        z = tl.program_id(1)
        kz = tl.load(iz0_ptr + row) + z - f0
        ky0 = tl.load(iy0_ptr + row) - y0
        kx0 = tl.load(ix0_ptr + row) - x0
        zok = (kz >= 0) & (kz < nf)
        bb = b * b
        acc = tl.zeros([BLK], dtype=tl.float32)
        accd = tl.zeros([BLK], dtype=tl.float32)
        for p0 in range(0, bb, BLK):
            offs = p0 + tl.arange(0, BLK)
            m = offs < bb
            yy = offs // b
            xx = offs - yy * b
            ky = ky0 + yy
            kx = kx0 + xx
            ok = m & zok & (ky >= 0) & (ky < ny) & (kx >= 0) & (kx < nx)
            f = tl.zeros([BLK], dtype=tl.float32)
            for t in tl.static_range(NT):
                r = tl.load(R_ptr + (t * n + row) * Dz + z)
                f += r * tl.load(P_ptr + (t * n + row).to(tl.int64) * bb + offs, mask=m, other=0.0)
            lin = (kz * ny + ky) * nx + kx
            ns = tl.load(NS_ptr + lin, mask=ok, other=-1).to(tl.float32)
            v = ok & (ns >= 0.0)
            mv = tl.load(M_ptr + lin, mask=v, other=1.0)
            if MODE == 0:
                w = tl.load(A_ptr + lin, mask=v, other=0.0) / mv
            else:
                w = 1.0 - tl.maximum(ns, 0.0) / mv
                accd += tl.where(v, f * f / tl.maximum(mv, 1e-6), 0.0)
            acc += tl.where(v, f * w, 0.0)
        rr = tl.load(rows_ptr + row)
        tl.atomic_add(g_ptr + rr, tl.sum(acc, axis=0))
        if MODE == 1:
            tl.atomic_add(fd_ptr + rr, tl.sum(accd, axis=0))


def _win_args(win):
    return int(win.f0), int(win.f1 - win.f0), int(win.y0), int(win.ny), int(win.x0), int(win.nx)


def scatter(mu, R, P, I, beta, iz0, iy0, ix0, win):
    """mu[voxel] += I_row f + beta_row over the in-window voxels of each reflection's box (as WindowModel.mu)."""
    T, n, Dz = R.shape
    b = P.shape[-1]
    if n == 0:
        return mu
    f0, nf, y0, ny, x0, nx = _win_args(win)
    _scatter_kernel[(n, Dz)](mu, R, P, I, beta if beta is not None else I, iz0, iy0, ix0,
                             n, Dz, b, f0, nf, y0, ny, x0, nx,
                             NT=T, HAS_BETA=beta is not None, BLK=BLOCK)
    return mu


def gather(g, fd, R, P, rows, iz0, iy0, ix0, win, A, M, NS, mode):
    """mode 0: g[rows] += sum f A / M; mode 1: g[rows] += sum f (1 - NS / M), fd[rows] += sum f^2 / M (voxels with
    NS >= 0, inside the window)."""
    T, n, Dz = R.shape
    b = P.shape[-1]
    if n == 0:
        return g, fd
    f0, nf, y0, ny, x0, nx = _win_args(win)
    _gather_kernel[(n, Dz)](g, fd if fd is not None else g, R, P, rows, iz0, iy0, ix0,
                            A if A is not None else M, M, NS, n, Dz, b, f0, nf, y0, ny, x0, nx,
                            NT=T, MODE=mode, BLK=BLOCK)
    return g, fd

"""Scaling, error model and outlier rejection , PyTorch (GPU when available).

Scale factor of observation j (inverse scale g_j, I_obs = g_j * I_h):

    log g_j = A(phi_j) - B(phi_j) s_j^2 / 2 + sum_lm c_lm Y_lm(s1_crystal,j) + M(x_j, y_j)

A, B: Gaussian-smoothed knot values along rotation (DECAY), Y_lm: real spherical harmonics of
the diffracted-beam direction in the crystal frame (ABSORP), M: bilinear detector-surface
modulation (MODULATION). Parameters are found by minimising the Hamilton-Rollett-Sparks target
with the merged intensities projected out (variable projection), using exact autodiff gradients
and L-BFGS. Error model: sigma'^2 = SdFac^2 (sigma^2 + SdB g I + (SdAdd g I)^2).
"""
from __future__ import annotations


import os

import torch

from . import tnp as jnp
from .backend import np_, segment_sum
import numpy as np
from scipy.special import sph_harm_y


def real_sph_harm_basis(v, lmax):
    """Real spherical harmonics (l=1..lmax) for unit vectors v (N,3) -> (N, (lmax+1)^2 - 1)."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    cols = []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m < 0:
                cols.append(np.sqrt(2) * (-1) ** m * Y.imag)
            elif m == 0:
                cols.append(Y.real)
            else:
                cols.append(np.sqrt(2) * (-1) ** m * Y.real)
    return np.stack(cols, 1).astype(np.float32)


def _bilinear_weights(x, y, nx, ny, gx, gy):
    fx = np.clip(x / nx * (gx - 1), 0, gx - 1 - 1e-6)
    fy = np.clip(y / ny * (gy - 1), 0, gy - 1 - 1e-6)
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    idx = np.stack([iy * gx + ix, iy * gx + ix + 1, (iy + 1) * gx + ix, (iy + 1) * gx + ix + 1], 1)
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], 1).astype(np.float32)
    return idx, w


def _sparse_knots(phi, knots, width, k=4):
    """Nearest-k Gaussian knot weights (normalised) as gather indices/weights."""
    d = np.abs(phi[:, None] - knots[None, :])
    idx = np.argsort(d, axis=1)[:, :k]
    w = np.exp(-0.5 * (np.take_along_axis(d, idx, 1) / width) ** 2)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return idx.astype(np.int32), w.astype(np.float32)


def inconsistent_regions(I, sig, uid, x, y, nx, ny, grid=16, min_ratio=0.5, isig=3.0, min_obs=30, block=None):
    """Detector regions whose strong observations are systematically weaker than the other measurements
    of the same reflections (median ratio to the per-reflection median of the *other* regions < min_ratio):
    unmasked shadows (beamstop arm, cryo nozzle) or misplaced detector modules. A detector response
    cannot differ by a factor of two between regions, so such observations are unusable.
    ``block`` (optional, e.g. rotation-range index): regions are judged per block, so rotation-dependent
    shadows (goniometer / kappa hardware entering the beam path for part of the sweep) are removed only where
    they occur. Returns (bad observation mask, list of (ix, iy, n, ratio, block))."""
    gx = np.clip((np.asarray(x) / nx * grid).astype(int), 0, grid - 1)
    gy = np.clip((np.asarray(y) / ny * grid).astype(int), 0, grid - 1)
    blk = np.zeros(np.size(I), np.int64) if block is None else np.asarray(block, np.int64)
    cell = (blk * grid + gy) * grid + gx
    # per-reflection median over observations (robust reference; contamination < 50% tolerated)
    order = np.lexsort((I, uid))
    u_s = uid[order]
    first = np.r_[0, np.flatnonzero(np.diff(u_s)) + 1]
    cnt = np.diff(np.r_[first, u_s.size])
    med = np.full(uid.max() + 1, np.nan)
    mid_lo = order[first + (cnt - 1) // 2]
    mid_hi = order[first + cnt // 2]
    med[u_s[first]] = 0.5 * (I[mid_lo] + I[mid_hi])
    mult = np.bincount(uid, minlength=uid.max() + 1)
    ref = med[uid]
    use = (mult[uid] >= 3) & np.isfinite(ref) & (ref / np.maximum(sig, 1e-6) > isig)
    bad = np.zeros(I.size, bool)
    rows = []
    # relative to the whole rotation block: a block where the crystal is weak overall (moving out of the beam,
    # decay) is the scaling model's job, not a detector-region problem
    blk_ratio = {}
    for bk in np.unique(blk[use]):
        mb = use & (blk == bk)
        blk_ratio[int(bk)] = float(np.median(I[mb] / ref[mb])) if mb.sum() >= min_obs else 1.0
    for c in np.unique(cell[use]):
        m = use & (cell == c)
        if m.sum() < min_obs:
            continue
        r = float(np.median(I[m] / ref[m])) / max(blk_ratio.get(int(c // (grid * grid)), 1.0), 1e-6)
        if r < min_ratio:
            bad |= cell == c
            rows.append((int(c % grid), int((c // grid) % grid), int(m.sum()), r, int(c // (grid * grid))))
    return bad, rows


# relative cost change at which the scale fit stops (scipy default 2.2e-9 ran every fit to maxiter)
GN_TOL = 1e-7           # relative objective change: final fit
GN_TOL_LOOSE = 1e-5     # intermediate fits
GN_CHUNK = 16384        # rows per dense normal-matrix product
GN_REUSE = 3            # Gauss-Newton steps a normal matrix is reused for while its steps are accepted
GAUGE_TAU = 1e-3   # width of the resolution-trend gauge on the detector corrections (log g per A^-2)
# resolution-profile gauge: the mean of the detector corrections' log g in each of GAUGE_BINS equal-count s^2 bins
# held at 0 (width GAUGE_BIN_TAU). Symmetry mates of a reflection lie at the same detector radius, so a radial profile
# of the modulation / per-position absorption is exactly degenerate with the merged intensities' resolution
# dependence: only the prior held it, and a wide prior let it drift (simulated large spots, FM data, prior 0.3: merged
# intensities +-4% non-monotonic in resolution, R vs truth 0.041 vs 0.014 at prior 0.03).
GAUGE_BINS = 10
GAUGE_BIN_TAU = 1e-3


def _csr_pair(rows, cols, vals, shape, dev):
    """CSR matrix A (duplicates summed) and its transpose, float64 on ``dev``."""
    import scipy.sparse as sp
    A = sp.csr_matrix((np.asarray(vals, float), (rows, cols)), shape=shape)
    A.sum_duplicates()
    At = A.T.tocsr()

    def t(M):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)       # "sparse CSR support is in beta"
            return torch.sparse_csr_tensor(torch.as_tensor(M.indptr, dtype=torch.int64),
                                           torch.as_tensor(M.indices, dtype=torch.int64),
                                           torch.as_tensor(M.data, dtype=torch.float64), size=M.shape,
                                           check_invariants=False).to(dev)
    return t(A), t(At)


class _LinMap(torch.autograd.Function):
    """y = A p with the backward product A^T g from the stored transpose."""

    @staticmethod
    def forward(p, A, At):
        return torch.mv(A, p)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.At = inputs[2]

    @staticmethod
    def backward(ctx, g):
        return torch.mv(ctx.At, g.contiguous()), None, None


def _f32(v):
    """0-d float32 tensor (as jnp.float32(v)): rounds the value to float32; float64 tensors keep their dtype."""
    return torch.tensor(np.float32(v), device=jnp.asarray(0.0).device)

class Scaler:
    """Scale model for one or several datasets (``ds`` index per observation):
    each dataset has its own rotation scale, decay B, absorption harmonics and detector modulation;
    all datasets share the merged intensities. Error-model parameters are per dataset."""

    def __init__(self, I, sig, uid, phi, s2, s1_crystal, x, y, nx, ny, corrections=("DECAY", "MODULATION", "ABSORP"),
                 rot_spacing=5.0, b_spacing=20.0, lmax=6, det_grid=None, ds=None, nsig=6.0, snrc=0.0, nbatch=None,
                 dabs_grid=(7, 7), prior_sd=0.1, eps=None, centric=None, batch_error=True, batch_error_width=5.0):
        """nsig: misfit limit (6 sigma; WFAC1 scales it); snrc: minimum I/sigma of observations
        used to determine the correction factors (SNRC); nbatch: number of scale batches per dataset
        (NBATCH, overrides rot_spacing)."""
        self.I = np.asarray(I, float)
        self.sig0 = np.asarray(sig, float)
        self.nsig = float(nsig)
        self.eps = np.ones(np.size(I)) if eps is None else np.asarray(eps, float)
        self.centric = np.zeros(np.size(I), bool) if centric is None else np.asarray(centric, bool)
        self.prior_sd = float(prior_sd)
        self.snrc = float(snrc)
        self.uid = np.asarray(uid)
        self.n_uniq = int(self.uid.max()) + 1
        self.corrections = [c.upper() for c in corrections]
        self.ds = np.zeros(self.I.size, np.int32) if ds is None else np.asarray(ds, np.int32)
        self.nds = int(self.ds.max()) + 1
        phi = np.asarray(phi, float)
        # per-rotation-batch error-model factor:
        # stretches where the crystal leaves the beam / ice / damage are noisier than the global model says
        self.batch_error = bool(batch_error)
        eb, off = np.zeros(self.I.size, np.int64), 0
        for d in range(self.nds):
            sel = self.ds == d
            k = np.floor((phi[sel] - phi[sel].min()) / batch_error_width).astype(np.int64)
            eb[sel] = k + off
            off += int(k.max()) + 1
        self.err_batch, self.batch_f = eb, np.ones(off)
        pk, bk, adj_a, adj_b = [], [], [], []
        self.ia = np.zeros((self.I.size, 4), np.int32)
        self.wa = np.zeros((self.I.size, 4), np.float32)
        self.ib = np.zeros((self.I.size, 4), np.int32)
        self.wb = np.zeros((self.I.size, 4), np.float32)
        oa = ob = 0
        self.knot_ranges = []
        self._knot_def = []
        for d in range(self.nds):
            sel = self.ds == d
            ph = phi[sel]
            prange = max(ph.max() - ph.min(), 1e-3)
            nb_d = nbatch[d] if isinstance(nbatch, (list, tuple, np.ndarray)) else nbatch
            spacing = prange / max(int(nb_d), 1) if nb_d else rot_spacing
            nk = max(int(np.ceil(prange / spacing)) + 1, 2)
            kn = np.linspace(ph.min(), ph.max(), nk)
            nb = max(int(np.ceil(prange / b_spacing)) + 1, 2)
            kb = np.linspace(ph.min(), ph.max(), nb)
            i_, w_ = _sparse_knots(ph, kn, max(spacing, prange / max(nk - 1, 1)), min(4, nk))
            self.ia[sel, :i_.shape[1]] = i_ + oa
            self.wa[sel, :i_.shape[1]] = w_
            i_, w_ = _sparse_knots(ph, kb, max(b_spacing, prange / max(nb - 1, 1)), min(4, nb))
            self.ib[sel, :i_.shape[1]] = i_ + ob
            self.wb[sel, :i_.shape[1]] = w_
            self._knot_def.append((kn, max(spacing, prange / max(nk - 1, 1)), min(4, nk),
                                   kb, max(b_spacing, prange / max(nb - 1, 1)), min(4, nb)))
            adj_a += [(oa + j, oa + j + 1) for j in range(nk - 1)]
            adj_b += [(ob + j, ob + j + 1) for j in range(nb - 1)]
            pk.append(kn)
            bk.append(kb)
            self.knot_ranges.append((oa, oa + nk, ob, ob + nb))
            oa += nk
            ob += nb
        self.phi_knots, self.b_knots = pk, bk
        self.adj_a = np.array(adj_a, np.int32).reshape(-1, 2)
        self.adj_b = np.array(adj_b, np.int32).reshape(-1, 2)
        self.s2 = np.asarray(s2, np.float32)
        self.use_abs = "ABSORP" in self.corrections or "ABSORPTION" in self.corrections
        self.use_mod = "MODULATION" in self.corrections
        self.use_decay = "DECAY" in self.corrections
        self.lmax = lmax if self.use_abs else 0
        self.Y = real_sph_harm_basis(s1_crystal, lmax) if self.use_abs else np.zeros((self.I.size, 0), np.float32)
        if det_grid is None:
            # detector modulation on ~100-pixel bins (MODPIX uses ~65-110 px), 5..32 nodes per axis
            det_grid = (int(np.clip(round(ny / 100), 5, 32)), int(np.clip(round(nx / 100), 5, 32)))
        self.det_grid = det_grid
        self.mod_idx, self.mod_w = _bilinear_weights(x, y, nx, ny, det_grid[1], det_grid[0])
        self.nx, self.ny = nx, ny
        self.na, self.nb_ = oa, ob
        self.nc1 = self.Y.shape[1]
        self.nc = self.nc1 * self.nds
        self.nm1 = det_grid[0] * det_grid[1] if self.use_mod else 0
        self.nm = self.nm1 * self.nds
        self.mod_idx = self.mod_idx + (self.ds[:, None] * self.nm1 if self.use_mod else 0)
        # absorption: smooth function of rotation (scale knots) x detector position (3x3 grid),
        # i.e. of incident *and* diffracted path through the crystal; zero mean over the grid per knot
        self.use_dabs = self.use_abs
        self.dabs_grid = tuple(dabs_grid)
        self.nd1 = self.dabs_grid[0] * self.dabs_grid[1] if self.use_dabs else 0
        self.nd = self.na * self.nd1
        self.dab_idx, self.dab_w = _bilinear_weights(x, y, nx, ny, self.dabs_grid[1], self.dabs_grid[0])
        self.x = np.zeros(self.na + self.nb_ + self.nc + self.nm + self.nd)
        self.active = np.ones(self.I.size, bool)
        self.misfit = np.zeros(self.I.size, bool)   # rejected by the outlier / alien tests (written with negative sigma in legacy-format output)
        self._gauge_row = self._resolution_gauge_row()
        self._gauge_bins = self._resolution_gauge_bins()
        self.sd = (1.0, 0.0, 0.0)
        self.sd_ds = np.tile(np.array([1.0, 0.0, 0.0]), (self.nds, 1))
        self._build()

    def _resolution_gauge_row(self):
        """Row vector c with c.p = least-squares slope of the detector-correction part of log g (modulation +
        per-position absorption) against s^2 over all observations (linear in p)."""
        if not (self.use_mod or self.nd):
            return None
        s2 = self.s2.astype(float)
        dc = s2 - s2.mean()
        wgt = dc / max(float(dc @ dc), 1e-30)
        row = np.zeros(self.na + self.nb_ + self.nc + self.nm + self.nd)
        o = self.na + self.nb_ + self.nc
        if self.use_mod:
            np.add.at(row, o + self.mod_idx.ravel(), (wgt[:, None] * self.mod_w.astype(float)).ravel())
        if self.nd:
            o2 = o + self.nm
            idx = o2 + self.ia[:, :, None] * self.nd1 + self.dab_idx[:, None, :]
            val = wgt[:, None, None] * self.wa.astype(float)[:, :, None] * self.dab_w.astype(float)[:, None, :]
            np.add.at(row, idx.ravel(), val.ravel())
        return row

    def _resolution_gauge_bins(self):
        """(K, n_par) G with G[k] . p = mean over the observations of s^2 bin k (equal counts) of the detector-
        correction part of log g (modulation + per-position absorption), or None (GAUGE_BINS = 0, no detector
        corrections, < 2000 observations)."""
        n = self.s2.size
        K = min(GAUGE_BINS, n // 1000)
        if K < 2 or not (self.use_mod or self.nd):
            return None
        import scipy.sparse as sp
        s2 = self.s2.astype(float)
        k = np.clip(np.searchsorted(np.quantile(s2, np.linspace(0, 1, K + 1)[1:-1]), s2, side="right"), 0, K - 1)
        w = 1.0 / np.bincount(k, minlength=K)[k]
        npar = self.na + self.nb_ + self.nc + self.nm + self.nd
        o = self.na + self.nb_ + self.nc
        rr, cc, vv = [], [], []
        if self.use_mod:
            rr.append(np.repeat(k, self.mod_idx.shape[1])); cc.append((o + self.mod_idx).ravel())
            vv.append((w[:, None] * self.mod_w.astype(float)).ravel())
        if self.nd:
            idx = o + self.nm + self.ia[:, :, None] * self.nd1 + self.dab_idx[:, None, :]
            val = w[:, None, None] * self.wa.astype(float)[:, :, None] * self.dab_w.astype(float)[:, None, :]
            rr.append(np.repeat(k, idx.shape[1] * idx.shape[2])); cc.append(idx.ravel()); vv.append(val.ravel())
        G = sp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(K, npar)).toarray()
        return G

    def _unpack(self, p):
        a = p[:self.na]
        b = p[self.na:self.na + self.nb_]
        c = p[self.na + self.nb_:self.na + self.nb_ + self.nc]
        m = p[self.na + self.nb_ + self.nc:self.na + self.nb_ + self.nc + self.nm]
        return a, b, c, m

    def _build(self):
        ia, wa, ib, wb = map(jnp.asarray, (self.ia, self.wa, self.ib, self.wb))
        adj_a, adj_b = jnp.asarray(self.adj_a), jnp.asarray(self.adj_b)
        uid = jnp.asarray(self.uid)
        nu = self.n_uniq
        use_mod = self.use_mod
        na, nb, nc = self.na, self.nb_, self.nc
        nm, nd, nd1 = self.nm, self.nd, self.nd1
        gauge = None if self._gauge_row is None else jnp.asarray(self._gauge_row)
        gbins = None if self._gauge_bins is None else jnp.asarray(self._gauge_bins)

        # log g is linear in the parameters: log g = A p with A sparse (rotation / B knots, absorption harmonics,
        # detector modulation, per-position absorption). One CSR product forward and one with the precomputed
        # transpose backward; autograd through the gathers instead scatters every observation into a few hundred
        # parameters, which serialises on GPU atomics.
        Rr, Cc, Vv, Ysp = self._dlogg_sparse()
        self._A_loc = (Rr, Cc, Vv)                          # local (sparse) part of d log g / d p, for the GN fit
        npar = na + nb + nc + nm + nd
        dev = jnp.asarray(0.0).device
        H_csr = Ht_csr = None
        if Ysp is not None:
            Yd, off = Ysp
            k_ = np.arange(Yd.shape[1])
            Rh = np.repeat(np.arange(Yd.shape[0]), Yd.shape[1])
            Ch = (off[:, None] + k_[None]).ravel()
            H_csr, Ht_csr = _csr_pair(Rh, Ch, Yd.ravel(), (self.I.size, npar), dev)      # absorption harmonics only
            Rr, Cc, Vv = np.concatenate([Rr, Rh]), np.concatenate([Cc, Ch]), np.concatenate([Vv, Yd.ravel()])
        A_csr, At_csr = _csr_pair(Rr, Cc, Vv, (self.I.size, npar), dev)
        self._A_full = (Rr, Cc, Vv)                         # (for the dense normal-matrix path of the GN fit)
        self._A_t, self._At_t = A_csr, At_csr               # d log g / d p (constant), for the Gauss-Newton fit

        def logg(p):
            return _LinMap.apply(p, A_csr, At_csr)

        def objective(p, I, w, psd):
            g = jnp.exp(logg(p))
            # merged intensities: the exact minimiser of f for the current scales, so df/dIh = 0 and their
            # dependence on p does not enter the gradient (envelope theorem); detached, no backward pass
            gd = g.detach()
            num = segment_sum(w * gd * I, uid, nu)
            den = segment_sum(w * gd * gd, uid, nu)
            Ih = (num / jnp.maximum(den, 1e-30))[uid]
            r = I - g * Ih
            f = jnp.sum(w * r * r)
            a = p[:na]
            b = p[na:na + nb]
            c = p[na + nb:na + nb + nc]
            m = p[na + nb + nc:na + nb + nc + nm]
            nobs = jnp.sum(w > 0)
            # Restraints are Gaussian priors in the same chi^2 units as f (sum of w r^2 ~ number of observations),
            # so their strength does not change with the size of the data set (they used to be added to the
            # per-observation mean, which for 1e5 observations made absorption/modulation priors ~0.5% wide
            # and effectively switched those corrections off).
            #   gauge: mean log scale 0, mean B 0; smoothness: 0.1 per scale knot step, 5 A^2 per B knot step;
            #   absorption harmonics and detector modulation: prior sd 0.1 (log units)
            reg = (jnp.mean(a) / 1e-3) ** 2 + (jnp.mean(b) / 0.1) ** 2
            if adj_a.shape[0]:
                reg = reg + jnp.sum(((a[adj_a[:, 1]] - a[adj_a[:, 0]]) / 0.1) ** 2)
            if adj_b.shape[0]:
                reg = reg + jnp.sum(((b[adj_b[:, 1]] - b[adj_b[:, 0]]) / 5.0) ** 2)
            if nc:
                reg = reg + jnp.sum((c / psd) ** 2)
                # absorption factors beyond x0.6-1.65 (|log| > 0.5) are unphysical even for long-wavelength
                # S-SAD; penalised steeply so that bad detector regions cannot be "absorbed"
                la = _LinMap.apply(p, H_csr, Ht_csr)                    # log absorption per observation
                reg = reg + jnp.sum((jnp.maximum(jnp.abs(la) - 0.5, 0.0) / 0.01) ** 2)
            if use_mod:
                # detector modulation is a few per cent to ~15% (MODPIX ~0.85-1.15): beyond +-30% it is
                # penalised steeply, so a region of bad measurements cannot drag the gauge along
                reg = reg + jnp.sum((m / psd) ** 2) + (jnp.mean(m) / 1e-3) ** 2 \
                    + jnp.sum((jnp.maximum(jnp.abs(m) - 0.3, 0.0) / 0.01) ** 2)
            if gauge is not None:
                # gauge: detector corrections (modulation, per-position absorption) carry no resolution trend; an
                # overall B is exactly degenerate with the merged intensities and belongs to the B knots
                reg = reg + (jnp.dot(gauge, p) / GAUGE_TAU) ** 2
            if gbins is not None:
                reg = reg + jnp.sum((gbins @ p / GAUGE_BIN_TAU) ** 2)
            if nd:
                D = p[na + nb + nc + nm:].reshape(na, nd1)
                reg = reg + jnp.sum((D / psd) ** 2) + jnp.sum((jnp.mean(D, axis=1) / 1e-3) ** 2)
                if adj_a.shape[0]:
                    reg = reg + jnp.sum(((D[adj_a[:, 1]] - D[adj_a[:, 0]]) / (0.5 * psd)) ** 2)
                reg = reg + jnp.sum((jnp.maximum(jnp.abs(D) - 0.4, 0.0) / 0.01) ** 2)
            return (f + reg) / nobs

        self._logg = torch.no_grad()(logg)
        self._obj = torch.no_grad()(objective)

    def scales(self):
        return np.exp(np_(self._logg(jnp.asarray(self.x))))

    def merged(self, exclude_self=False):
        g = self.scales()
        sig = getattr(self, "_sig_cur", self.sig0)
        w = self.active / np.maximum(sig, 1e-6) ** 2
        num = np.bincount(self.uid, w * g * self.I, self.n_uniq)
        den = np.bincount(self.uid, w * g * g, self.n_uniq)
        Ih = num / np.maximum(den, 1e-30)
        if exclude_self:
            numj = num[self.uid] - w * g * self.I
            denj = den[self.uid] - w * g * g
            Ihj = numj / np.maximum(denj, 1e-30)
            return Ih, Ihj, denj
        return Ih, 1.0 / np.sqrt(np.maximum(den, 1e-30))

    def robust_weights(self, c=4.0):
        """Huber weights from leave-one-out normalised deviations (1 within c sigma, c/|d| beyond)."""
        if not np.any(self.x):
            return np.ones(self.I.size)
        g = self.scales()
        Ih, Ihj, denj = self.merged(exclude_self=True)
        sig = getattr(self, "_sig_cur", self.sig0)
        d = (self.I - g * Ihj) / np.sqrt(sig ** 2 + g ** 2 / np.maximum(denj, 1e-30))
        return np.where(np.isfinite(d) & (denj > 0), np.minimum(1.0, c / np.maximum(np.abs(d), 1e-9)), 1.0)

    def fit_scales(self, exclude=None, final=True):
        # intermediate fits only need scales near the minimum; the fit that defines the output is converged tightly
        return self.fit_scales_gn(exclude=exclude, tol=GN_TOL if final else GN_TOL_LOOSE, maxiter=80 if final else 40)

    # ------------------------------------------------------------------ Gauss-Newton (converged) scale fit
    def _dlogg_sparse(self):
        """Sparse d log g / d p for every observation: (rows, cols, vals) of the local parameters (rotation / B
        knots, detector modulation, per-position absorption) and the dense block of absorption harmonics."""
        n = self.I.size
        na, nb, nc, nm = self.na, self.nb_, self.nc, self.nm
        R, C, V = [], [], []
        ar = np.arange(n)
        for k in range(self.ia.shape[1]):
            R.append(ar); C.append(self.ia[:, k]); V.append(self.wa[:, k].astype(float))
        if self.use_decay:
            for k in range(self.ib.shape[1]):
                R.append(ar); C.append(na + self.ib[:, k]); V.append(-0.5 * self.wb[:, k].astype(float) * self.s2)
        if self.use_mod:
            for k in range(self.mod_idx.shape[1]):
                R.append(ar); C.append(na + nb + nc + self.mod_idx[:, k]); V.append(self.mod_w[:, k].astype(float))
        if self.nd:
            o = na + nb + nc + nm
            for k in range(self.ia.shape[1]):
                for l in range(self.dab_idx.shape[1]):
                    R.append(ar); C.append(o + self.ia[:, k] * self.nd1 + self.dab_idx[:, l])
                    V.append(self.wa[:, k].astype(float) * self.dab_w[:, l].astype(float))
        Ysp = None
        if nc:
            Ysp = (self.Y.astype(float), na + nb + self.ds * self.nc1)     # dense harmonics, column offset per obs
        return np.concatenate(R), np.concatenate(C), np.concatenate(V), Ysp

    def _absorption_log(self, c):
        """Per-observation log absorption factor sum_k Y[i, k] c[ds[i], k], as one matrix-vector product per data set
        on a cached float64 copy of Y."""
        if getattr(self, "_Y64", None) is None or self._Y64[0] != self.Y.shape:
            if self.nds == 1:
                self._Y64 = (self.Y.shape, [(None, np.ascontiguousarray(self.Y, dtype=np.float64))])
            else:
                rows = [np.flatnonzero(self.ds == d) for d in range(self.nds)]
                self._Y64 = (self.Y.shape, [(r, np.ascontiguousarray(self.Y[r], dtype=np.float64)) for r in rows])
        C = np.asarray(c, float).reshape(self.nds, self.nc1)
        blocks = self._Y64[1]
        if blocks[0][0] is None:
            return blocks[0][1] @ C[0]
        out = np.empty(self.Y.shape[0])
        for d, (rows, Yd) in enumerate(blocks):
            out[rows] = Yd @ C[d]
        return out

    def _hinge_signature(self, x):
        """Which hinge priors are active at x (hashable)."""
        na, nb, nc, nm = self.na, self.nb_, self.nc, self.nm
        o = na + nb + nc
        parts = []
        if nc:
            la = self._absorption_log(x[na + nb:o])
            parts.append(np.flatnonzero(np.abs(la) > 0.5).tobytes() + np.sign(la[np.abs(la) > 0.5]).tobytes())
        if self.use_mod:
            m = x[o:o + nm]
            parts.append(np.flatnonzero(np.abs(m) > 0.3).tobytes() + np.sign(m[np.abs(m) > 0.3]).tobytes())
        if self.nd:
            D = x[o + nm:o + nm + self.nd]
            parts.append(np.flatnonzero(np.abs(D) > 0.4).tobytes() + np.sign(D[np.abs(D) > 0.4]).tobytes())
        return tuple(parts)

    def _prior_rows(self, x, psd, x_act=None, hinges=True):
        """Prior terms of the objective as sum rho^2 with rho = M x - c (hinge terms linearised at x; which hinges
        are active, and on which side, is decided at x_act, default x): returns (sparse M, rho). hinges=False: only
        the rows that do not depend on the hinge activity (all of them have rho = row . x)."""
        xa = x if x_act is None else x_act
        import scipy.sparse as sp
        na, nb, nc, nm, nd, nd1 = self.na, self.nb_, self.nc, self.nm, self.nd, self.nd1
        rows, cols, vals, rho = [], [], [], []
        r0 = [0]

        def add(cidx, cval, value):
            k = r0[0]
            rows.append(np.full(len(cidx), k)); cols.append(np.asarray(cidx)); vals.append(np.asarray(cval, float))
            rho.append(value); r0[0] += 1

        def add_many(cidx, cval, value):          # one row per element (diagonal-type terms)
            m = len(cidx)
            ks = r0[0] + np.arange(m)
            rows.append(ks); cols.append(np.asarray(cidx)); vals.append(np.asarray(cval, float)); rho.extend(list(value))
            r0[0] += m
        a = x[:na]; b = x[na:na + nb]
        add(np.arange(na), np.full(na, 1.0 / (na * 1e-3)), a.mean() / 1e-3)
        add(na + np.arange(nb), np.full(nb, 1.0 / (nb * 0.1)), b.mean() / 0.1)
        for (adj, off, width, src) in ((self.adj_a, 0, 0.1, a), (self.adj_b, na, 5.0, b)):
            if adj.shape[0]:
                ks = r0[0] + np.arange(adj.shape[0])
                rows += [ks, ks]; cols += [off + adj[:, 1], off + adj[:, 0]]
                vals += [np.full(len(ks), 1.0 / width), np.full(len(ks), -1.0 / width)]
                rho.extend(list((src[adj[:, 1]] - src[adj[:, 0]]) / width)); r0[0] += len(ks)
        if nc:
            c = x[na + nb:na + nb + nc]
            add_many(na + nb + np.arange(nc), np.full(nc, 1.0 / psd), c / psd)
            la = self._absorption_log(c) if hinges else None
            la_a = la if (xa is x or not hinges) else self._absorption_log(xa[na + nb:na + nb + nc])
            act = np.flatnonzero(np.abs(la_a) > 0.5) if hinges else []
            for i in act:
                sg_ = np.sign(la_a[i])
                cidx = na + nb + self.ds[i] * self.nc1 + np.arange(self.nc1)
                add(cidx, sg_ * self.Y[i].astype(float) / 0.01, (sg_ * la[i] - 0.5) / 0.01)
        if self.use_mod:
            o = na + nb + nc
            m = x[o:o + nm]
            add_many(o + np.arange(nm), np.full(nm, 1.0 / psd), m / psd)
            add(o + np.arange(nm), np.full(nm, 1.0 / (nm * 1e-3)), m.mean() / 1e-3)
            ma = xa[o:o + nm]
            act = np.flatnonzero(np.abs(ma) > 0.3) if hinges else np.zeros(0, int)
            if act.size:
                add_many(o + act, np.sign(ma[act]) / 0.01, (np.sign(ma[act]) * m[act] - 0.3) / 0.01)
        if nd:
            o = na + nb + nc + nm
            D = x[o:o + nd].reshape(na, nd1)
            add_many(o + np.arange(nd), np.full(nd, 1.0 / psd), D.ravel() / psd)
            for j in range(na):
                add(o + j * nd1 + np.arange(nd1), np.full(nd1, 1.0 / (nd1 * 1e-3)), D[j].mean() / 1e-3)
            if self.adj_a.shape[0]:
                w_ = 0.5 * psd
                j1 = (self.adj_a[:, 1][:, None] * nd1 + np.arange(nd1)[None]).ravel()
                j0 = (self.adj_a[:, 0][:, None] * nd1 + np.arange(nd1)[None]).ravel()
                ks = r0[0] + np.arange(j1.size)
                rows += [ks, ks]; cols += [o + j1, o + j0]
                vals += [np.full(j1.size, 1.0 / w_), np.full(j1.size, -1.0 / w_)]
                rho.extend(list((D.ravel()[j1] - D.ravel()[j0]) / w_)); r0[0] += j1.size
            Da = xa[o:o + nd]
            act = np.flatnonzero(np.abs(Da) > 0.4) if hinges else np.zeros(0, int)
            if act.size:
                add_many(o + act, np.sign(Da[act]) / 0.01, (np.sign(Da[act]) * D.ravel()[act] - 0.4) / 0.01)
        if self._gauge_row is not None:
            nzr = np.flatnonzero(self._gauge_row)
            add(nzr, self._gauge_row[nzr] / GAUGE_TAU, float(self._gauge_row @ x) / GAUGE_TAU)
        if self._gauge_bins is not None:
            for gk in self._gauge_bins:
                nzr = np.flatnonzero(gk)
                add(nzr, gk[nzr] / GAUGE_BIN_TAU, float(gk @ x) / GAUGE_BIN_TAU)
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0[0], x.size))
        return M, np.asarray(rho, float)

    def _prior_normal(self, x, psd, dev, x_act=None):
        """M^T M (dense, on dev) and M^T rho of the prior rows (_prior_rows) without forming M per call: the rows that
        do not depend on the hinge activity have rho = row. x, so their part is a constant matrix C (cached per
        psd) and C x; the active hinges add diagonal terms (modulation, per-position corrections) and per-data-set
        blocks sum_i Y_i^T Y_i (absorption)."""
        xa = x if x_act is None else x_act
        na, nb, nc, nm, nd = self.na, self.nb_, self.nc, self.nm, self.nd
        key = (float(psd), x.size, None if self._gauge_row is None else float(np.abs(self._gauge_row).sum()),
               None if self._gauge_bins is None else float(np.abs(self._gauge_bins).sum()))
        if getattr(self, "_prior_c", None) is None or self._prior_c[0] != key or self._prior_c[1].device != dev:
            Mc = self._prior_rows(np.zeros(x.size), psd, hinges=False)[0]
            self._prior_c = (key, torch.as_tensor((Mc.T @ Mc).toarray(), dtype=torch.float64, device=dev))
        C = self._prior_c[1]
        MtM = C.clone()
        g = np.zeros(x.size)
        dg = np.zeros(x.size)
        if nc:
            o = na + nb
            la = self._absorption_log(x[o:o + nc])
            la_a = la if xa is x else self._absorption_log(xa[o:o + nc])
            act = np.flatnonzero(np.abs(la_a) > 0.5)
            if act.size:
                sg = np.sign(la_a[act])
                rr = (sg * la[act] - 0.5) / 0.01
                for d in np.unique(self.ds[act]):
                    k = self.ds[act] == d
                    Yd = self.Y[act[k]].astype(float)
                    c0 = o + int(d) * self.nc1
                    MtM[c0:c0 + self.nc1, c0:c0 + self.nc1] += torch.as_tensor(Yd.T @ Yd / 1e-4, device=dev)
                    g[c0:c0 + self.nc1] += Yd.T @ (sg[k] * rr[k]) / 0.01
        for (o, n_, lim, on) in ((na + nb + nc, nm, 0.3, self.use_mod), (na + nb + nc + nm, nd, 0.4, nd > 0)):
            if not on:
                continue
            v, va = x[o:o + n_], xa[o:o + n_]
            act = np.flatnonzero(np.abs(va) > lim)
            if act.size:
                sg = np.sign(va[act])
                dg[o + act] += 1.0 / 0.01 ** 2
                g[o + act] += sg / 0.01 * (sg * v[act] - lim) / 0.01
        MtM.diagonal().add_(torch.as_tensor(dg, device=dev))
        Mtr = torch.mv(C, torch.as_tensor(x, dtype=torch.float64, device=dev)) + torch.as_tensor(g, device=dev)
        return MtM, Mtr

    def _objective_np(self, x, w, psd):
        """The scaling objective (sum w r^2 + priors, NOT divided by the number of observations) via the jitted
        function, for step acceptance."""
        nobs = float(np.sum(w > 0))
        v = self._obj(jnp.asarray(x), jnp.asarray(self.I), jnp.asarray(w), _f32(psd))
        return float(v) * nobs

    def _ell(self, Aloc):
        """Local Jacobian rows as fixed-width index / value arrays (rows have at most a few dozen nonzeros: scale and
        B knots, detector-modulation nodes), observations sorted by unique reflection; cached."""
        c = getattr(self, "_ell_cache", None)
        if c is not None and c[0] is Aloc:
            return c[1]
        perm = np.argsort(self.uid, kind="stable")
        A = Aloc[perm].tocsr()
        nnz = np.diff(A.indptr)
        k = max(int(nnz.max()), 1)
        n = A.shape[0]
        cols = np.zeros((n, k), np.int64)
        vals = np.zeros((n, k))
        pos = np.arange(A.indices.size) - np.repeat(A.indptr[:-1], nnz)
        rr = np.repeat(np.arange(n), nnz)
        cols[rr, pos] = A.indices
        vals[rr, pos] = A.data
        out = dict(perm=perm, cols=cols, vals=vals, k=k, us=self.uid[perm])
        self._ell_cache = (Aloc, out)
        return out

    def _normal_ell(self, fac, a_, aa, uid, nu, Aloc, Yt, rows_ds, n, npar, dev, rows_per_chunk=150000):
        """J^T P J and diag(S^T S) for the variable-projection step (merged intensities eliminated exactly; Kaufman /
        Golub-Pereyra): J^T P J = J^T J - S^T S, P projecting out each unique reflection's direction a = sqrt(w) g,
        with one row of S per unique reflection, S_h = sum_i a_i J_i / sqrt(sum_i a_i^2). Every step is one exact
        Cholesky solve. J^T J is accumulated from per-observation outer products of the (few) local columns, S
        built densely per chunk of unique reflections, S^T S and the harmonic cross terms by dense (multi-threaded)
        products."""
        from .backend import np_ as _np, segment_sum as _ss
        e = self._ell(Aloc)
        perm, cols, vals, us = e["perm"], e["cols"], e["vals"], e["us"]
        facp = _np(fac)[perm]
        V = vals * facp[:, None]                                       # local J rows (sorted by reflection)
        # J^T J: scatter-add of per-observation outer products (torch index_add_ is multi-threaded: 2x np.bincount)
        Adf = torch.zeros(npar * npar, dtype=torch.float64)
        ct, vt = torch.as_tensor(cols), torch.as_tensor(V)
        for r0 in range(0, n, rows_per_chunk):
            c, v = ct[r0:r0 + rows_per_chunk], vt[r0:r0 + rows_per_chunk]
            Adf.index_add_(0, (c[:, :, None] * npar + c[:, None, :]).reshape(-1), (v[:, :, None] * v[:, None, :]).reshape(-1))
        Ad = Adf.reshape(npar, npar).to(dev)
        dS = torch.zeros(npar, dtype=torch.float64, device=dev)
        offs = [self.na + self.nb_ + d * self.nc1 for d in range(len(rows_ds))] if Yt is not None else []
        SY = None
        qp = _np(a_ / torch.sqrt(aa)[uid])[perm]
        qV = V * qp[:, None]
        if Yt is not None:
            FY = fac[:, None] * Yt
            qt = a_ / torch.sqrt(aa)[uid]
            SY = [_ss(qt[rows, None] * FY[rows], uid[rows], nu) for rows in rows_ds]   # (nu, nc1) per set
            crossS = [torch.zeros((npar, self.nc1), dtype=torch.float64, device=dev) for _ in rows_ds]
        # chunks of whole reflections (us sorted): dense S rows, S^T S by GEMM
        bounds = np.searchsorted(us, np.arange(0, nu + 1, max(1, int(4e6 // max(npar, 1)))), "left")
        bounds = np.unique(np.r_[bounds, n])
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            if b1 <= b0:
                continue
            u0 = int(us[b0])
            lu = us[b0:b1] - u0
            nuc = int(lu[-1]) + 1
            # S rows over the columns this chunk touches only: the same products, an ~8x smaller GEMM
            idx = np.unique(cols[b0:b1])
            m = idx.size
            D = np.bincount((lu[:, None] * m + np.searchsorted(idx, cols[b0:b1])).ravel(), qV[b0:b1].ravel(),
                            nuc * m).reshape(nuc, m)
            Dt = torch.as_tensor(D, device=dev)
            it = torch.as_tensor(idx, device=dev)
            Ad[it[:, None], it[None, :]] -= Dt.T @ Dt
            dS[it] += (Dt * Dt).sum(0)
            if SY is not None:
                for d in range(len(rows_ds)):
                    crossS[d][it] += Dt.T @ SY[d][u0:u0 + nuc]
        if Yt is not None:
            Alt = getattr(self, "_gn_cache", (None, None))[1]
            for d, rows in enumerate(rows_ds):
                o = offs[d]
                Fd = FY[rows]
                Ad[o:o + self.nc1, o:o + self.nc1] += Fd.T @ Fd
                Z = torch.zeros((n, self.nc1), dtype=torch.float64, device=dev)
                Z[rows] = fac[rows, None] * Fd
                cross = Alt @ Z - crossS[d]
                dS[o:o + self.nc1] += (SY[d] ** 2).sum(0)
                for d2 in range(len(rows_ds)):
                    Ad[o:o + self.nc1, offs[d2]:offs[d2] + self.nc1] -= SY[d].T @ SY[d2]
                Ad[:, o:o + self.nc1] += cross
                Ad[o:o + self.nc1, :] += cross.T
        return Ad, dS, offs

    def _normal_dense(self, fac, q, dev, rows_per_chunk=None):
        """J^T P J and diag(S^T S) as dense matrix products over chunks of observations sorted by unique reflection
        (J chunk densified on the device; S = per-reflection sums of q J within the chunk). ~1e12 flop per step:
        fast on a GPU, where the host sparse products were the bottleneck."""
        npar = self.x.size
        rows_per_chunk = rows_per_chunk or GN_CHUNK
        if getattr(self, "_dense_cache", None) is None:
            import scipy.sparse as sp
            Rr, Cc, Vv = self._A_full
            A = sp.csr_matrix((Vv, (Rr, Cc)), shape=(self.I.size, npar))
            A.sum_duplicates()
            perm = np.argsort(self.uid, kind="stable")
            us = self.uid[perm]
            chunks, r0 = [], 0
            while r0 < perm.size:
                r1 = min(perm.size, r0 + rows_per_chunk)
                if r1 < perm.size:                       # end the chunk at a unique-reflection boundary
                    r1 = int(np.searchsorted(us, us[r1 - 1], "right"))
                Ac = A[perm[r0:r1]].tocsr()
                lu = np.unique(us[r0:r1], return_inverse=True)[1]
                t = torch.sparse_csr_tensor(torch.as_tensor(Ac.indptr, dtype=torch.int64),
                                            torch.as_tensor(Ac.indices, dtype=torch.int64),
                                            torch.as_tensor(Ac.data, dtype=torch.float64), size=Ac.shape,
                                            check_invariants=False).to(dev)
                chunks.append((r0, r1, t, torch.as_tensor(lu, device=dev), int(lu.max()) + 1))
                r0 = r1
            self._dense_cache = (torch.as_tensor(perm, device=dev), chunks)
        perm, chunks = self._dense_cache
        facp, qp = fac[perm], q[perm]
        Ad = torch.zeros((npar, npar), dtype=torch.float64, device=dev)
        dS = torch.zeros(npar, dtype=torch.float64, device=dev)
        for r0, r1, t, lu, nuc in chunks:
            Jd = t.to_dense() * facp[r0:r1, None]
            Sd = torch.zeros((nuc, npar), dtype=torch.float64, device=dev).index_add_(0, lu, qp[r0:r1, None] * Jd)
            Ad += Jd.T @ Jd - Sd.T @ Sd
            dS += (Sd * Sd).sum(0)
        return Ad, dS

    def fit_scales_gn(self, maxiter=40, exclude=None, tol=1e-7):
        """Levenberg-Marquardt / Gauss-Newton fit of the scale model with the merged intensities profiled out
        (variable projection, Kaufman). The Jacobian is
        diag(fac) A with A = d log g / d p constant and sparse, so every product runs on the compute device with the
        stored CSR pair; the fixed-intensity normal matrix J^T J (+ priors) is formed once per step (sparse product on
        the host), factorised on the device, and preconditions conjugate gradients on the projected system. Steps
        are accepted on the exact objective. Converges in ~20 steps to the MAP (the 200-iteration L-BFGS fit it
        replaced stopped well short of it), so the result does not depend on rounding."""
        import scipy.sparse as sp
        from .backend import np_ as _np, segment_sum as _ss
        sig = getattr(self, "_sig_cur", self.sig0)
        use = self.active & (self.I / np.maximum(sig, 1e-6) >= self.snrc) if self.snrc > 0 else self.active
        if exclude is not None:
            use = use & ~exclude
        w = use * self.robust_weights() / np.maximum(sig, 1e-6) ** 2
        psd = self.prior_sd
        n, npar, nu = self.I.size, self.x.size, self.n_uniq
        A_t, At_t = self._A_t, self._At_t
        dev = A_t.device
        f64 = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=dev)
        if getattr(self, "_gn_cache", None) is None:
            # normal matrix pieces: local columns (rotation / B knots, modulation, per-position absorption; <= ~30
            # per observation) as a sparse product on the host; the dense absorption-harmonics columns (one block per
            # data set) and their cross terms with the local columns as dense / sparse x dense products on the device
            Rr, Cc, Vv = self._A_loc
            Aloc = sp.csr_matrix((Vv, (Rr, Cc)), shape=(n, npar))
            Aloc.sum_duplicates()
            Alt = _csr_pair(Rr, Cc, Vv, (n, npar), dev)[1] if self.nc else None
            Yt = f64(self.Y) if self.nc else None
            rows_ds = [torch.as_tensor(np.flatnonzero(self.ds == d), device=dev) for d in range(self.nds)] if self.nc else []
            self._gn_cache = (Aloc, Alt, Yt, rows_ds)
        Aloc, Alt, Yt, rows_ds = self._gn_cache
        dense_nm = dev.type == "cuda"         # dense normal-matrix GEMMs on a GPU, the ELL accumulation elsewhere
        wt, It = f64(w), f64(self.I)
        uid = torch.as_tensor(self.uid, dtype=torch.int64, device=dev)
        x = self.x.copy()
        F = self._objective_np(x, w, psd)
        lam = 1e-3
        info = dict(nit=0, F0=F, nbuild=0)
        # normal-matrix reuse (chord / modified Gauss-Newton): the gradient is recomputed exactly at every step, the
        # normal matrix (the expensive part: ~6 s per build for 3000 parameters / 10^6 observations) is kept while
        # steps with it keep being accepted at the first trial. Same stationary point; convergence is declared only
        # on a step with a freshly built matrix (full Gauss-Newton precision at the end).
        reuse_max = GN_REUSE
        cached, age, fresh_next = None, 0, True
        for it in range(maxiter * (1 + max(reuse_max, 0))):
            if info["nbuild"] >= maxiter and (fresh_next or age >= reuse_max):
                break
            g = torch.exp(torch.mv(A_t, f64(x)))
            num = _ss(wt * g * It, uid, nu)
            den = _ss(wt * g * g, uid, nu)
            Ih = (num / torch.clamp(den, min=1e-30))[uid]
            sw = torch.sqrt(wt)
            r = sw * (It - g * Ih)
            fac = -sw * g * Ih                                 # d r / d log g
            a_ = sw * g
            aa = torch.clamp(_ss(a_ * a_, uid, nu), min=1e-300)
            fresh = fresh_next or cached is None or age >= reuse_max
            if fresh:
                if dense_nm:
                    Ad, dS = self._normal_dense(fac, a_ / torch.sqrt(aa)[uid], dev)
                else:
                    Ad, dS, _ = self._normal_ell(fac, a_, aa, uid, nu, Aloc, Yt, rows_ds, n, npar, dev)
                cached, age = (Ad, dS), 0
                info["nbuild"] += 1
            else:
                Ad, dS = cached
                age += 1
            fresh_next = False
            gvec0 = torch.mv(At_t, fac * r)         # J^T P r = J^T r (r is orthogonal to a at the merged intensities)
            Ad0 = Ad
            accepted = False
            sig_x = self._hinge_signature(x)
            for trial in range(12):
                if trial == 1 and not fresh and reuse_max > 0:
                    break                          # a stale matrix's step was rejected: rebuild instead of damping
                # hinge priors (absorption / modulation / per-position limits) are quadratic beyond their thresholds:
                # active-set iteration -- model the hinges active at the trial point, re-solve until that set is
                # stable (a hinge crossed by the step is otherwise invisible to the quadratic model: rejected steps,
                # high damping, slow convergence) the prior terms depend on x_act only through its hinge signature: a
                # step that keeps the hinges active at x is final after one solve
                x_act, sig_prev = x, sig_x
                for _k in range(4):
                    MtMd, Mtr = self._prior_normal(x, psd, dev, x_act=x_act)
                    Ad = Ad0 + MtMd
                    gvec = gvec0 + Mtr
                    # LM damping scaled by the fixed-intensity diagonal (positive along the projected-out directions)
                    d_ = torch.clamp(torch.diagonal(Ad) + dS, min=1e-12)
                    L, bad = torch.linalg.cholesky_ex(Ad + lam * torch.diag(d_))
                    if int(bad) != 0:
                        break
                    step = -torch.cholesky_solve(gvec[:, None], L)[:, 0]
                    xn = x + _np(step)
                    sig = self._hinge_signature(xn)
                    if sig_prev is not None and sig == sig_prev:
                        break
                    x_act, sig_prev = xn, sig
                if int(bad) != 0:
                    lam *= 10
                    continue
                Fn = self._objective_np(xn, w, psd)
                if np.isfinite(Fn) and Fn < F:
                    accepted = True
                    break
                lam *= 4
            info["nit"] = it + 1
            if not accepted:
                if not fresh:
                    fresh_next = True
                    continue
                break
            rel = (F - Fn) / max(F, 1e-30)
            x, F = xn, Fn
            lam = max(lam / 3, 1e-9)
            if rel < tol:
                if fresh:
                    break
                fresh_next = True                  # confirm convergence with a full Gauss-Newton step
        self.x = x
        info["F"] = F
        return info

    def fit_error_model(self, nbins=20, min_mult=2, two_param=False, apply=True, robust=True):
        """Fit SdFac, SdB, SdAdd (per dataset) so that normalised deviations from the leave-one-out
        mean have unit (robust) variance in bins of intensity (computed self-consistently with trial sigmas)."""
        from scipy.optimize import least_squares
        g = self.scales()
        self._sig_cur = getattr(self, "_sig_cur", self.sig0)
        Ih = self.merged()[0]
        mult = np.bincount(self.uid, self.active.astype(int), self.n_uniq)
        Ie = np.maximum(g * Ih[self.uid], 0.0)
        Ij = self.I
        uid, nu, act = self.uid, self.n_uniq, self.active
        base_s2 = self._sig_cur ** 2
        fb2 = self.obs_error_factor() ** 2
        out_two = {}
        for d in range(self.nds):
            ok = act & (mult[uid] >= min_mult) & (self.ds == d)
            if ok.sum() < 200:
                continue
            q = np.quantile(Ie[ok], np.linspace(0, 1, nbins + 1))
            bi = np.clip(np.searchsorted(q, Ie) - 1, 0, nbins - 1)
            sel_d = self.ds == d
            # the deviations are only needed for the observations in the fit (ok), grouped by intensity bin: the
            # index sets are fixed during the fit (the residual used to rebuild 20 full-length masks per call)
            oi = np.flatnonzero(ok)
            oi = oi[np.argsort(bi[oi], kind="stable")]
            b_lo = np.searchsorted(bi[oi], np.arange(nbins), "left")
            b_hi = np.searchsorted(bi[oi], np.arange(nbins), "right")
            g_o, Ij_o, act_o, uid_o = g[oi], Ij[oi], act[oi], uid[oi]
            resid_t = self._errmodel_resid_t(base_s2, fb2, Ie, Ij, g, act, uid, nu, sel_d, oi, b_lo, b_hi,
                                             two_param) if robust else None
            # RMS fit (robust=False, the ISa model): the |d| < 6 outlier set is fixed at the start parameters (the
            # residual's first evaluation). Re-cut at every trial point the objective jumped as observations crossed
            # the cut, and the fit landed in different minima from slightly different starts.
            cut6 = [None]

            def resid(pars):
                if resid_t is not None:
                    return resid_t(pars)
                sdfac, sdb, sdadd = pars
                if two_param:
                    sdb = 0.0
                s2 = base_s2.copy()
                s2[sel_d] = fb2[sel_d] * sdfac ** 2 * (self.sig0[sel_d] ** 2 + abs(sdb) * Ie[sel_d] + (sdadd * Ie[sel_d]) ** 2)
                w = act * g * g / s2
                num = np.bincount(uid, act * g * Ij / s2, nu)
                den = np.bincount(uid, w, nu)
                s2_o, w_o = s2[oi], w[oi]
                numj = num[uid_o] - act_o * g_o * Ij_o / s2_o
                denj_ = den[uid_o] - w_o
                okj = denj_ > 0
                Ehj_ = g_o * numj / np.maximum(denj_, 1e-30)
                dd = (Ij_o - Ehj_) / np.sqrt(s2_o + g_o * g_o / np.maximum(denj_, 1e-30))
                okj &= np.isfinite(dd)        # a non-finite sigma / scale must not stop the fit
                out = []
                for i in range(nbins):
                    sl = slice(b_lo[i], b_hi[i])
                    db = dd[sl][okj[sl]]          # this bin's deviations, in observation order
                    if not robust:
                        if cut6[0] is None:
                            cut6[0] = np.abs(np.where(okj, dd, np.inf)) < 6.0
                        db = dd[sl][(okj & cut6[0])[sl]]
                        if db.size > 30:
                            out.append(np.log(np.mean(db ** 2)))
                        continue
                    if db.size > 30:
                        # robust variance (1.4826 MAD)^2: equals <d^2> for Gaussian errors but is not driven
                        # by gross misfits (e.g. a misplaced detector module), which an RMS fit would turn
                        # into an inflated error model and ever larger sigmas
                        out.append(2 * np.log(1.4826 * np.median(np.abs(db)) + 1e-12))
                return np.array(out)

            x0 = np.clip(self.sd_ds[d] if self.sd_ds[d, 0] != 1.0 or self.sd_ds[d, 2] != 0 else [1.0, 0.0, 0.03],
                         [0.31, 0.0, 1e-4], [4.9, 9.9, 0.49])
            r = least_squares(resid, x0=x0, bounds=([0.3, 0.0, 0.0], [5.0, 10.0, 0.5]))
            if two_param:
                r.x[1] = 0.0
            elif r.fun.size > 4:
                # SdB (linear in I) and SdAdd (quadratic) trade off; on many data sets the fit drives SdAdd to 0
                # (ISa undefined) without describing the deviations better. Keep the three-parameter model only
                # when it beats the two-parameter model (SdB = 0) by BIC; otherwise use the two-parameter model.
                r2 = least_squares(lambda q: resid([q[0], 0.0, q[1]]), x0=[x0[0], max(x0[2], 1e-3)],
                                   bounds=([0.3, 0.0], [5.0, 0.5]))
                n_ = r.fun.size
                bic3 = n_ * np.log(max(np.sum(r.fun ** 2), 1e-30) / n_) + 3 * np.log(n_)
                bic2 = n_ * np.log(max(np.sum(r2.fun ** 2), 1e-30) / n_) + 2 * np.log(n_)
                if bic3 > bic2 - 2.0:
                    r.x = np.array([r2.x[0], 0.0, r2.x[1]])
            if not apply:
                out_two[d] = r.x.copy()
                continue
            self.sd_ds[d] = r.x
            base_s2[sel_d] = fb2[sel_d] * r.x[0] ** 2 * (self.sig0[sel_d] ** 2 + r.x[1] * Ie[sel_d] + (r.x[2] * Ie[sel_d]) ** 2)
        if not apply:
            return out_two
        self.sd = tuple(float(v) for v in self.sd_ds[0])
        P = self.sd_ds[self.ds]
        if self.batch_error and two_param is False and robust:
            self._update_batch_factors(g, Ij, base_s2, mult, min_mult)
        P = self.sd_ds[self.ds]
        self._sig_cur = self.obs_error_factor() * P[:, 0] * np.sqrt(self.sig0 ** 2 + P[:, 1] * Ie + (P[:, 2] * Ie) ** 2)
        return self.sd

    def _errmodel_resid_t(self, base_s2, fb2, Ie, Ij, g, act, uid, nu, sel_d, oi, b_lo, b_hi, two_param):
        """fit_error_model's robust residual (log robust variance of the leave-one-out deviations per intensity bin)
        evaluated with torch on the scaling device: the constant products once, the per-reflection sums by
        index_add, the per-bin medians from two stable sorts (NumPy's median: mean of the two middle values) and
        one transfer per call."""
        dev = self._A_t.device
        T = lambda a_: torch.as_tensor(np.ascontiguousarray(a_, dtype=np.float64), device=dev)
        ag = act * g
        agI, agg = T(ag * Ij), T(ag * g)
        base = T(base_s2)
        sel = torch.as_tensor(np.flatnonzero(sel_d), device=dev)
        fb2_d, s0_d, Ie_d = T(fb2[sel_d]), T(self.sig0[sel_d] ** 2), T(Ie[sel_d])
        uid_t = torch.as_tensor(uid, device=dev)
        oi_t = torch.as_tensor(oi, device=dev)
        uid_o, g_o, Ij_o = uid_t[oi_t], T(g[oi]), T(Ij[oi])
        nbins = len(b_lo)
        binpos = torch.as_tensor(np.repeat(np.arange(nbins), b_hi - b_lo), device=dev)
        inf = torch.tensor(float("inf"), dtype=torch.float64, device=dev)

        def resid(pars):
            sdfac, sdb, sdadd = (float(v) for v in pars)
            if two_param:
                sdb = 0.0
            s2 = base.clone()
            s2[sel] = fb2_d * sdfac ** 2 * (s0_d + abs(sdb) * Ie_d + (sdadd * Ie_d) ** 2)
            a1, a2 = agI / s2, agg / s2
            num = torch.zeros(nu, dtype=torch.float64, device=dev).index_add_(0, uid_t, a1)
            den = torch.zeros(nu, dtype=torch.float64, device=dev).index_add_(0, uid_t, a2)
            denj = den[uid_o] - a2[oi_t]
            dj = torch.clamp(denj, min=1e-30)
            Ehj = g_o * (num[uid_o] - a1[oi_t]) / dj
            dd = (Ij_o - Ehj) / torch.sqrt(s2[oi_t] + g_o * g_o / dj)
            ok = (denj > 0) & torch.isfinite(dd)
            v = torch.where(ok, dd.abs(), inf)
            cnt = torch.zeros(nbins, dtype=torch.int64, device=dev).index_add_(0, binpos, ok.long()).cpu().numpy()
            # per-bin median of the finite values: the observations are in bin order and the others are +inf, so
            # the k-th smallest of a bin's segment (k <= its count) is the k-th smallest finite value -- selection
            # per bin instead of two stable sorts of all observations (~460 calls per SCALE)
            med = np.full(nbins, np.inf)
            for b in np.flatnonzero(cnt > 30):
                seg = v[b_lo[b]:b_hi[b]]
                k1, k2 = (int(cnt[b]) - 1) // 2 + 1, int(cnt[b]) // 2 + 1
                x1 = torch.kthvalue(seg, k1).values
                med[b] = float((x1 + (x1 if k2 == k1 else torch.kthvalue(seg, k2).values)) / 2)
            keep = cnt > 30
            return 2 * np.log(1.4826 * med[keep] + 1e-12)
        return resid

    def obs_error_factor(self):
        """Per-observation sigma factor of its rotation batch."""
        return self.batch_f[self.err_batch]

    def _update_batch_factors(self, g, Ij, s2, mult, min_mult, min_obs=30, max_f=3.0, min_f=0.7):
        """Robust scale of leave-one-out normalised deviations per rotation batch (running median over 3 batches),
        relative to the median over the sweep -> multiplicative sigma factor per batch."""
        uid, nu, act = self.uid, self.n_uniq, self.active
        w = act * g * g / s2
        num = np.bincount(uid, act * g * Ij / s2, nu)
        den = np.bincount(uid, w, nu)
        numj = num[uid] - act * g * Ij / s2
        denj = den[uid] - w
        ok = act & (mult[uid] >= min_mult) & (denj > 0)
        dd = (Ij - g * numj / np.maximum(denj, 1e-30)) / np.sqrt(s2 + g * g / np.maximum(denj, 1e-30))
        r = np.full(self.batch_f.size, np.nan)
        for bb in range(r.size):
            sel = ok & (self.err_batch == bb)
            if sel.sum() >= min_obs:
                r[bb] = 1.4826 * np.median(np.abs(dd[sel]))
        for d in range(self.nds):
            bs = np.unique(self.err_batch[self.ds == d])
            v = r[bs]
            if np.isfinite(v).sum() < 5:
                continue
            sm = np.array([np.nanmedian(v[max(i - 1, 0):i + 2]) if np.isfinite(v[max(i - 1, 0):i + 2]).any()
                           else np.nan for i in range(v.size)])
            sm = np.where(np.isfinite(sm), sm, np.nanmedian(sm))
            sm = sm / np.nanmedian(sm)
            f = np.clip(self.batch_f[bs] * sm, min_f, max_f)
            self.batch_f[bs] = f / np.median(f)

    def reject_outliers(self, nsig=None, min_mult=3):
        nsig = self.nsig if nsig is None else nsig
        g = self.scales()
        Ih, Ihj, denj = self.merged(exclude_self=True)
        sig = self._sig_cur
        mult = np.bincount(self.uid, self.active.astype(int), self.n_uniq)
        d = (self.I - g * Ihj) / np.sqrt(sig ** 2 + g ** 2 / np.maximum(denj, 1e-30))
        out = self.active & (mult[self.uid] >= min_mult) & (np.abs(d) > nsig)
        # reject only the worst observation of each reflection per cycle
        worst = np.zeros(self.n_uniq)
        np.maximum.at(worst, self.uid, np.where(out, np.abs(d), 0.0))
        rej = out & (np.abs(d) >= worst[self.uid])
        self.active &= ~rej
        self.misfit |= rej
        return int(rej.sum())

    def select_prior(self, candidates=(0.01, 0.03, 0.1, 0.3), test_fraction=0.1, verbose=True, log=print):
        """Flexible models are needed for real detectors and crystals but overfit clean data."""
        if not (self.nc or self.nm or self.nd):
            return self.prior_sd
        if os.environ.get("XPG_PRIOR_GRID"):
            candidates = tuple(float(v) for v in os.environ["XPG_PRIOR_GRID"].split(","))
        if len(candidates) == 1:                  # a fixed width: no held-out fit
            self.prior_sd = float(candidates[0])
            if verbose:
                log(f"  SCALE: absorption/modulation prior width {self.prior_sd:g} (fixed)")
            return self.prior_sd
        mult = np.bincount(self.uid, minlength=self.n_uniq)
        rng = np.random.default_rng(12345)
        test = (rng.random(self.I.size) < test_fraction) & (mult[self.uid] >= 2) & self.active
        if test.sum() < 500:
            return self.prior_sd
        sig = getattr(self, "_sig_cur", self.sig0)
        x0 = self.x.copy()
        res, sols = [], []
        for psd in candidates:
            self.prior_sd = float(psd)
            # continuation from the previous (narrower) prior's solution: the fits do depend on their start, so the
            # narrow-to-wide order is a homotopy, not just a warm start
            self.x = (sols[-1] if sols else x0).copy()
            self.fit_scales(exclude=test, final=False)
            sols.append(self.x.copy())
            g = self.scales()
            w = (self.active & ~test) / np.maximum(sig, 1e-6) ** 2
            num = np.bincount(self.uid, w * g * self.I, self.n_uniq)
            den = np.bincount(self.uid, w * g * g, self.n_uniq)
            ok = test & (den[self.uid] > 0)
            Ih = num[self.uid] / np.maximum(den[self.uid], 1e-30)
            d = (self.I - g * Ih)[ok] / np.sqrt(sig[ok] ** 2 + g[ok] ** 2 / den[self.uid][ok])
            res.append(float(np.mean(np.minimum(d * d, 36.0))))     # clipped: outliers must not decide
        best = int(np.argmin(res))
        self.prior_sd = float(candidates[best])
        self.x = sols[best].copy()      # warm start of the final (all-data) fit
        if verbose:
            log("  SCALE: absorption/modulation prior width by cross-validation (held-out chi^2): " +
                ", ".join(f"{c:g}: {r:.4f}" for c, r in zip(candidates, res)) + f" -> {self.prior_sd:g}")
        return self.prior_sd

    def reject_aliens(self, k_acentric=20.0, k_centric=40.0, nbins=20, verbose=True, log=print):
        """'alien' rejection: an observation far above the Wilson expectation for its resolution
        (I > k eps <I>; acentric P = e^-20, centric k=40 for a similar probability) that its symmetry mates do
        not support (their mean < I/5, or no mates). Catches ice-crystal spots and zingers on reflections
        measured only once or twice, which the equivalence-based misfit test cannot."""
        g = self.scales()
        Is = self.I / g
        act = self.active
        if act.sum() < 2 * nbins:
            return 0
        q = np.quantile(self.s2[act], np.linspace(0, 1, nbins + 1))
        b = np.clip(np.searchsorted(q, self.s2) - 1, 0, nbins - 1)
        meanI = np.array([np.mean(np.clip((Is / self.eps)[act & (b == i)], 0, None)) if np.any(act & (b == i)) else np.nan
                          for i in range(nbins)])
        expect = self.eps * meanI[b]
        k = np.where(self.centric, k_centric, k_acentric)
        cand = act & np.isfinite(expect) & (Is > k * expect)
        n = np.bincount(self.uid, act.astype(float), self.n_uniq)
        sm = np.bincount(self.uid, np.where(act, Is, 0.0), self.n_uniq)
        others = np.where(n[self.uid] > 1, (sm[self.uid] - Is) / np.maximum(n[self.uid] - 1, 1), -np.inf)
        alien = cand & ((n[self.uid] <= 1) | (others < Is / 5.0))
        self.active &= ~alien
        self.misfit |= alien
        if verbose and alien.any():
            log(f"  SCALE: {int(alien.sum())} 'alien' observations rejected (I > 20 x Wilson expectation, not supported "
                f"by symmetry mates: ice-crystal spots, zingers)")
        return int(alien.sum())

    def run(self, n_cycles=3, verbose=True, log=print, select_prior=True):
        if select_prior:
            self.select_prior(verbose=verbose, log=log)
        for cyc in range(n_cycles):
            self.fit_scales(final=False)
            if cyc == 0:
                self.reject_aliens(verbose=verbose, log=log)
            sd = self.fit_error_model()
            nrej = self.reject_outliers()
            if verbose:
                isa = self.isa(0)
                log(f"  SCALE: cycle {cyc + 1}: SdFac {sd[0]:.3f} SdB {sd[1]:.3f} SdAdd {sd[2]:.4f} "
                    f"({'limit I/sigma %.1f' % isa if isa < 500 else 'no I/sigma plateau'}); rejected {nrej} outliers")
        self.fit_scales()
        return self

    def eval_terms(self, d, phi, s2=None, s1c=None, x=None, y=None):
        """Inverse scale factors g of dataset d split into terms (numpy), for correction images:
        returns dict(scale=, decay=, absorption=, modulation=) evaluated where inputs are given."""
        a, b, c, m = self._unpack(self.x)
        kn, wa_, ka, kb, wb_, kbn = self._knot_def[d]
        a0, a1, b0, b1 = self.knot_ranges[d]
        phi = np.asarray(phi, float)
        out = {}
        i_, w_ = _sparse_knots(phi, kn, wa_, ka)
        out["scale"] = np.exp(np.sum(w_ * a[a0:a1][i_], axis=1))
        if s2 is not None:
            i_, w_ = _sparse_knots(phi, kb, wb_, kbn)
            Bp = np.sum(w_ * b[b0:b1][i_], axis=1)
            out["decay"] = np.exp(-0.5 * Bp * np.asarray(s2, float)) if self.use_decay else np.ones_like(Bp)
        if s1c is not None and self.nc:
            Y = real_sph_harm_basis(np.asarray(s1c, float), self.lmax)
            out["absorption"] = np.exp(Y @ c[d * self.nc1:(d + 1) * self.nc1])
        if x is not None and self.use_mod:
            idx, w = _bilinear_weights(np.asarray(x, float), np.asarray(y, float), self.nx, self.ny,
                                       self.det_grid[1], self.det_grid[0])
            out["modulation"] = np.exp(np.sum(w * m[d * self.nm1:(d + 1) * self.nm1][idx], axis=1))
        return out

    def fix_scale(self, sel, factor=1.0):
        """DATA_RANGE_FIXED_SCALE_FACTOR: shift the (free) overall scale so that the observations
        ``sel`` are multiplied by ``factor`` on output (I_out = I_obs / g, so mean log g = -log factor)."""
        if not np.any(sel):
            return
        la = np.log(self.scales())
        for d in np.unique(self.ds[sel]):
            k0, k1 = self.knot_ranges[d][:2]
            m = sel & (self.ds == d)
            self.x[k0:k1] += -np.log(factor) - la[m].mean()

    def isa_two_param(self, d=0):
        """ISa with the two-parameter model (Diederichs 2010) sigma^2 = a (sigma0^2 + b I^2), ISa = 1/sqrt(ab)
        (the SdB term otherwise absorbs part of the systematic error)."""
        saved = (self.sd_ds.copy(), self.sd, getattr(self, "_sig_cur", None))
        self._sig_cur = self.sig0.copy()
        try:
            fit = self.fit_error_model(two_param=True, apply=False, robust=False)
        finally:
            self.sd_ds, self.sd = saved[0], saved[1]
            if saved[2] is not None:
                self._sig_cur = saved[2]
        if d not in fit:
            return float("nan")
        f, _, a = fit[d]
        return min(1.0 / (f * a), 999.0) if a > 0 else 999.0

    def isa(self, d=0):
        """ISa = 1/(SdFac*SdAdd); reported as 999 when no systematic error term is detectable."""
        f, _, a = self.sd_ds[d]
        return min(1.0 / (f * a), 999.0) if a > 0 else 999.0

    def scaled(self):
        """(I, sigma) of each observation on the common scale, plus active flags."""
        g = self.scales()
        sig = self._sig_cur if hasattr(self, "_sig_cur") else self.sig0
        return self.I / g, sig / g, self.active.copy()

    def model_summary(self):
        a, b, c, m = self._unpack(self.x)
        out = []
        for d, (a0, a1, b0, b1) in enumerate(self.knot_ranges):
            out.append(dict(phi_knots=self.phi_knots[d], scale=np.exp(a[a0:a1]), b_knots=self.b_knots[d], B=b[b0:b1],
                            absorption_coeffs=c[d * self.nc1:(d + 1) * self.nc1] if self.nc else None,
                            modulation=np.exp(m[d * self.nm1:(d + 1) * self.nm1]).reshape(self.det_grid) if self.nm else None))
        return out[0] if self.nds == 1 else out

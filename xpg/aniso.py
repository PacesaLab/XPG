"""Anisotropy analysis, anisotropic diffraction cut-off and Bayesian amplitude estimation.

Follows the published method of Tickle et al.:

* local mean I/sigma(I) around each reflection, weighted by w = exp(-4 (s/r*)^2) within a
  sphere of radius r* (default 0.15 1/A), computed as an FFT convolution on the hkl grid;
* anisotropic cut-off: keep reflections whose local mean I/sigma >= threshold (default 1.2);
  report principal diffraction limits from an ellipsoid fitted to the cut-off surface;
* maximum-likelihood anisotropic Debye-Waller tensor (traceless, Laue-symmetry constrained)
  using the error-aware Wilson likelihood (French & Wilson formalism; closed form for acentric
  reflections) and a free (binned) isotropic intensity profile;
* French & Wilson posterior <F>, sigma(F), <I>, sigma(I) with the *anisotropic* prior, computed
  by quadrature on the GPU. I(+) and I(-) are treated separately, as recommended.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from torch.special import log_ndtr

from . import tnp as jnp
from .backend import np_
from scipy.optimize import minimize

from .geometry import cell_to_B
from .symops import rot_matrices


# ---------------------------------------------------------------------------
# local I/sigma and cut-off surface
# ---------------------------------------------------------------------------

def expand_p1(hkl, values, sg, friedel=True):
    R = rot_matrices(sg)
    H = np.concatenate([hkl @ Rk for Rk in R], 0)
    V = np.concatenate([values] * len(R), 0)
    if friedel:
        H = np.concatenate([H, -H], 0)
        V = np.concatenate([V, V], 0)
    # remove duplicates (special positions)
    from .symops import hkl_key
    _, idx = np.unique(hkl_key(H), return_index=True)
    return H[idx], V[idx]


def local_mean_isig(hkl, isig, cell, sg, radius=0.15):
    """Gaussian-weighted local average of I/sigma (FFT convolution on the hkl grid)."""
    H, V = expand_p1(hkl, isig, sg)
    hmax = np.abs(H).max(0)
    B = cell_to_B(cell)
    # kernel half-width in index units along each axis
    kw = np.ceil(radius / np.linalg.norm(B, axis=0)).astype(int)
    shape = tuple(int(2 * (hmax[i] + kw[i]) + 1) for i in range(3))
    shape = tuple(int(2 ** np.ceil(np.log2(n))) if n > 64 else n for n in shape)
    grid_v = np.zeros(shape, np.float32)
    grid_n = np.zeros(shape, np.float32)
    idx = tuple((H[:, i] % shape[i]) for i in range(3))
    grid_v[idx] = V
    grid_n[idx] = 1.0
    # kernel on the same periodic grid
    ks = [np.arange(-kw[i], kw[i] + 1) for i in range(3)]
    K = np.stack(np.meshgrid(*ks, indexing="ij"), -1).reshape(-1, 3)
    s = np.linalg.norm(K @ B.T, axis=1)
    w = np.exp(-4.0 * (s / radius) ** 2) * (s <= radius)
    grid_k = np.zeros(shape, np.float32)
    grid_k[tuple(K[:, i] % shape[i] for i in range(3))] = w
    Fk = jnp.fft.rfftn(jnp.asarray(grid_k))
    num = jnp.fft.irfftn(jnp.fft.rfftn(jnp.asarray(grid_v)) * Fk, s=shape)
    den = jnp.fft.irfftn(jnp.fft.rfftn(jnp.asarray(grid_n)) * Fk, s=shape)
    num, den = np_(num), np_(den)
    sel = tuple(hkl[:, i] % shape[i] for i in range(3))
    return num[sel] / np.maximum(den[sel], 1e-6)


@dataclass
class AnisoSurface:
    Q: np.ndarray               # ellipsoid matrix (Cartesian reciprocal), s^T Q s = 1 on surface
    limits: np.ndarray          # principal diffraction limits (A), sorted best..worst
    directions: np.ndarray      # principal directions (Cartesian reciprocal, unit)
    directions_hkl: np.ndarray  # same expressed as reciprocal-lattice (a*, b*, c*) combinations


def fit_cutoff_surface(hkl, local, cell, sg, threshold=1.2, ndir=2000):
    """Ellipsoid through the outermost reflections with local I/sigma >= threshold."""
    H, L = expand_p1(hkl, local, sg)
    B = cell_to_B(cell)
    S = H @ B.T
    r = np.linalg.norm(S, axis=1)
    keep = (L >= threshold) & (r > 0)
    if keep.sum() < 50:
        keep = r > 0
    u = S[keep] / r[keep, None]
    # directions on a Fibonacci sphere
    k = np.arange(ndir) + 0.5
    th = np.arccos(1 - 2 * k / ndir)
    ph = np.pi * (1 + 5 ** 0.5) * k
    D = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    from scipy.spatial import cKDTree
    tree = cKDTree(u)
    pts = []
    for i, d in enumerate(D):
        nb = tree.query_ball_point(d, 2 * np.sin(np.radians(4.0)))
        if len(nb) < 3:
            continue
        rr = r[keep][nb]
        pts.append(d * np.quantile(rr, 0.98))
    P = np.array(pts)
    # least squares for symmetric Q: p^T Q p = 1 (6 unknowns)
    A = np.c_[P[:, 0] ** 2, P[:, 1] ** 2, P[:, 2] ** 2, 2 * P[:, 0] * P[:, 1], 2 * P[:, 0] * P[:, 2], 2 * P[:, 1] * P[:, 2]]
    q, *_ = np.linalg.lstsq(A, np.ones(len(P)), rcond=None)
    Q = np.array([[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]])
    Q = _symmetrize_tensor(Q, cell, sg)
    ev, evec = np.linalg.eigh(Q)
    # a limit beyond the best measured resolution is not determined by the data: bounded there
    ev = np.maximum(ev, (1.0 / r.max()) ** 2)
    Q = (evec * ev) @ evec.T
    limits = np.sqrt(ev)                # d = 1/|s| at the surface along principal directions
    order = np.argsort(limits)
    dirs = evec[:, order].T
    # express as combinations of reciprocal axes (columns of B)
    dh = np.linalg.solve(B, dirs.T).T
    dh = dh / np.abs(dh).max(1, keepdims=True)
    return AnisoSurface(Q, limits[order], dirs, dh)


# ---------------------------------------------------------------------------
# symmetry-constrained symmetric tensors
# ---------------------------------------------------------------------------

def _cart_ops(cell, sg):
    """Point-group rotations in the Cartesian reciprocal frame (s = B h; h' = h R)."""
    B = cell_to_B(cell)
    Bi = np.linalg.inv(B)
    return [B @ R.T @ Bi for R in rot_matrices(sg)]


def _symmetrize_tensor(T, cell, sg):
    ops = _cart_ops(cell, sg)
    return sum(O @ T @ O.T for O in ops) / len(ops)


def traceless_basis(cell, sg):
    """Orthonormal basis of traceless symmetric tensors invariant under the point group."""
    E = []
    for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        M = np.zeros((3, 3))
        M[i, j] = M[j, i] = 1.0
        E.append(_symmetrize_tensor(M, cell, sg))
    V = np.array([m.ravel() for m in E])
    # remove isotropic component
    iso = np.eye(3).ravel() / np.sqrt(3)
    V = V - np.outer(V @ iso, iso)
    u, s, vt = np.linalg.svd(V, full_matrices=False)
    rank = int((s > 1e-8 * max(s.max(), 1e-30)).sum())
    return [vt[k].reshape(3, 3) for k in range(rank)]


# ---------------------------------------------------------------------------
# ML anisotropy
# ---------------------------------------------------------------------------

def _ex_gauss_logpdf(I, sig, Sigma):
    """log of int_0^inf (1/Sigma) e^{-J/Sigma} N(I; J, sig^2) dJ (acentric Wilson x Gaussian error)."""
    return -jnp.log(Sigma) - I / Sigma + sig ** 2 / (2 * Sigma ** 2) + log_ndtr((I - sig ** 2 / Sigma) / sig)


@dataclass
class AnisoModel:
    B_aniso: np.ndarray        # traceless anisotropic B (Cartesian reciprocal frame), A^2
    s2_knots: np.ndarray
    logk: np.ndarray           # isotropic log-intensity profile at knots
    eigen_B: np.ndarray
    eigen_dirs: np.ndarray

    def sigma(self, hkl, cell, eps):
        S = hkl @ cell_to_B(cell).T
        s2 = np.sum(S * S, 1)
        iso = np.interp(s2, self.s2_knots, self.logk)
        an = -0.5 * np.einsum("ni,ij,nj->n", S, self.B_aniso, S)
        return eps * np.exp(iso + an)


def fit_anisotropy(hkl, I, sig, centric, eps, cell, sg, nknots=20, verbose=True, log=print):
    B = cell_to_B(cell)
    S = hkl @ B.T
    s2 = np.sum(S * S, 1)
    basis = traceless_basis(cell, sg)
    knots = np.linspace(s2.min(), s2.max(), nknots)
    # initial isotropic profile from bin means
    bi = np.clip(np.searchsorted(knots, s2) - 1, 0, nknots - 2)
    t = (s2 - knots[bi]) / (knots[bi + 1] - knots[bi])
    mean_bins = np.array([np.mean(I[(np.abs(s2 - k)).argsort()[:200]] / eps[(np.abs(s2 - k)).argsort()[:200]])
                          for k in knots])
    logk0 = np.log(np.maximum(mean_bins, 1e-3 * np.abs(mean_bins).max() + 1e-6))
    use = ~centric & (sig > 0)
    Sj, Ij, sj, ej = (jnp.asarray(a[use]) for a in (S, I, sig, eps))
    bij, tj = jnp.asarray(bi[use]), jnp.asarray(t[use])
    basis_j = jnp.asarray(np.array(basis)) if basis else jnp.zeros((0, 3, 3))
    nb = len(basis)

    def nll(p):
        logk = p[:nknots]
        bcoef = p[nknots:]
        iso = logk[bij] * (1 - tj) + logk[bij + 1] * tj
        Bt = jnp.einsum("k,kij->ij", bcoef, basis_j) if nb else jnp.zeros((3, 3))
        an = -0.5 * jnp.einsum("ni,ij,nj->n", Sj, Bt, Sj)
        Sigma = ej * jnp.exp(iso + an)
        return -jnp.mean(_ex_gauss_logpdf(Ij, sj, Sigma)) + 1e-4 * jnp.sum(jnp.diff(logk, 2) ** 2)

    def vg(p):
        p = jnp.asarray(p).requires_grad_(True)
        v = nll(p)
        (gr,) = torch.autograd.grad(v, p)
        return float(v.detach()), np_(gr).astype(float)

    p0 = np.concatenate([logk0, np.zeros(nb)])
    res = minimize(vg, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    logk = res.x[:nknots]
    bcoef = res.x[nknots:]
    Bt = sum(c * b for c, b in zip(bcoef, basis)) if nb else np.zeros((3, 3))
    ev, evec = np.linalg.eigh(Bt)
    m = AnisoModel(Bt, knots, logk, ev, evec.T)
    if verbose:
        log(f"  ANISO: anisotropic B eigenvalues (traceless) {np.array2string(ev, precision=2)} A^2; "
            f"Delta B = {ev.max() - ev.min():.2f} A^2 ({nb} free tensor components)")
    return m


# ---------------------------------------------------------------------------
# French & Wilson with anisotropic prior
# ---------------------------------------------------------------------------

@torch.no_grad()
def _fw_moments(I, sig, Sigma, centric):
    n = 384
    Imax = jnp.maximum(I, 0.0) + 8.0 * sig
    Imin = jnp.maximum(I - 8.0 * sig, 0.0)
    Fmax = jnp.sqrt(jnp.maximum(Imax, 1e-12))
    Fmin = jnp.sqrt(Imin)
    # include the prior-dominated region near 0 for weak data
    Fmin = jnp.where(I < 4 * sig, 0.0, Fmin)
    u = (jnp.arange(n) + 0.5) / n
    F = Fmin[:, None] + (Fmax - Fmin)[:, None] * u[None]
    J = F * F
    # log prior density in F (acentric: 2F/S e^{-F^2/S}; centric: sqrt(2/(pi S)) e^{-F^2/(2S)})
    lp_a = jnp.log(2 * F / Sigma[:, None] + 1e-300) - J / Sigma[:, None]
    lp_c = 0.5 * jnp.log(2.0 / (jnp.pi * Sigma[:, None])) - J / (2 * Sigma[:, None])
    lp = jnp.where(centric[:, None], lp_c, lp_a)
    ll = -0.5 * ((I[:, None] - J) / sig[:, None]) ** 2
    lw = lp + ll
    lw = lw - jnp.max(lw, axis=1, keepdims=True)
    w = jnp.exp(lw)
    Z = jnp.sum(w, axis=1)
    EF = jnp.sum(w * F, axis=1) / Z
    EJ = jnp.sum(w * J, axis=1) / Z
    # central second moments (not E[x^2] - E[x]^2, which cancels catastrophically in float32 for strong
    # reflections: ~1e-3 relative noise in sigma(F) at I/sigma ~ 15)
    sF = jnp.sqrt(jnp.sum(w * (F - EF[:, None]) ** 2, axis=1) / Z)
    sJ = jnp.sqrt(jnp.sum(w * (J - EJ[:, None]) ** 2, axis=1) / Z)
    return EF, sF, EJ, sJ


def french_wilson(I, sig, Sigma, centric, chunk=65536):
    out = [np.zeros(len(I)) for _ in range(4)]
    ok = np.isfinite(I) & np.isfinite(sig) & (sig > 0) & np.isfinite(Sigma) & (Sigma > 0)
    idx = np.flatnonzero(ok)
    for c0 in range(0, idx.size, chunk):
        s = idx[c0:c0 + chunk]
        pad = (-s.size) % 1024
        sp = np.concatenate([s, np.repeat(s[-1:], pad)]) if s.size else s
        r = _fw_moments(jnp.asarray(I[sp], jnp.float32), jnp.asarray(sig[sp], jnp.float32),
                        jnp.asarray(Sigma[sp], jnp.float32), jnp.asarray(centric[sp]))
        for k in range(4):
            out[k][s] = np_(r[k])[:s.size]
    for k in range(4):
        out[k][~ok] = np.nan
    return tuple(out)


def run_anisotropic(merged, cell, sg, threshold=1.2, radius=0.15, verbose=True, log=print):
    """Full anisotropy workflow on merged data (dict from merging.merging_statistics)."""
    hkl = merged["hkl"]
    I, sig = merged["I"], merged["sig"]
    isig = np.where(sig > 0, I / sig, 0.0)
    local = local_mean_isig(hkl, isig, cell, sg, radius)
    surf = fit_cutoff_surface(hkl, local, cell, sg, threshold)
    inside = local >= threshold
    # tensor fitted to the reflections inside the cut-off surface, as in the published method; fitting all data within
    # the best limit gave a ~15% smaller anisotropy (the model's Gaussian fall-off is only an approximation over a
    # wide resolution range)
    fit_sel = inside
    model = fit_anisotropy(hkl[fit_sel], I[fit_sel], sig[fit_sel], merged["centric"][fit_sel],
                           merged["eps"][fit_sel], cell, sg, verbose=verbose, log=log)
    Sigma = model.sigma(hkl, cell, merged["eps"])
    F, sF, Ip, sIp = french_wilson(I, sig, Sigma, merged["centric"])
    res = dict(local_isig=local, inside=inside, F=F, SIGF=sF, Ipost=Ip, SIGIpost=sIp, Sigma=Sigma)
    for tag in ("plus", "minus"):
        Ia, sa = merged[f"I{tag}"], merged[f"sigI{tag}"]
        Fa, sFa, _, _ = french_wilson(Ia, sa, Sigma, merged["centric"])
        res[f"F{tag}"], res[f"SIGF{tag}"] = Fa, sFa
    res["DANO"] = res["Fplus"] - res["Fminus"]
    res["SIGDANO"] = np.sqrt(res["SIGFplus"] ** 2 + res["SIGFminus"] ** 2)
    # anisotropy-corrected amplitudes (Debye-Waller correction of the traceless part)
    S = hkl @ cell_to_B(cell).T
    corr = np.exp(0.25 * np.einsum("ni,ij,nj->n", S, model.B_aniso, S))
    res["F_aniso"] = F * corr
    res["SIGF_aniso"] = sF * corr
    if verbose:
        lim = surf.limits
        log(f"  ANISO: diffraction limits along principal axes: {lim[0]:.2f} {lim[1]:.2f} {lim[2]:.2f} A "
            f"(local <I/sigI> >= {threshold}); {inside.sum()}/{len(hkl)} reflections inside the cut-off surface")
        for d, v in zip(lim, surf.directions_hkl):
            log(f"     {d:6.2f} A along {v[0]:+.3f} a* {v[1]:+.3f} b* {v[2]:+.3f} c*")
    return res, surf, model


def ellipsoidal_completeness(hkl_measured, cell, sg, limits, directions_hkl):
    """Completeness inside the anisotropic cut-off surface, approximated by the ellipsoid with the diffraction limits
    `limits` (A) along `directions_hkl` (reciprocal-lattice directions): sum_i ((s . e_i) d_i)^2 <= 1 with s the
    reciprocal vector. Returns (measured unique inside, possible unique inside, completeness) -- the
    'ellipsoidal completeness'. Possible reflections: all symmetry-unique indices of the space group to
    the best limit."""
    import gemmi
    lim = np.asarray(limits, float)
    if lim.size == 0 or not np.all(np.isfinite(lim)):
        return None
    uc = gemmi.UnitCell(*cell) if not isinstance(cell, gemmi.UnitCell) else cell
    Bm = np.array(uc.frac.mat, float).T                       # s (cartesian, 1/A) = B h
    hm_ = np.asarray(hkl_measured, float)
    if len(hm_):                       # limits beyond the measured resolution: bounded there (see fit_cutoff_surface)
        lim = np.maximum(lim, 1.0 / np.linalg.norm(hm_ @ Bm.T, axis=1).max())
    E = np.array([Bm @ np.asarray(d, float) for d in directions_hkl])
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    def inside(h):
        s = np.asarray(h, float) @ Bm.T
        return np.sum(((s @ E.T) * lim[None]) ** 2, axis=1) <= 1.0
    sgx = sg if isinstance(sg, gemmi.SpaceGroup) else gemmi.SpaceGroup(sg)
    allh = np.asarray(gemmi.make_miller_array(uc, sgx, float(lim.min())), int)
    n_pos = int(inside(allh).sum())
    hm = np.asarray(hkl_measured, int)
    asu = gemmi.ReciprocalAsu(sgx)
    ops = sgx.operations()
    red = np.array([asu.to_asu(list(h), ops)[0] for h in hm], int) if len(hm) else np.zeros((0, 3), int)
    red = np.unique(red, axis=0) if len(red) else red
    n_meas = int(inside(red).sum()) if len(red) else 0
    return n_meas, n_pos, (n_meas / n_pos if n_pos else np.nan)

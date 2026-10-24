"""Bayesian merging (Careless / Abismal style, on XPG's SCALE scales): posterior of every unique reflection's true
intensity J >= 0 given its scaled observations, with a robust Student-t likelihood (outliers down-weighted instead of
rejected by a threshold) and a Wilson prior (acentric: exponential, centric: chi^2 with 1 degree of freedom, mean
eps * Sigma(s)) whose shell means Sigma(s) are re-estimated from the posteriors (empirical Bayes). Posterior moments
of J and F = sqrt(J) by quadrature on a grid in F, batched on the compute device. This replaces outlier rejection +
French-Wilson by one model; the scales stay those of SCALE.

    merged = bayes_merge(uid, I, sig, centric, eps, s2, nu=4.0)
"""
from __future__ import annotations

import numpy as np
import torch

from .backend import device


def _shell_index(s2, nbins):
    edges = np.quantile(s2, np.linspace(0, 1, nbins + 1))
    return np.clip(np.searchsorted(edges, s2, side="right") - 1, 0, nbins - 1)


def bayes_merge(uid, I, sig, centric, eps, s2, nu=4.0, nbins=20, n_grid=96, n_iter=3, chunk=20000, dev=None):
    """uid (n_obs,) unique index 0..n_u-1 of each observation; I, sig (n_obs,) scaled intensities; centric, eps, s2
    (n_u,) per unique reflection (s2 = 1/d^2). Returns dict of (n_u,) arrays: I, sigI, F, sigF, n_obs, and the shell
    means Sigma per unique."""
    dev = dev or device()
    uid = np.asarray(uid, np.int64)
    I = np.asarray(I, float)
    sig = np.asarray(sig, float)
    ok = np.isfinite(I) & np.isfinite(sig) & (sig > 0)
    uid, I, sig = uid[ok], I[ok], sig[ok]
    nu_ = len(s2)
    order = np.argsort(uid, kind="stable")
    uid, I, sig = uid[order], I[order], sig[order]
    nobs = np.bincount(uid, minlength=nu_)
    shell = _shell_index(np.asarray(s2, float), nbins)
    centric = np.asarray(centric, bool)
    eps = np.asarray(eps, float)
    # starting shell means: weighted mean scaled intensity per shell (>= a small positive floor)
    wmean = np.bincount(uid, I / sig ** 2, nu_) / np.maximum(np.bincount(uid, 1 / sig ** 2, nu_), 1e-30)
    Sigma_sh = np.array([max(np.mean(wmean[shell == k] / eps[shell == k]) if np.any(shell == k) else 1.0, 1e-6)
                         for k in range(nbins)])
    out = None
    for it in range(n_iter):
        Sigma = Sigma_sh[shell] * eps
        out = _posterior(uid, I, sig, nobs, centric, Sigma, nu, n_grid, chunk, dev)
        # empirical Bayes: shell mean of the posterior E[J] / eps
        Sigma_sh = np.array([max(np.mean(out["I"][shell == k] / eps[shell == k]) if np.any(shell == k) else 1.0, 1e-6)
                             for k in range(nbins)])
    out["Sigma"] = Sigma_sh[shell] * eps
    out["n_obs"] = nobs
    return out


def _posterior(uid, I, sig, nobs, centric, Sigma, nu, n_grid, chunk, dev):
    """Posterior moments per unique reflection on an F grid (0 .. F_max), observations of chunks of reflections at a
    time (segment sums of the Student-t log-likelihood)."""
    nu_ = len(Sigma)
    starts = np.r_[0, np.cumsum(nobs)]
    outI, outsI, outF, outsF = (np.zeros(nu_) for _ in range(4))
    f64 = torch.float64
    t = lambda a: torch.as_tensor(np.asarray(a), device=dev)
    lg = float(torch.lgamma(torch.tensor((nu + 1) / 2.0)) - torch.lgamma(torch.tensor(nu / 2.0)))
    for u0 in range(0, nu_, chunk):
        u1 = min(u0 + chunk, nu_)
        o0, o1 = starts[u0], starts[u1]
        Su = t(Sigma[u0:u1]).to(f64)
        # grid in F per reflection: up to max(5 sqrt(Sigma), sqrt(max(I) + 5 sig)) of its observations
        Iu, su = t(I[o0:o1]).to(f64), t(sig[o0:o1]).to(f64)
        ru = t(uid[o0:o1] - u0)
        top = torch.zeros(u1 - u0, dtype=f64, device=dev).scatter_reduce_(0, ru, Iu + 5 * su, reduce="amax",
                                                                          include_self=True)
        Fmax = torch.sqrt(torch.clamp(torch.maximum(top, 25.0 * Su), min=1e-12))
        # strong reflections: the posterior is far narrower than a 0..Fmax grid (I 1e5 +- 300: width 0.5 in F vs a
        # spacing of 3): grid on the likelihood peak (weighted mean +- 8 of its sd); weak ones from 0
        wsum = torch.zeros(u1 - u0, dtype=f64, device=dev).index_add_(0, ru, 1.0 / su ** 2)
        m = torch.zeros(u1 - u0, dtype=f64, device=dev).index_add_(0, ru, Iu / su ** 2) / torch.clamp(wsum, min=1e-30)
        sm = 1.0 / torch.sqrt(torch.clamp(wsum, min=1e-30))
        strong = (m > 5 * sm) & (wsum > 0)
        lo = torch.where(strong, torch.sqrt(torch.clamp(m - 8 * sm, min=0)), torch.zeros_like(m))
        hi = torch.where(strong, torch.sqrt(m + 8 * sm), Fmax)
        x = (torch.arange(n_grid, dtype=f64, device=dev) + 0.5) / n_grid           # midpoints in (0, 1)
        Fg = lo[:, None] + (hi - lo)[:, None] * x[None]                            # (nu, K)
        Jg = Fg ** 2
        # Wilson prior on J, written as a density in F (dJ = 2F dF): acentric p(J) = exp(-J/S)/S;
        # centric p(J) = exp(-J/2S) / sqrt(2 pi S J)
        cen = t(centric[u0:u1])
        lp_a = -Jg / Su[:, None] - torch.log(Su)[:, None] + torch.log(2 * Fg)
        lp_c = -Jg / (2 * Su[:, None]) - 0.5 * torch.log(2 * np.pi * Su[:, None] * Jg) + torch.log(2 * Fg)
        logp = torch.where(cen[:, None], lp_c, lp_a)
        # Student-t log-likelihood of every observation at every grid point, summed per reflection
        z = (Iu[:, None] - Jg[ru]) / su[:, None]
        ll = lg - 0.5 * np.log(nu * np.pi) - torch.log(su)[:, None] - (nu + 1) / 2 * torch.log1p(z * z / nu)
        logp = logp.index_add(0, ru, ll)
        w = torch.softmax(logp, dim=1)
        EJ = (w * Jg).sum(1)
        EJ2 = (w * Jg * Jg).sum(1)
        EF = (w * Fg).sum(1)
        EF2 = (w * Jg).sum(1)
        outI[u0:u1] = EJ.cpu().numpy()
        outsI[u0:u1] = torch.sqrt(torch.clamp(EJ2 - EJ ** 2, min=0)).cpu().numpy()
        outF[u0:u1] = EF.cpu().numpy()
        outsF[u0:u1] = torch.sqrt(torch.clamp(EF2 - EF ** 2, min=0)).cpu().numpy()
    return dict(I=outI, sigI=outsI, F=outF, sigF=outsF)

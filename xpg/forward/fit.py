"""Forward model fit: joint Poisson fit of the intensities of all reflections of a window and of a smooth
background, with geometry and widths fixed at XPG's values.

    mu_v = B_v + sum_h I_h f_h(v),   B_v = b_k * T(p) * exp(S(p))

v runs over all trusted pixels of the window's frames (dense), k is the frame of v, T XPG's static background
template, S a smooth correction on a bilinear grid over the detector (shared by the frames of the window), b_k a
per-frame level. The intensities of the reflections centred in the window (owned) are free; reflections centred in
its margins enter mu at their current values (block Gauss-Seidel over consecutive windows).

Intensities: Newton (Fisher scoring) on the Poisson log-likelihood with the exact Fisher matrix-vector product
H v = sum_vox f_h (sum_h' f_h' v_h') / mu (one scatter + one gather pass), solved by Jacobi-preconditioned conjugate
gradients; overlapping reflections are coupled exactly. Unconstrained (weak reflections may come out negative);
mu is kept >= 1e-3 B. sigma(I) from the Fisher diagonal (the neighbour coupling is dropped, as the shoebox
integrator's model variances do). Background: Newton on (log b_k, S) with all trusted pixels."""
from __future__ import annotations

import time

import numpy as np
import torch


class WindowFit:
    def __init__(self, wm, counts, tmpl, n_frames, grid=48, owned=None, log=print, min_valid=0.75, damping=1e-2,
                 chunk_voxels=None, min_valid_overload=0.2, newton_tol=0.01, cg_tol=0.003, bg_block_frames=None):
        """wm: model.WindowModel; counts: (n_frames * ny * nx,) float32 tensor (< 0 untrusted); tmpl: (ny, nx)."""
        self.wm, self.dev = wm, wm.dev
        self.nf, self.ny, self.nx = n_frames, wm.win.ny, wm.win.nx
        # counts store: raw photon counts are integers, kept as int16 with -1 = untrusted (2 B / voxel instead of
        # float32 counts + a bool mask); otherwise float32 (< 0 untrusted). valid / n0 are views built on use; the
        # hot paths read the store per frame chunk or at the gathered voxels only.
        npx = self.ny * self.nx
        # frames per chunk of the background / likelihood sums: bounded temporaries (a 100-frame window of a 16M
        # detector is 1.8e9 voxels, 7 GB per float array); default 1/8 of the window, 1e7-2e8 voxels
        if chunk_voxels is None:
            chunk_voxels = float(np.clip(n_frames * npx / 8.0, 1e7, 2e8))
        self.fchunk = max(1, int(chunk_voxels // max(npx, 1)))
        # conversion by frame chunks too (no full-window temporaries)
        ints = True
        if counts.dtype != torch.int16:
            for k0, k1, sl in self.chunks():
                c = counts[sl]
                if float(c.max()) >= 32767 or not bool(torch.all(c == torch.round(c))):
                    ints = False
                    break
        if ints and counts.dtype == torch.int16 and counts.device == torch.device(self.dev):
            self.ns = counts                              # already the store, on the device (driver.tile_counts)
        elif ints and counts.dtype == torch.int16:        # already the store's format (-1 = untrusted)
            self.ns = torch.empty(n_frames * npx, dtype=torch.int16, device=self.dev)
            for k0, k1, sl in self.chunks():
                self.ns[sl] = counts[sl].to(self.dev)
        elif ints:
            self.ns = torch.empty(n_frames * npx, dtype=torch.int16, device=self.dev)
            for k0, k1, sl in self.chunks():
                c = counts[sl].to(self.dev)
                # -2 = overloaded (kept apart from -1 = masked / gaps: overload recovery below), other negatives -1
                self.ns[sl] = torch.where(c >= 0, c, torch.where(c == -2, c, torch.full((), -1.0, device=self.dev))
                                          ).to(torch.int16)
        else:
            self.ns = counts.to(self.dev, copy=True)
        if tmpl is not None and np.shape(tmpl) != (self.ny, self.nx):      # detector tile: the template's slice
            w_ = wm.win
            tmpl = np.asarray(tmpl)[w_.y0:w_.y0 + self.ny, w_.x0:w_.x0 + self.nx]
        T = torch.as_tensor(np.maximum(np.asarray(tmpl, np.float32), 1e-3) if tmpl is not None else
                            np.ones((self.ny, self.nx), np.float32), device=self.dev)
        self.logT = torch.log(T).reshape(-1)
        # bilinear basis of the smooth correction: node weights of each pixel (4 nodes)
        G = grid
        yy, xx = torch.meshgrid(torch.arange(self.ny, device=self.dev, dtype=torch.float32),
                                torch.arange(self.nx, device=self.dev, dtype=torch.float32), indexing="ij")
        fy, fx = yy / max(self.ny - 1, 1) * (G - 1), xx / max(self.nx - 1, 1) * (G - 1)
        iy, ix = torch.clamp(fy.floor().long(), max=G - 2), torch.clamp(fx.floor().long(), max=G - 2)
        ty, tx = fy - iy, fx - ix
        self.G = G
        self.nodes = torch.stack([iy * G + ix, iy * G + ix + 1, (iy + 1) * G + ix, (iy + 1) * G + ix + 1], -1).reshape(-1, 4)
        self.wts = torch.stack([(1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx], -1).reshape(-1, 4)
        # the bilinear weights are separable (w = wy(y) wx(x)): node sums of a pixel image F are Wy^T F Wx with the
        # dense 1D interpolation matrices (two small matmuls instead of a scatter of 4 x npx values into G^2 nodes)
        def interp1(n_):
            f_ = torch.arange(n_, device=self.dev, dtype=torch.float32) / max(n_ - 1, 1) * (G - 1)
            i_ = torch.clamp(f_.floor().long(), max=G - 2)
            t_ = f_ - i_
            W = torch.zeros((n_, G), device=self.dev)
            W[torch.arange(n_, device=self.dev), i_] = 1 - t_
            W[torch.arange(n_, device=self.dev), i_ + 1] += t_
            return W
        self.Wy, self.Wx = interp1(self.ny), interp1(self.nx)
        self.logb = torch.zeros(self.nf, device=self.dev)
        # the smooth correction S per block of frames (bg_block_frames; None: one for the whole window): the
        # background's spatial pattern changes with rotation, and the window length is a memory choice. The
        # background objective separates exactly by block (per-frame levels belong to one block each).
        tb = int(bg_block_frames) if bg_block_frames else self.nf
        self.n_bblk = max(1, int(np.ceil(self.nf / max(tb, 1))))
        self.bblk_frames = int(np.ceil(self.nf / self.n_bblk))            # equal blocks
        self.bblk_of = torch.as_tensor(np.minimum(np.arange(self.nf) // self.bblk_frames, self.n_bblk - 1),
                                       device=self.dev)
        self.S = torch.zeros((self.n_bblk, G * G), device=self.dev)
        nref = len(wm.refl["x"])
        self.owned = torch.zeros(nref, dtype=torch.bool, device=self.dev)
        if owned is not None:
            self.owned[torch.as_tensor(owned, device=self.dev)] = True
        # only reflections with >= min_valid of the profile on trusted pixels are fitted (MINPK): with the profile
        # in masked pixels the Fisher information vanishes and the Newton step explodes (|I| ~ 1e4-1e6 counts)
        self.pvalid = wm.valid_fraction(self.valid)
        # overload recovery: a reflection whose missing profile mass is (mostly) overloaded pixels is fitted from its
        # unsaturated pixels (no likelihood term on overloads; sigma from the remaining pixels) when they carry at
        # least min_valid_overload of the profile -- the strongest low-resolution reflections are otherwise lost
        self.pover = wm.valid_fraction(self.ns == -2)
        recov = (self.pvalid < min_valid) & (self.pvalid >= min_valid_overload) & \
            (self.pvalid + self.pover >= min_valid)
        self.owned &= (self.pvalid >= min_valid) | recov
        # every fitted reflection with overloaded pixels in its profile is flagged (the likelihood has no term on
        # them, so the fit is valid): downstream, the shoebox integrator's overload rejection then does not apply
        self.recovered = self.owned & (self.pover > 1e-6)
        self.log = log
        self.damping = damping
        self.newton_tol, self.cg_tol = float(newton_tol), float(cg_tol)        # intensity-solve stopping rules (sigma)
        self.newton_first = None                     # Newton steps of the first outer cycle (None: 4)

    @property
    def valid(self):
        """Trusted voxels (bool, full window; built on use)."""
        return self.ns >= 0

    @property
    def n0(self):
        """Counts with untrusted voxels at 0 (float32, full window; built on use)."""
        return torch.clamp(self.ns, min=0).float()

    def valid_at(self, idx):
        return self.ns[idx] >= 0

    def n0_at(self, idx):
        return torch.clamp(self.ns[idx], min=0).float()

    def B_at(self, idx, pix=None):
        """Background at the voxel indices idx (any shape): per-frame level x template x smooth correction. pix: the
        per-pixel part exp(logT + S) precomputed (pixel_background())"""
        npx = self.ny * self.nx
        if pix is None:
            pix = self.pixel_background()
        k = idx // npx
        return torch.exp(self.logb[k]) * pix.reshape(-1)[self.bblk_of[k] * npx + idx % npx]

    def _Sp(self, b):
        """Smooth correction of background block b per pixel (ny * nx,)."""
        return (self.S[b][self.nodes] * self.wts).sum(-1)

    def pixel_background(self):
        """exp(logT + S) per background block and pixel (n_bblk, ny * nx)."""
        return torch.exp(self.logT[None] + torch.stack([self._Sp(b) for b in range(self.n_bblk)]))

    def mu_full(self, I, beta="model"):
        """mu = max(B + sum_h I_h f_h, 1e-3 B) without a full-window background array (B added per frame chunk)."""
        mu = self.wm.mu(I, None, beta=beta)
        for k0, k1, sl in self.chunks():
            Bc = self.background(k0, k1)
            mu[sl] += Bc
            torch.maximum(mu[sl], 1e-3 * Bc, out=mu[sl])
        return mu

    # ---------------------------------------------------------------- background
    def background(self, k0=0, k1=None):
        """B over frames [k0, k1) of the window (flattened)."""
        k1 = self.nf if k1 is None else k1
        if self.n_bblk == 1:
            Sp = self._Sp(0)[None]                                            # (1, ny*nx)
        else:
            bl = self.bblk_of[k0:k1]
            b0 = int(bl[0])
            Sp = torch.stack([self._Sp(b) for b in range(b0, int(bl[-1]) + 1)])[bl - b0]
        lb = self.logb[k0:k1, None] + (self.logT[None] + Sp)                  # (nf, ny*nx)
        return torch.exp(lb).reshape(-1)

    def floor_mu(self, mu, B):
        """mu = max(mu, 1e-3 B) in place, by frame chunks."""
        for _, _, sl in self.chunks():
            torch.maximum(mu[sl], 1e-3 * B[sl], out=mu[sl])
        return mu

    def chunks(self, lo=0, hi=None):
        npx = self.ny * self.nx
        hi = self.nf if hi is None else hi
        for k0 in range(lo, hi, self.fchunk):
            k1 = min(k0 + self.fchunk, hi)
            yield k0, k1, slice(k0 * npx, k1 * npx)

    def pois_sum(self, mu):
        """sum over trusted voxels of mu - n log mu (float64), by frame chunks."""
        tot, zero = 0.0, torch.zeros((), device=self.dev)
        for _, _, sl in self.chunks():
            m = mu[sl]
            tot += float(torch.where(self.valid_at(sl), m - self.n0_at(sl) * torch.log(m), zero).sum(dtype=torch.float64))
        return tot

    def init_background(self, I):
        """Per-frame level from the counts minus the model signal at the starting intensities."""
        sig = self.wm.mu(I, None)
        T = torch.exp(self.logT)
        for k in range(self.nf):
            sl = slice(k * self.ny * self.nx, (k + 1) * self.ny * self.nx)
            v = self.valid_at(sl)
            num = (self.n0_at(sl) - sig[sl])[v].sum()
            den = T[v].sum()
            self.logb[k] = torch.log(torch.clamp(num / den, min=1e-4))

    def fit_background(self, I, n_iter=3, ridge=1e2):
        """Newton on (log b_k, S) with the signal fixed. Fisher: sum B^2/mu x basis outer products; S has a
        ridge (smoothness towards 0 = the template's shape) worth ``ridge`` counts^2 per node. Statistics and the
        objective are accumulated by frame chunks (no full-window temporaries besides the signal). With background
        blocks (bg_block_frames) each block's (log b_k, S_b) is fitted on its own frames (the objective separates)."""
        sig = self.wm.mu(I, None)
        for b in range(self.n_bblk):
            lo = b * self.bblk_frames
            hi = min(self.nf, (b + 1) * self.bblk_frames) if b < self.n_bblk - 1 else self.nf
            self._fit_background_block(sig, b, lo, hi, n_iter, ridge)

    def _node_sums(self, F):
        """sum_p F[k, p] w_a(p) for every background node a: (k, ny, nx) -> (k, G^2) (separable bilinear weights)."""
        return (self.Wy.T @ (F @ self.Wx)).reshape(F.shape[0], -1)

    def _fit_background_block(self, sig, b, lo, hi, n_iter, ridge):
        npx = self.ny * self.nx
        G2 = self.G * self.G
        nfb = hi - lo
        zero = torch.zeros((), device=self.dev)
        for _ in range(n_iter):
            g_b = torch.zeros(nfb, device=self.dev, dtype=torch.float64)
            h_b = torch.zeros(nfb, device=self.dev, dtype=torch.float64)
            rp = torch.zeros(npx, device=self.dev)
            fp = torch.zeros(npx, device=self.dev)
            HbS = torch.zeros((nfb, G2), device=self.dev)
            for k0, k1, sl in self.chunks(lo, hi):
                B = self.background(k0, k1)
                mu = torch.clamp(B + sig[sl], min=1e-6)
                v = self.valid_at(sl)
                r2 = torch.where(v, B * (1.0 - self.n0_at(sl) / mu), zero).reshape(k1 - k0, npx)   # dNLL/dlogB
                f2 = torch.where(v, B * B / mu, zero).reshape(k1 - k0, npx)                   # Fisher weight
                del B, mu
                g_b[k0 - lo:k1 - lo] = r2.sum(1, dtype=torch.float64)
                h_b[k0 - lo:k1 - lo] = f2.sum(1, dtype=torch.float64)
                rp += r2.sum(0)
                fp += f2.sum(0)
                HbS[k0 - lo:k1 - lo] = self._node_sums(f2.reshape(k1 - k0, self.ny, self.nx))
                del r2, f2
            g_S = self._node_sums(rp.reshape(1, self.ny, self.nx))[0]
            # Fisher of S: sum_p fp w_a w_b at node pairs (4x4 per pixel), cross term with b_k: sum_p f2[k] w_a
            HS = torch.zeros(G2 * G2, device=self.dev)
            for a_ in range(4):
                for b_ in range(4):
                    HS.index_add_(0, self.nodes[:, a_] * G2 + self.nodes[:, b_], fp * self.wts[:, a_] * self.wts[:, b_])
            HS = HS.reshape(G2, G2) + ridge * torch.eye(G2, device=self.dev) / max(npx, 1) * fp.sum()
            H = torch.zeros((nfb + G2, nfb + G2), device=self.dev, dtype=torch.float64)
            H[:nfb, :nfb] = torch.diag(h_b)
            H[:nfb, nfb:] = HbS.double()
            H[nfb:, :nfb] = HbS.T.double()
            H[nfb:, nfb:] = HS.double()
            g = torch.cat([g_b, (g_S + ridge / max(npx, 1) * fp.sum() * self.S[b]).double()])
            step = torch.linalg.solve(H + 1e-9 * torch.eye(H.shape[0], device=self.dev, dtype=torch.float64), g)
            ridge_w = 0.5 * ridge / max(npx, 1) * float(fp.sum())

            # backtracking on the Poisson objective (+ the ridge on S) of this block's frames
            def obj():
                tot = 0.0
                for k0, k1, sl in self.chunks(lo, hi):
                    mu_ = torch.clamp(self.background(k0, k1) + sig[sl], min=1e-6)
                    tot += float(torch.where(self.valid_at(sl), mu_ - self.n0_at(sl) * torch.log(mu_), zero)
                                 .sum(dtype=torch.float64))
                return tot + ridge_w * float((self.S[b] ** 2).sum())
            f_old, lb0, S0 = obj(), self.logb[lo:hi].clone(), self.S[b].clone()
            t = 1.0
            for _ in range(6):
                self.logb[lo:hi] = lb0 - t * step[:nfb].float()
                self.S[b] = S0 - t * step[nfb:].float()
                if obj() <= f_old:
                    break
                t *= 0.5
            else:
                self.logb[lo:hi], self.S[b] = lb0, S0
                break

    # ---------------------------------------------------------------- intensities

    def nll(self, I):
        return self.pois_sum(self.mu_full(I))

    def fit_intensities(self, I, n_newton=4, n_cg=10):
        """Newton / Fisher scoring on the owned intensities (CG on the exact Fisher matrix)."""
        own = self.owned
        zero = torch.zeros((), device=self.dev)
        ns = self.ns
        vat = lambda vc: ns[vc] >= 0
        # stopping rules in sigma units (A1): Newton stops when its accepted step moves no owned intensity by more
        # than newton_tol sigma, CG when its last update is below cg_tol sigma (0: the fixed iteration counts)
        newton_tol, cg_tol = self.newton_tol, self.cg_tol
        st = self.solve_stats = getattr(self, "solve_stats", dict(newton=0, ls_fail=0, cg_cap=0))
        mu = self.mu_full(I)
        for it in range(n_newton):
            st["newton"] += 1
            # weights are computed on the gathered voxels only
            g, fd = self.wm.grad_newton(mu, ns)
            g = torch.where(own, g, zero)
            Minv = torch.where(own & (fd > 0), 1.0 / torch.clamp((1 + self.damping) * fd, min=1e-12), zero)
            sfd = torch.where(own & (fd > 0), torch.sqrt(torch.clamp(fd, min=0)), zero)     # 1 / sigma

            lam = self.damping * torch.where(own, fd, zero)          # Levenberg-Marquardt: H + lam diag(H)

            def Hv(v):
                a = self.wm.mu(v, None, beta=None)
                hv = self.wm.grad_ratio(a, mu, ns)
                return torch.where(own, hv + lam * v, zero)
            # CG for H d = -g
            d = torch.zeros_like(I)
            r = -g
            z = Minv * r
            p = z.clone()
            rz = (r * z).sum()
            for k_cg in range(n_cg):
                if k_cg == n_cg - 1:
                    st["cg_cap"] += 1              # the last allowed CG iteration (no convergence before it)
                Hp = Hv(p)
                alpha = rz / torch.clamp((p * Hp).sum(), min=1e-30)
                d = d + alpha * p
                r = r - alpha * Hp
                z = Minv * r
                rz_new = (r * z).sum()
                if float(rz_new) <= 1e-8 * float(rz):
                    break
                if cg_tol > 0 and float((torch.abs(alpha * p) * sfd).max()) < cg_tol:
                    break
                p = z + (rz_new / torch.clamp(rz, min=1e-30)) * p
                rz = rz_new
            # backtracking: the Poisson objective must not rise (quadratic model can overshoot on weak data); the
            # accepted point's mu is kept for the next iteration (one likelihood pass fewer per iteration)
            f_old = self.pois_sum(mu)
            del mu
            mu, t = None, 1.0
            for _ in range(5):
                I_try = I + t * d
                mu_try = self.mu_full(I_try)
                if self.pois_sum(mu_try) <= f_old:
                    mu = mu_try
                    break
                del mu_try
                t *= 0.5
            if mu is None:
                st["ls_fail"] += 1                 # no decrease along d (the objective's own noise at convergence?)
                mu = self.mu_full(I)
                break
            I = I_try
            if newton_tol > 0 and float((torch.abs(t * d) * sfd).max()) < newton_tol:
                break
        # sigma from the Fisher information with mu held at the background level at least: a weak reflection driven
        # negative takes mu towards its floor (1e-3 B) on pixels with zero counts, where 1 / mu explodes and sigma
        # collapsed to ~0.1 counts. XPG's zero-intensity variance uses the background model the same way.
        pixB = self.pixel_background()
        _, fd = self.wm.grad(lambda vc, ok: torch.zeros(vc.shape, device=self.dev),
                             lambda vc: torch.maximum(mu[vc], self.B_at(vc, pixB)), valid=vat)
        sig = torch.where(own & (fd > 0), 1.0 / torch.sqrt(torch.clamp(fd, min=1e-30)), torch.full((), float("nan"), device=self.dev))
        return I, sig

    def run(self, I0, n_outer=3, log_every=True, outer_tol=0.0, max_outer=4):
        """n_outer cycles of background fit + intensity solve. outer_tol > 0: adaptive -- after n_outer cycles,
        further cycles (up to max_outer) while the 99.9% quantile of |dI| / sigma between cycles exceeds
        outer_tol (the background / intensity coupling converges slowly where many reflections overlap)."""
        t0 = time.time()
        I = I0.clone()
        if not getattr(self, "_bg_init", False):
            self.init_background(I)
            self._bg_init = True
        sig = None
        n_max = max(n_outer, max_outer) if outer_tol > 0 else n_outer
        self.n_outer_done = 0
        for o in range(n_max):
            I_prev = I
            self.fit_background(I)
            # the first cycle's intensities only feed the background refit: optionally fewer Newton steps there
            nn = self.newton_first if (o == 0 and n_max > 1 and self.newton_first) else 4
            I, sig = self.fit_intensities(I, n_newton=nn)
            self.n_outer_done = o + 1
            if log_every:
                self.log(f"    FM window: outer {o + 1}: NLL {self.nll(I):.6g} ({time.time() - t0:.0f}s)")
            if outer_tol > 0 and o + 1 >= n_outer and o + 1 < n_max:
                ok = self.owned & torch.isfinite(sig) & (sig > 0)
                if not bool(ok.any()):
                    break
                z = (torch.abs(I - I_prev) / sig)[ok]
                k = max(0, min(z.numel() - 1, int(0.999 * z.numel())))
                if float(torch.kthvalue(z.float().cpu(), k + 1).values) < outer_tol:
                    break
        return I, sig

# ---------------------------------------------------------------------------- M2: widths by likelihood
THETA = ("log sigma_D", "log sigma_M", "logit halo w", "log halo kappa")


def theta_of(wm):
    w = wm.width
    hw, hk = w["halo"] if w["halo"] is not None else (0.05, 2.5)
    return np.array([np.log(w["sd"]), np.log(w["sm"]), np.log(hw / (1 - hw)), np.log(hk)])


def set_theta(wm, th):
    """Widths from theta; the halo is switched on unless wm.halo_free is False and it is off."""
    if not getattr(wm, "halo_free", True) and wm.width["halo"] is None:
        wm.set_widths(sd=float(np.exp(th[0])), sm=float(np.exp(th[1])))
        return
    hw = 1.0 / (1.0 + np.exp(-th[2]))
    wm.set_widths(sd=float(np.exp(th[0])), sm=float(np.exp(th[1])), halo=(float(hw), float(np.exp(th[3]))))


def width_terms(fit, I, h=2e-3, n_free=4):
    """Gradient (4,), Fisher (4, 4) and NLL of one window w.r.t. theta at fixed intensities and background, from a
    forward-difference Jacobian of mu over the window's voxels (all reflections, neighbours included)."""
    wm, B = fit.wm, fit.background()
    th0 = theta_of(wm)
    set_theta(wm, th0)
    zero = torch.zeros((), device=fit.dev)
    mu0 = torch.maximum(wm.mu(I, B), 1e-3 * B)
    inv = torch.where(fit.valid, 1.0 / mu0, zero)
    r = torch.where(fit.valid, 1.0 - fit.n0 / mu0, zero)
    nll = float(torch.where(fit.valid, mu0 - fit.n0 * torch.log(mu0), zero).sum(dtype=torch.float64))
    J = []
    for a in range(n_free):
        th = th0.copy()
        th[a] += h
        set_theta(wm, th)
        J.append((torch.maximum(wm.mu(I, B), 1e-3 * B) - mu0) / h)
    set_theta(wm, th0)
    g = np.array([float((r * Ja).sum(dtype=torch.float64)) for Ja in J])
    F = np.array([[float((J[a] * J[b] * inv).sum(dtype=torch.float64)) for b in range(n_free)] for a in range(n_free)])
    return g, F, nll


def window_nll(fit, I):
    return fit.nll(I)


def fit_widths(fits, Is, n_iter=5, log=print, damping=1e-3, n_free=4):
    """Global widths (shared by all windows) by damped Gauss-Newton on the summed Poisson NLL, alternating with the
    intensity / background solves of each window. fits: WindowFit per window (same theta); Is: their intensities.
    Returns theta, Laplace sd of theta (inverse Fisher, conditional on the intensities) and the updated Is."""
    th = theta_of(fits[0].wm)
    sd_th = None
    for f in fits:
        set_theta(f.wm, th)
    for it in range(n_iter):
        G, F, nll = np.zeros(n_free), np.zeros((n_free, n_free)), 0.0
        for f, I in zip(fits, Is):
            g_, F_, n_ = width_terms(f, I, n_free=n_free)
            G += g_; F += F_; nll += n_
        step = np.zeros(4)
        step[:n_free] = -np.linalg.solve(F + damping * np.diag(np.diag(F)) + 1e-12 * np.eye(n_free), G)
        t, best = 1.0, None
        for _ in range(6):
            th_try = th + t * step
            for f in fits:
                set_theta(f.wm, th_try)
            n_try = sum(window_nll(f, I) for f, I in zip(fits, Is))
            if n_try < nll:
                best = (th_try, n_try)
                break
            t *= 0.5
        if best is None:
            for f in fits:
                set_theta(f.wm, th)
            log(f"  M2 iteration {it + 1}: no decrease of the NLL; stop")
            break
        th = best[0]
        # intensities and background at the new widths
        Is = [f.run(I, n_outer=1, log_every=False)[0] for f, I in zip(fits, Is)]
        sd_th = np.sqrt(np.maximum(np.diag(np.linalg.pinv(F)), 0))
        hw = 1.0 / (1.0 + np.exp(-th[2]))
        log(f"  M2 iteration {it + 1}: NLL {nll:.6g} -> {best[1]:.6g} (step x{t:g}); sigma_D {np.degrees(np.exp(th[0])):.4f} "
            f"sigma_M {np.degrees(np.exp(th[1])):.4f} deg, halo w {hw:.3f} kappa {np.exp(th[3]):.2f}; Laplace sd of theta "
            + " ".join(f"{v:.3g}" for v in sd_th))
    return th, sd_th, Is


# ---------------------------------------------------------------------------- M3: geometry against the pixels
# forward-difference steps: rotations and beam tilt 2e-5 rad (~4% of a typical sigma_D), detector shift 5 um, cell
# 1e-5 (relative length / rad)
GEOM_STEP = np.array([2e-5, 2e-5, 2e-5, 5e-3, 5e-3, 5e-3, 2e-5, 2e-5])


def geom_steps(wm):
    return np.r_[GEOM_STEP, np.full(len(wm.geom) - len(GEOM_STEP), 1e-5)]


def geometry_terms(fit, I, gamma, free=None, thr=0.02, theta=None, theta_free=(), vox_frac=1.0, seed=0):
    """Profile-likelihood Gauss-Newton terms of one window w.r.t. the geometry correction gamma (model.GEOM), with
    the owned intensities eliminated (Schur complement over a diagonal intensity Fisher matrix; the background is
    held fixed). The geometry and the intensities are strongly coupled: at fixed intensities a geometry step gains
    little because the intensities, fitted at the old geometry, absorb part of the error, and alternating converges
    slowly (injection test: about 1/2 of the step per cycle) with conditional sds that are too small.
    The Jacobian of mu is kept only on trusted voxels with predicted signal > thr counts (elsewhere ~0).
    Returns dict: g, F (reduced, for the free components), nll, and C (n_free, n_refl), d, gI to form the joint
    intensity increment dI = -(gI + C^T dgamma) / d on the owned reflections.
    theta / theta_free: profile widths (M2 parameterisation, THETA) at which to evaluate and the components fitted
    jointly with the geometry (distance and cell move the spots' size in pixels, so too-wide widths drive them);
    their columns come first, in the order of theta_free."""
    wm, B = fit.wm, fit.background()
    free = np.arange(len(wm.geom)) if free is None else np.asarray(free)
    steps = geom_steps(wm)
    theta_free = np.asarray(theta_free, int)
    if theta is not None:
        set_theta(wm, theta)
    wm.set_geometry(gamma)
    zero = torch.zeros((), device=fit.dev)
    mu0 = torch.maximum(wm.mu(I, B), 1e-3 * B)
    nll = float(torch.where(fit.valid, mu0 - fit.n0 * torch.log(mu0), zero).sum(dtype=torch.float64))
    w = torch.where(fit.valid, 1.0 - fit.n0 / mu0, zero)
    gI, d = wm.grad(w, torch.where(fit.valid, mu0, torch.full((), 1e30, device=fit.dev)))
    own = fit.owned & (d > 0)
    gI, d = torch.where(own, gI, zero), torch.where(own, d, torch.ones((), device=fit.dev))
    idx = torch.nonzero(fit.valid & (mu0 - B > thr)).squeeze(1)
    wsub = 1.0
    if vox_frac < 1.0:
        # random subsample of the signal voxels (crowded data: ~every voxel carries signal and the Jacobian columns
        # do not fit in memory); reweighted, so g and F stay unbiased
        # (multiplicative hash of the voxel index on the device: no 1e9-element random draw)
        h = ((idx + 7919 * seed) * 2654435761) % 1000003
        idx = idx[h < int(vox_frac * 1000003)]
        wsub = 1.0 / vox_frac
    m0 = mu0[idx]
    inv = 1.0 / m0
    r = w[idx]
    J, C = [], []
    full = torch.zeros_like(mu0)
    for a in theta_free:
        th = np.array(theta, float)
        th[a] += 2e-3
        set_theta(wm, th)
        J.append((torch.maximum(wm.mu(I, B)[idx], 1e-3 * B[idx]) - m0) / 2e-3)
    if len(theta_free):
        set_theta(wm, theta)
    for a in free:
        g_ = np.array(gamma, float)
        g_[a] += steps[a]
        wm.set_geometry(g_)
        J.append((torch.maximum(wm.mu(I, B)[idx], 1e-3 * B[idx]) - m0) / steps[a])
    wm.set_geometry(gamma)
    for Ja in J:
        full.zero_()
        full[idx] = Ja * inv * wsub
        C.append(torch.where(own, wm.grad(full)[0], zero))           # sum_v f_h J_a / mu
    C = torch.stack(C)                                                # (n_free, n_refl)
    g = wsub * np.array([float((r * Ja).sum(dtype=torch.float64)) for Ja in J])
    F = wsub * np.array([[float((Ja * Jb * inv).sum(dtype=torch.float64)) for Jb in J] for Ja in J])
    Cd = C / d
    g_red = g - (Cd @ gI).double().cpu().numpy()
    F_red = F - (Cd.double() @ C.double().T).cpu().numpy()
    return dict(g=g_red, F=F_red, F_cond=F, nll=nll, C=C, d=d, gI=gI, own=own)


def fit_geometry(fit, I, n_iter=4, free=None, log=print, damping=1e-3, refit=True):
    """Geometry correction of one window by damped Gauss-Newton on the profile likelihood (intensities eliminated,
    see geometry_terms), each step applied jointly with its intensity increment, then the window's intensity /
    background solve. Returns gamma, its Laplace sd (inverse reduced Fisher: marginal over the intensities), the
    correlation matrix of the fitted components and the intensities."""
    names = fit.wm.geom_names
    free = np.arange(len(names)) if free is None else np.asarray(free)
    gamma = np.array(fit.wm.geom, float)
    sd, corr = None, None
    zero = torch.zeros((), device=fit.dev)
    for it in range(n_iter):
        T = geometry_terms(fit, I, gamma, free)
        g, F, nll = T["g"], T["F"], T["nll"]
        step = -np.linalg.solve(F + damping * np.diag(np.diag(F)) + 1e-30 * np.eye(len(free)), g)
        dI = torch.where(T["own"], -(T["gI"] + (torch.as_tensor(step, dtype=torch.float32, device=fit.dev)[:, None]
                                                 * T["C"]).sum(0)) / T["d"], zero)
        t, best = 1.0, None
        for _ in range(6):
            g_try = gamma.copy()
            g_try[free] += t * step
            fit.wm.set_geometry(g_try)
            n_try = window_nll(fit, I + t * dI)
            if n_try < nll:
                best = (g_try, n_try)
                break
            t *= 0.5
        if best is None:
            fit.wm.set_geometry(gamma)
            log(f"  M3 iteration {it + 1}: no decrease of the NLL; stop")
            break
        gamma = best[0]
        I = I + t * dI
        if refit:
            I = fit.run(I, n_outer=1, log_every=False)[0]
        cov = np.linalg.pinv(F)
        sd = np.sqrt(np.maximum(np.diag(cov), 0))
        corr = cov / np.maximum(np.outer(sd, sd), 1e-300)
        sd_c = np.sqrt(np.maximum(np.diag(np.linalg.pinv(T["F_cond"])), 0))
        log(f"  M3 iteration {it + 1}: NLL {nll:.10g} -> {best[1]:.10g} (dNLL {best[1] - nll:+.1f}, step x{t:g}); "
            + ", ".join(f"{names[a]} {_geom_fmt(a, gamma[a])}" for a in free)
            + f"; marginal / conditional sd {np.median(sd / np.maximum(sd_c, 1e-300)):.2f}")
    return gamma, sd, corr, I


def _geom_fmt(a, v):
    if 3 <= a < 6:
        return f"{v * 1e3:+.4f} um"
    return f"{np.degrees(v) * 1e3:+.3f} mdeg" if a < 8 else f"{v * 1e6:+.1f} ppm"



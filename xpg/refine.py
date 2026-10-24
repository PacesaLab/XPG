"""Geometric refinement of diffraction parameters against spot positions (x, y, phi).

Levenberg-Marquardt with forward-difference Jacobians (one batched residual evaluation), robust (Cauchy) IRLS
weights and iterative outlier rejection. Refinable blocks:
BEAM, AXIS, POSITION (ORGX, ORGY, DISTANCE; detector axes fixed), DETECTOR_ORIENTATION, ORIENTATION, CELL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .backend import device, np_, t64
from .detector import Segment, local_arrays, panel_affine_t
from .geometry import Geometry, predict_phi, rotvec_to_matrix_t, unit
from .models import CELL_PARAMS, B_t, Experiment, cell_from_free, free_from_cell


def _perp_basis(v):
    v = unit(v)
    t = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(v, t))
    e2 = np.cross(v, e1)
    return e1, e2


class Parameterisation:
    """Maps a flat parameter vector to experiment quantities (torch-differentiable)."""

    def __init__(self, exp: Experiment, flags):
        flags = [f.upper() for f in flags]
        g, c = exp.geom, exp.crystal
        self.exp = exp
        self.flags = flags
        self.blocks = []
        self.beam0, self.axis0 = unit(g.beam), unit(g.axis)
        self.be1, self.be2 = _perp_basis(self.beam0)
        self.ae1, self.ae2 = _perp_basis(self.axis0)
        self.ed1_0, self.ed2_0 = g.ed1.copy(), g.ed2.copy()
        self.U0 = c.U.copy()
        self.system = c.system
        x0 = []
        names = []
        # units: angles in mrad, positions px / mm, cell Angstrom / degree
        if "BEAM" in flags:
            self.blocks.append(("beam", len(x0), 2)); x0 += [0.0, 0.0]; names += ["beam_r1", "beam_r2"]
        if "AXIS" in flags:
            self.blocks.append(("axis", len(x0), 2)); x0 += [0.0, 0.0]; names += ["axis_r1", "axis_r2"]
        if "POSITION" in flags or "DISTANCE" in flags or "ORIGIN" in flags:
            self.blocks.append(("origin", len(x0), 3)); x0 += [g.orgx, g.orgy, g.distance]
            names += ["orgx", "orgy", "distance"]
        # POSITION = ORGX, ORGY, DISTANCE with fixed detector axes. Detector rotations together with
        # BEAM + AXIS + ORIENTATION are gauge-redundant (rotating the detector = rotating beam, axis and crystal),
        # so they are refined only on explicit request, or with SEGMENT (the segment gauge restraints keep the
        # mean segment motion zero and the global detector pose carries it)
        if "DETECTOR_ORIENTATION" in flags or ("POSITION" in flags and "SEGMENT" in flags):
            self.blocks.append(("dettilt", len(x0), 3)); x0 += [0.0, 0.0, 0.0]; names += ["det_r1", "det_r2", "det_r3"]
        if "ORIENTATION" in flags:
            self.blocks.append(("orient", len(x0), 3)); x0 += [0.0, 0.0, 0.0]; names += ["U_r1", "U_r2", "U_r3"]
        if "CELL" in flags:
            fc = free_from_cell(self.system, c.cell)
            self.blocks.append(("cell", len(x0), fc.size)); x0 += list(fc); names += CELL_PARAMS[self.system]
        self.segs = g.segment_list()
        K = len(self.segs)
        self.segmask = np.zeros((K, 6))
        if "SEGMENT" in flags and K > 1:
            for k, sg in enumerate(self.segs):
                ref = sg.refine or ("POSITION", "ORIENTATION")
                self.segmask[k, :3] = "POSITION" in ref
                self.segmask[k, 3:] = "ORIENTATION" in ref
            self.blocks.append(("segment", len(x0), 6 * K)); x0 += [0.0] * (6 * K)
            names += [f"seg{k + 1}_{n}" for k in range(K) for n in ("d1", "d2", "d3", "r1", "r2", "r3")]
        self.n_extra = 6 if any(b[0] == "segment" for b in self.blocks) else 0
        self.x0 = np.array(x0, float)
        self.names = names

    def consts(self):
        """Array constants of the parameterisation (float64 tensors on the compute device)."""
        g, c = self.exp.geom, self.exp.crystal
        f = t64
        C, EX, EY = local_arrays(self.segs, g.qx, g.qy)
        cen = np.array([sg.centre for sg in self.segs], float)
        return dict(beam0=f(self.beam0), axis0=f(self.axis0), be1=f(self.be1), be2=f(self.be2),
                    ae1=f(self.ae1), ae2=f(self.ae2), ed1_0=f(self.ed1_0), ed2_0=f(self.ed2_0), U0=f(self.U0),
                    cell0=f(c.cell), org0=f([g.orgx, g.orgy, g.distance]), C=f(C), EX=f(EX), EY=f(EY),
                    cen=f(cen), q=f([g.qx, g.qy]), segmask=f(self.segmask))

    @property
    def key(self):
        return (tuple(self.blocks), self.system)

    def unpack(self, x, k=None):
        return _unpack(self.key, x, self.consts() if k is None else k)

    def apply(self, x) -> Experiment:
        d = {k: np_(v).astype(float) for k, v in self.unpack(t64(x)).items() if v is not None}
        g = self.exp.geom.copy(beam=d["beam"] / np.linalg.norm(d["beam"]),
                               axis=d["axis"] / np.linalg.norm(d["axis"]),
                               orgx=float(d["orgx"]), orgy=float(d["orgy"]), distance=float(d["dist"]),
                               ed1=d["ed1"], ed2=d["ed2"])
        if self.exp.geom.segmented:
            g.segments = [Segment.from_local(sg.rect, d["C"][k], d["EX"][k], d["EY"][k], g.qx, g.qy, sg.refine)
                          for k, sg in enumerate(self.segs)]
        c = self.exp.crystal.copy(U=d["U"], cell=d["cell"])
        return Experiment(g, c)


def _mv(R, v):
    """(..., 3, 3) @ (..., 3) -> (..., 3)."""
    return (R @ v[..., None])[..., 0]


def _unpack(key, x, k):
    """Experiment quantities of parameter vector(s) x (..., p): batched over leading dimensions (the
    forward-difference Jacobian evaluates all perturbed vectors at once), except for segment parameters."""
    blocks, system = key
    beam, axis = k["beam0"], k["axis0"]
    orgx, orgy, dist = k["org0"][0], k["org0"][1], k["org0"][2]
    ed1, ed2 = k["ed1_0"], k["ed2_0"]
    U, cell = k["U0"], k["cell0"]
    C, EX, EY = k["C"], k["EX"], k["EY"]
    seg_v = None
    for name, i, n in blocks:
        v = x[..., i:i + n]
        if name == "beam":
            beam = _mv(rotvec_to_matrix_t(1e-3 * (v[..., 0:1] * k["be1"] + v[..., 1:2] * k["be2"])), beam)
        elif name == "axis":
            axis = _mv(rotvec_to_matrix_t(1e-3 * (v[..., 0:1] * k["ae1"] + v[..., 1:2] * k["ae2"])), axis)
        elif name == "origin":
            orgx, orgy, dist = v[..., 0], v[..., 1], v[..., 2]
        elif name == "dettilt":
            e3 = torch.linalg.cross(k["ed1_0"], k["ed2_0"])
            R = rotvec_to_matrix_t(1e-3 * (v[..., 0:1] * k["ed1_0"] + v[..., 1:2] * k["ed2_0"] + v[..., 2:3] * e3))
            ed1, ed2 = _mv(R, ed1), _mv(R, ed2)
        elif name == "orient":
            U = rotvec_to_matrix_t(1e-3 * v) @ U
        elif name == "cell":
            cell = cell_from_free(system, v)
        elif name == "segment":
            if x.dim() != 1:
                raise ValueError("segment parameters are not batched")
            # each segment: shift (mm) and rotation (mrad) about its centre, in the detector system
            seg_v = v.reshape(-1, 6) * k["segmask"]
            qx, qy = k["q"][0], k["q"][1]
            cx, cy = k["cen"][:, 0:1], k["cen"][:, 1:2]
            m = C + qx * cx * EX + qy * cy * EY
            R = rotvec_to_matrix_t(1e-3 * seg_v[:, 3:])
            EX = torch.einsum("kij,kj->ki", R, EX)
            EY = torch.einsum("kij,kj->ki", R, EY)
            C = m + seg_v[:, :3] - qx * cx * EX - qy * cy * EY
    Astar = U @ B_t(cell)
    P0, Bx, By = panel_affine_t(C, EX, EY, orgx, orgy, dist, ed1, ed2, k["q"][0], k["q"][1])
    return dict(beam=beam, axis=axis, orgx=orgx, orgy=orgy, dist=dist, ed1=ed1, ed2=ed2, U=U, cell=cell, Astar=Astar,
                C=C, EX=EX, EY=EY, P0=P0, Bx=Bx, By=By, seg_v=seg_v)


def _residual_fns(key):
    """Residual, Jacobian and weighted-normal-equation functions for a parameterisation structure."""
    segmented = any(b[0] == "segment" for b in key[0])
    def residuals(x, k, d):
        """Normalised residuals of parameter vector(s) x (..., p) -> (..., 3 n [+ gauge rows])."""
        q = _unpack(key, x, k)
        t, S, phi = predict_xyphi_t(q, d["H"], d["PH"], d["BR"], d["wl"])
        # intersect with the plane of the segment on which each spot was observed
        P0, Bx, By = q["P0"][..., d["SEG"], :], q["Bx"][..., d["SEG"], :], q["By"][..., d["SEG"], :]
        n = torch.linalg.cross(Bx, By)
        tt = (P0 * n).sum(-1) / (S * n).sum(-1)
        P = tt[..., None] * S - P0
        xp = q["orgx"][..., None] + (P * Bx).sum(-1) / (Bx * Bx).sum(-1)
        yp = q["orgy"][..., None] + (P * By).sum(-1) / (By * By).sum(-1)
        # thick-sensor parallax (zero when thickness unknown)
        e = S / torch.linalg.norm(S, dim=-1, keepdim=True)
        u1, u2 = Bx / d["qx"], By / d["qy"]
        un = torch.linalg.cross(u1, u2)
        cosa = torch.clamp((e * un).sum(-1), 1e-3, 1.0)
        L = d["tsens"] / cosa
        lbar = 1.0 / d["mu"] - L * torch.exp(-d["mu"] * L) / torch.clamp(-torch.expm1(-d["mu"] * L), min=1e-12)
        xp = xp + d["plx"] * lbar * (e * u1).sum(-1) / d["qx"]
        yp = yp + d["plx"] * lbar * (e * u2).sum(-1) / d["qy"]
        out = [(xp - d["XO"]) / d["sxy"], (yp - d["YO"]) / d["sxy"], (phi - d["PH"]) / d["sphi"]]
        if not out[0].shape == out[1].shape == out[2].shape:      # batched, phi independent of the parameters
            out = list(torch.broadcast_tensors(*out))
        if q["seg_v"] is not None:
            # gauge: segments move relative to the detector, not together with it
            # (mean shift 1 um, mean rotation 0.01 mrad scale)
            sv = q["seg_v"]
            out.append(torch.cat([sv[:, :3].mean(0) / 1e-3, sv[:, 3:].mean(0) / 1e-2]))
        return torch.cat(out, -1)

    def jac_r(x, k, d, r0=None):
        """Jacobian and the residuals at x."""
        # Jacobian only steers the Gauss-Newton / LM path, the solution is set by the residuals
        h = 1e-7 * torch.clamp(x.abs(), min=1.0)
        if x.numel() and not segmented:
            # x and its p perturbations in one batched evaluation
            X = x + torch.diag(h)
            if r0 is None:
                R = residuals(torch.cat([x[None], X]), k, d)
                r0, R = R[0], R[1:]
            else:
                R = residuals(X, k, d)
            return ((R - r0) / h[:, None]).T, r0
        r0 = residuals(x, k, d) if r0 is None else r0
        cols = []
        for i in range(x.numel()):
            xi = x.clone()
            xi[i] = xi[i] + h[i]
            cols.append((residuals(xi, k, d) - r0) / h[i])
        J = torch.stack(cols, 1) if cols else torch.zeros((r0.numel(), 0), dtype=r0.dtype, device=r0.device)
        return J, r0

    def jacobian(x, k, d, r0=None):
        return jac_r(x, k, d, r0)[0]

    def normal(x, k, d, w):
        # weighted normal equations J^T W J, J^T W r formed on the device: only p x p numbers leave it
        J, r = jac_r(x, k, d)
        JW = J * w[:, None]
        return JW.T @ J, JW.T @ r

    return residuals, jacobian, normal


_REFINE_SUB = 40000   # spots per refinement (larger lists are subsampled evenly)
_GPU_MIN = 2000       # refinements of fewer spots run on the CPU ...
_FEW_MAX = 20000      # ... and below this with one thread
_PREFILTER_N = 20000   # refine: spot lists larger than this are pre-filtered against the starting model


def predict_xyphi_t(q, hkl, phi_obs, branch, wavelength):
    """Predicted (x, y, phi[rad]) for hkl with the diffraction root chosen by ``branch``; batched over leading
    dimensions of the model quantities in q (shapes (..., n))."""
    m = q["axis"] / torch.linalg.norm(q["axis"], dim=-1, keepdim=True)
    s0 = q["beam"] / torch.linalg.norm(q["beam"], dim=-1, keepdim=True) / wavelength
    p0 = hkl @ q["Astar"].transpose(-1, -2)
    m_, s0_ = m[..., None, :], s0[..., None, :]
    par = (p0 * m_).sum(-1)[..., None] * m_
    perp = p0 - par
    mxp = torch.linalg.cross(m_.expand_as(p0), p0)
    a = (perp * s0_).sum(-1)
    b = (mxp * s0_).sum(-1)
    c = -0.5 * (p0 * p0).sum(-1) - (par * s0_).sum(-1)
    r = torch.sqrt(a * a + b * b)
    ratio = torch.clamp(c / r, -1 + 1e-9, 1 - 1e-9)
    phi = torch.atan2(b, a) + branch * torch.arccos(ratio)
    phi = phi + 2 * torch.pi * torch.round((phi_obs - phi) / (2 * torch.pi))
    cph, sph = torch.cos(phi)[..., None], torch.sin(phi)[..., None]
    p = par + cph * perp + sph * mxp
    S = p + s0_
    ed3 = torch.linalg.cross(q["ed1"], q["ed2"])
    t = q["dist"][..., None] / (S * ed3[..., None, :]).sum(-1)
    # pixel sizes are fixed (not refined)
    return t, S, phi


@dataclass
class RefineResult:
    exp: Experiment
    rmsd_xy: float
    rmsd_phi: float  # degrees
    n_used: int
    inliers: np.ndarray
    x: np.ndarray
    names: list
    esd: np.ndarray | None = None


def choose_branch(geom: Geometry, Astar, hkl, phi_obs_rad):
    pa, pb, valid = predict_phi(geom, hkl @ Astar.T)
    da = np.abs(np.angle(np.exp(1j * (pa - phi_obs_rad))))
    db = np.abs(np.angle(np.exp(1j * (pb - phi_obs_rad))))
    # pa = base + delta (branch +1), pb = base - delta (branch -1)
    return np.where(da <= db, 1.0, -1.0), valid, np.minimum(da, db)


def refine(exp: Experiment, hkl, xobs, yobs, zobs, flags, *args, **kw):
    """Geometry refinement (see _refine). Problems below _GPU_MIN spots run on the CPU."""
    from .backend import float64_on, torch_threads
    n = int(np.shape(hkl)[0])
    small = n < _GPU_MIN
    # small problems: the CPU, with one thread (thread-pool synchronisation dominates tiny tensors)
    threads = 1 if small and n < _FEW_MAX else None
    with float64_on("cpu" if small else None), torch_threads(threads):
        return _refine(exp, hkl, xobs, yobs, zobs, flags, *args, **kw)


def _refine(exp: Experiment, hkl, xobs, yobs, zobs, flags, sigma_xy=0.5, sigma_phi_deg=None,
           max_iter=30, reject=True, n_reject_cycles=4, verbose=False, log=print, robust=True,
           fixed_inliers=None, max_err_xy=3.0, max_err_phi=2.0, prior=None) -> RefineResult:
    """Outliers: absolute limits (MAXIMUM_ERROR_OF_SPOT_POSITION [px] / SPINDLE_POSITION [deg])
    plus a statistical limit of 6x the median normalised residual (~4.4 sigma for 3D Gaussian errors).
    prior: {parameter name: (centre, sd)} Gaussian restraints (e.g. per-block detector position around the global
    one), in the same chi^2 units as the normalised residuals."""
    g = exp.geom
    hkl = np.asarray(hkl, float)
    if reject and fixed_inliers is None and len(hkl) > _PREFILTER_N and flags:
        # large spot lists (indexing assigns every spot an index): refine on the spots that can become inliers
        # (within 3x the limits of the starting model), then classify all spots with the refined model
        pre = refine(exp, hkl, xobs, yobs, zobs, [], sigma_xy, sigma_phi_deg, reject=True,
                     max_err_xy=3.0 * max_err_xy, max_err_phi=3.0 * max_err_phi).inliers
        filtered = 0 < pre.sum() < 0.9 * len(hkl)
        idx = np.flatnonzero(pre) if filtered else np.arange(len(hkl))
        xo, yo, zo = np.asarray(xobs), np.asarray(yobs), np.asarray(zobs)
        if _REFINE_SUB and idx.size > _REFINE_SUB:
            # a few 10^4 spots spread over the rotation determine the ~20 geometry parameters far below their
            # uncertainty: refine on every k-th spot in rotation order, then polish on all
            k = int(np.ceil(idx.size / _REFINE_SUB))
            ss = idx[np.argsort(zo[idx], kind="stable")[::k]]
            sub = refine(exp, hkl[ss], xo[ss], yo[ss], zo[ss], flags, sigma_xy, sigma_phi_deg, max_iter, reject,
                         n_reject_cycles, verbose, log, robust, None, max_err_xy, max_err_phi, prior)
            sub = refine(sub.exp, hkl[idx], xo[idx], yo[idx], zo[idx], flags, sigma_xy, sigma_phi_deg, 4, reject, 1,
                         verbose, log, robust, np.ones(idx.size, bool), max_err_xy, max_err_phi, prior)   # (no re-entry)
        elif filtered:
            sub = refine(exp, hkl[idx], xo[idx], yo[idx], zo[idx], flags, sigma_xy, sigma_phi_deg, max_iter, reject,
                         n_reject_cycles, verbose, log, robust, None, max_err_xy, max_err_phi, prior)
        else:
            sub = None
        if sub is not None:
            full = refine(sub.exp, hkl, xobs, yobs, zobs, [], sigma_xy, sigma_phi_deg, reject=True,
                          n_reject_cycles=1, max_err_xy=max_err_xy, max_err_phi=max_err_phi, robust=False)
            return RefineResult(sub.exp, full.rmsd_xy, full.rmsd_phi, full.n_used, full.inliers, sub.x, sub.names,
                                sub.esd)
    phi_obs = np.radians(g.z_to_phi(zobs))
    if sigma_phi_deg is None:
        sigma_phi_deg = max(abs(g.osc) * 0.5, 0.02)
    branch, valid, _ = choose_branch(g, exp.crystal.Astar, hkl, phi_obs)
    par = Parameterisation(exp, flags)
    n = hkl.shape[0]
    ne = par.n_extra
    seg = g.segment_of(xobs, yobs)
    valid = valid & (seg >= 0)
    xc, yc = g.corrected_xy(xobs, yobs)          # geometric-distortion tables (X/Y-GEO_CORR)
    data = dict(H=t64(hkl), PH=t64(phi_obs), BR=t64(branch),
                XO=t64(np.asarray(xc, float)), YO=t64(np.asarray(yc, float)),
                SEG=torch.as_tensor(np.maximum(seg, 0), dtype=torch.int64, device=device()),
                wl=g.wavelength, qx=g.qx, qy=g.qy, sxy=float(sigma_xy), sphi=float(np.radians(sigma_phi_deg)),
                tsens=float(max(g.sensor_thickness, 1e-6)), mu=float(max(g.mu, 1e-6)),
                plx=float(g.has_parallax))
    consts = par.consts()
    f_res, f_jac, f_norm = _residual_fns(par.key)

    def res_j(x):
        return np_(f_res(t64(x), consts, data))

    def normal_j(x, w1):
        """(J^T W J, J^T W r) with per-spot weights w1 (restraint rows weight 1)."""
        wfull = t64(np.concatenate([np.tile(np.asarray(w1, float), 3), np.ones(ne)]))
        A, g_ = f_norm(t64(x), consts, data, wfull)
        return np_(A), np_(g_)

    inl = valid.copy() if fixed_inliers is None else (np.asarray(fixed_inliers) & valid)
    x = par.x0.copy()
    if x.size == 0:  # classification only (no free parameters)
        rr = res_j(x)[:3 * n].reshape(3, n)
        if reject:
            rn = np.sqrt((rr ** 2).sum(0))
            med = np.median(rn[inl]) if inl.any() else 1.0
            inl = inl & (rn < max(6.0 * med, 3.0)) & (np.hypot(rr[0], rr[1]) * sigma_xy <= max_err_xy) \
                & (np.abs(rr[2]) * sigma_phi_deg <= max_err_phi)
        return _result(par, x, rr, inl, sigma_xy, sigma_phi_deg, None)

    pidx = np.array([par.names.index(k) for k in (prior or {}) if k in par.names], int)
    pc = np.array([prior[par.names[i]][0] for i in pidx], float)
    ps = np.array([prior[par.names[i]][1] for i in pidx], float)

    def prior_cost(xx):
        return float(np.sum(((xx[pidx] - pc) / ps) ** 2)) if pidx.size else 0.0

    def cost_w(r, w, xx=None):
        return float(np.sum(w * r * r)) + (prior_cost(xx) if xx is not None else 0.0)

    for cycle in range(n_reject_cycles + 1 if reject else 1):
        lam = 1e-3
        r = res_j(x)
        for it in range(max_iter):
            rr = r[:3 * n].reshape(3, n)
            rn = np.sqrt((rr ** 2).sum(0))
            w1 = inl.astype(float)
            if robust:
                c = 3.0
                w1 = w1 / (1.0 + (rn / c) ** 2)
            w = np.concatenate([np.tile(w1, 3), np.ones(ne)])
            A, gr = normal_j(x, w1)
            if pidx.size:
                A[pidx, pidx] += 1.0 / ps ** 2
                gr[pidx] += (x[pidx] - pc) / ps ** 2
            D = np.diag(A).copy()
            D[D <= 0] = 1.0
            c0 = cost_w(r, w, x)
            improved = False
            for _ in range(10):
                try:
                    dx = -np.linalg.solve(A + lam * np.diag(D), gr)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                xn = x + dx
                rn_ = res_j(xn)
                c1 = cost_w(rn_, w, xn)
                if np.isfinite(c1) and c1 < c0:
                    x, r = xn, rn_
                    lam = max(lam / 3, 1e-9)
                    improved = True
                    break
                lam *= 4
            if not improved or (c0 - c1) < 1e-7 * c0:
                break
        if not reject:
            break
        rr = r[:3 * n].reshape(3, n)
        rn = np.sqrt((rr ** 2).sum(0))
        med = np.median(rn[inl]) if inl.any() else 1.0
        dxy = np.hypot(rr[0], rr[1]) * sigma_xy
        dphi = np.abs(rr[2]) * sigma_phi_deg
        new_inl = valid & (rn < max(6.0 * med, 3.0)) & (dxy <= max_err_xy) & (dphi <= max_err_phi)
        if fixed_inliers is not None:
            new_inl &= fixed_inliers
        if verbose:
            log(f"    refine cycle {cycle}: {inl.sum()} spots, median |r| {med:.2f}")
        if np.array_equal(new_inl, inl):
            break
        inl = new_inl
    # ESDs from the normal matrix
    try:
        w = np.concatenate([np.tile(inl.astype(float), 3), np.ones(ne)])
        A, _ = normal_j(x, inl.astype(float))
        dof = max(int(3 * inl.sum()) - x.size, 1)
        s2 = float(np.sum((w * r * r)[:3 * n])) / dof
        esd = np.sqrt(np.clip(np.diag(np.linalg.pinv(A)) * s2, 0, None))
    except Exception:
        esd = None
    return _result(par, x, r[:3 * n].reshape(3, n), inl, sigma_xy, sigma_phi_deg, esd)


def _result(par, x, rr, inl, sigma_xy, sigma_phi, esd):
    exp = par.apply(x)
    if inl.any():
        rxy = float(np.sqrt(np.mean((rr[0, inl] ** 2 + rr[1, inl] ** 2)))) * sigma_xy
        rphi = float(np.sqrt(np.mean(rr[2, inl] ** 2))) * sigma_phi
    else:
        rxy = rphi = np.inf
    return RefineResult(exp, rxy, rphi, int(inl.sum()), inl, x, par.names, esd)

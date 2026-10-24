"""Molecular replacement search on the GPU: rotation search, translation search and likelihood scoring
of a search model against merged amplitudes.

1. Rotation: the model's structure factors on a P1 reciprocal grid (gemmi density + FFT, model centred at the
   origin of a large box), and for every orientation R of a rotation grid the incoherent sum over the space group's
   rotations S_k of |E_model((S_k R)^T h)|^2 (translation-free: the cross terms between symmetry copies average
   out), sampled by trilinear interpolation. Score: correlation with E_obs^2 (data and model normalised in
   resolution shells, data 10-3.5 A by default).
2. Translation: for each of the best orientations, the correlation translation function by FFT: with
   F_k(h) = F_model((S_k R)^T h) the model's transform for symmetry copy k placed at t, |sum_k F_k(h) e^{2 pi i
   h.S_k t}|^2 = sum_jk F_j F_k* e^{2 pi i h.(S_j - S_k) t}; the pair terms are summed into one map by FFT
   (each pair's coefficients E_o^2 F_j F_k* placed at the Miller index (S_j - S_k)^T h of a fine grid).
3. Scoring: the translation-function peaks ranked by a fast sigma_A LLG (structure factors of each placement from
   the copies' transforms by phase shifts), then the placed model's full sigma_A LLG (xpg.llg) and rigid-body
   refinement for the best, best first.

Data pathologies (xpg.mr finds them with the processing analyses): with tNCS (translation t) the copies are placed
as pairs: each translation-function peak is completed by a partner of the same orientation at +t and at -t, the LLG
picks, and the pair is rigid-body refined as one; the rotation function's target is divided by the tNCS
modulation. Anisotropy is removed from that target and Wilson outliers are left out; a twin law and fraction enter
the LLG.

    python -m xpg.mrsearch data.mtz model.pdb [--step 6] [--top 20]"""
from __future__ import annotations

import argparse
import time

import gemmi
import numpy as np
import torch

from .backend import device


def rotation_grid(step_deg):
    """Approximately uniform rotations (quaternions on S^3 by a Hopf-like construction: Euler angles alpha, gamma in
    steps of `step`, beta with sin(beta) weighting). Returns (n, 3, 3) float64."""
    st = np.radians(step_deg)
    Rs = []
    for b in np.arange(st / 2, np.pi, st):
        na = max(1, int(round(2 * np.pi * np.sin(b) / st)))
        for a in np.arange(na) * 2 * np.pi / na:
            for g in np.arange(0, 2 * np.pi, st):
                ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
                Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
                Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
                Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
                Rs.append(Rz1 @ Ry @ Rz2)
    return np.array(Rs)


def model_transform(model_path, dmin, box_pad=1.6, rate=2.0):
    """The model centred at the origin in a P1 box (box_pad x its extent): complex F on a grid indexed by the box's
    Miller indices, plus the box cell. Returns (Fgrid (n1, n2, n3) complex64 with index offsets, box edge (A),
    centre (A, the model's original centroid))."""
    st = gemmi.read_structure(str(model_path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    xyz = np.array([a.pos.tolist() for ch in st[0] for res in ch for a in res])
    c = xyz.mean(0)
    ext = float(np.max(np.linalg.norm(xyz - c, axis=1)))
    L = 2 * ext * box_pad
    st2 = gemmi.Structure()
    st2.cell = gemmi.UnitCell(L, L, L, 90, 90, 90)
    st2.spacegroup_hm = "P 1"
    m = gemmi.Model("1")
    for ch in st[0]:
        ch2 = gemmi.Chain(ch.name)
        for res in ch:
            r2 = gemmi.Residue()
            r2.name, r2.seqid = res.name, res.seqid
            for a in res:
                a2 = gemmi.Atom()
                a2.name, a2.element, a2.occ, a2.b_iso = a.name, a.element, a.occ, a.b_iso
                p = np.array(a.pos.tolist()) - c + L / 2
                a2.pos = gemmi.Position(*p)
                r2.add_atom(a2)
            ch2.add_residue(r2)
        m.add_chain(ch2)
    st2.add_model(m)
    st2.setup_entities()
    dc = gemmi.DensityCalculatorX()
    dc.d_min = dmin
    dc.rate = rate
    dc.set_grid_cell_and_spacegroup(st2)
    dc.set_refmac_compatible_blur(st2[0])
    dc.put_model_density_on_grid(st2[0])
    fg = gemmi.transform_map_to_f_phi(dc.grid, half_l=False)
    arr = np.array(fg, copy=False)                       # (nu, nv, nw) complex, FFT order
    # model centred at L/2: phase shift to the origin, e^{-2 pi i h.(1/2,1/2,1/2)} = (-1)^(h+k+l)
    n = np.array(arr.shape)
    idx = [np.fft.fftfreq(k, 1.0 / k).astype(int) for k in n]
    H, K, Lh = np.meshgrid(*idx, indexing="ij")
    sign = np.where((H + K + Lh) % 2 == 0, 1.0, -1.0)
    inv_d2 = (H ** 2 + K ** 2 + Lh ** 2) / L ** 2
    F = arr * sign * np.exp(dc.blur * inv_d2 / 4.0)
    return np.fft.fftshift(F).astype(np.complex64), L, c, st


def _trilinear(G, q):
    """Sample a complex grid G (n1, n2, n3; centred: index 0 at n//2) at fractional Miller indices q (..., 3) of the
    box. Returns complex (...)."""
    n = torch.tensor(G.shape, device=q.device, dtype=q.dtype)
    f = q + n // 2
    i0 = torch.floor(f)
    t = f - i0
    i0 = i0.long()
    out = torch.zeros(q.shape[:-1], dtype=G.dtype, device=q.device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = torch.clamp(i0[..., 0] + dx, 0, G.shape[0] - 1)
                iy = torch.clamp(i0[..., 1] + dy, 0, G.shape[1] - 1)
                iz = torch.clamp(i0[..., 2] + dz, 0, G.shape[2] - 1)
                w = ((t[..., 0] if dx else 1 - t[..., 0]) * (t[..., 1] if dy else 1 - t[..., 1])
                     * (t[..., 2] if dz else 1 - t[..., 2]))
                out = out + G[ix, iy, iz] * w.to(G.dtype)
    return out


def _shell_norm(x2, s2, nbins=20):
    edges = np.quantile(s2, np.linspace(0, 1, nbins + 1))
    sh = np.clip(np.searchsorted(edges, s2, side="right") - 1, 0, nbins - 1)
    m = np.bincount(sh, x2, nbins) / np.maximum(np.bincount(sh, minlength=nbins), 1)
    return x2 / np.maximum(m[sh], 1e-30), sh


def rotation_search(data, model, step=6.0, dmin=3.5, dmax=10.0, top=20, dev=None, chunk=2048, log=print, sg=None,
                    fixed=None, tncs=None):
    """Rotation function scores for a rotation grid. Returns (rotations (n, 3, 3), scores (n,)) sorted by score,
    the best `top` refined... (no refinement here; see translation_search).
    fixed: copies already placed (gemmi.Structure or path). Their intensity is removed from the target: averaged
    over the unknown translation, |F_fixed + F_new|^2 = |F_fixed|^2 + |F_new|^2, so the new copy's orientation
    correlates with E_obs^2 - b |E_fixed|^2 (b per resolution shell by least squares).
    tncs: fractional tNCS translation; the target is divided by the tNCS modulation it causes.
    info['rf'] scores any set of rotations (refine_orientation)."""
    dev = dev or device()
    mtz = gemmi.read_mtz_file(str(data))
    labels = [c.label for c in mtz.columns]
    fl = next(l for l in ("F", "FP", "F-obs", "FOBS") if l in labels)
    hkl = np.asarray(mtz.make_miller_array(), np.int64)
    F = np.asarray(mtz.column_with_label(fl), float)
    d = np.asarray(mtz.make_d_array(), float)
    ok = np.isfinite(F) & (F > 0) & (d >= dmin) & (d <= dmax)
    hkl, F, d = hkl[ok], F[ok], d[ok]
    sg = gemmi.SpaceGroup(sg) if isinstance(sg, str) else (sg or mtz.spacegroup)
    eps = np.array([sg.operations().epsilon_factor(list(h)) for h in hkl], float)
    cen = np.array([sg.operations().is_reflection_centric(list(h)) for h in hkl], bool)
    Eo2, _ = _shell_norm(F ** 2 / eps, 1 / d ** 2)
    # the target without anisotropy and tNCS modulation (both are translation-free intensity patterns the
    # orientation of one copy cannot explain), Wilson outliers left out
    from .llg import _aniso_factor, tncs_modulation, wilson_outliers
    Eo2 = Eo2 / _aniso_factor(hkl, mtz.cell, Eo2, cen)
    mod = tncs_modulation(hkl, Eo2, 1 / d ** 2, tncs) if tncs is not None else np.ones(len(Eo2))
    ok = ~wilson_outliers(Eo2 / mod, cen)
    hkl, F, d, eps, cen, Eo2, mod = hkl[ok], F[ok], d[ok], eps[ok], cen[ok], Eo2[ok] / mod[ok], mod[ok]
    if fixed is not None:
        from .llg import model_fcalc
        Ff, _ = model_fcalc(fixed, mtz, hkl, dmin, solvent=False, sg=sg)
        Ef2, shf = _shell_norm(np.abs(np.nan_to_num(Ff)) ** 2 / eps, 1 / d ** 2)
        Ef2 = Ef2 / mod
        a, b = Eo2 - 1.0, Ef2 - 1.0
        slope = np.bincount(shf, a * b) / np.maximum(np.bincount(shf, b * b), 1e-30)
        Eo2 = Eo2 - np.clip(slope, 0.0, None)[shf] * Ef2
    # |F|^2 of the model is the transform of its Patterson (extent 4 x the radius): box >= 4 x radius (pad 2.5)
    G, L, centre, st = model_transform(model, dmin, box_pad=2.5)
    G2 = np.abs(G) ** 2
    # Cartesian reciprocal vectors of the data: s = h B (B: the data cell's fractionalisation^T); the model's box
    # index of a Cartesian vector s is s * L
    Bm = np.array(mtz.cell.frac.mat.tolist())
    s = hkl.astype(float) @ Bm                                     # (n, 3) Cartesian, 1/A
    rots = [np.array(op.rot, float) / gemmi.Op.DEN for op in sg.operations().sym_ops]
    # symmetry rotations in Cartesian space: S_cart = O S_frac O^-1 acting on real-space vectors; on reciprocal
    # vectors s (row) -> s S_cart^T ... use: model orientation in copy k is S_k_cart R
    O = np.array(mtz.cell.orth.mat.tolist())
    Sc = [O @ S @ np.linalg.inv(O) for S in rots]
    Rg = rotation_grid(step)
    log(f"MR rotation search: {len(Rg)} orientations x {len(Sc)} symmetry copies x {len(F)} reflections "
        f"({dmax:g}-{dmin:g} A) on {dev}")
    t0 = time.time()
    Gt = torch.as_tensor(G2.astype(np.float32), device=dev)
    st_ = torch.as_tensor(s, device=dev, dtype=torch.float32)
    eo = torch.as_tensor(Eo2 - Eo2.mean(), device=dev, dtype=torch.float32)
    sh_ = torch.as_tensor(_shell_norm(np.ones_like(F), 1 / d ** 2)[1], device=dev)
    nb = int(sh_.max()) + 1

    def score(Rg):
        out = np.zeros(len(Rg))
        for c0 in range(0, len(Rg), chunk):
            R = torch.as_tensor(Rg[c0:c0 + chunk], device=dev, dtype=torch.float32)        # (c, 3, 3)
            tot = None
            for S in Sc:
                M = torch.as_tensor(S, device=dev, dtype=torch.float32) @ R                # (c, 3, 3)
                # model frame reciprocal vector: s_model = (S R)^T s  (a real-space rotation of the model by S R
                # multiplies its transform argument by (S R)^T)
                q = torch.einsum("cji,nj->cni", M, st_) * L                               # (c, n, 3) box indices
                v = _trilinear(Gt, q)
                tot = v if tot is None else tot + v
            # shell normalisation of the model intensities per orientation, then correlation with E_obs^2
            cnt = torch.bincount(sh_, minlength=nb).float()
            mean = torch.zeros((tot.shape[0], nb), device=dev).index_add_(1, sh_, tot) / cnt
            e = tot / torch.clamp(mean[:, sh_], min=1e-30)
            e = e - e.mean(1, keepdim=True)
            cc = (e * eo).sum(1) / torch.sqrt((e * e).sum(1) * (eo * eo).sum())
            out[c0:c0 + len(R)] = cc.cpu().numpy()
        return out
    scores = score(Rg)
    order = np.argsort(-scores)
    log(f"  {time.time() - t0:.1f}s; best scores " + " ".join(f"{scores[i]:.4f}" for i in order[:5])
        + f"; mean {scores.mean():.4f} sd {scores.std():.4f}")
    return Rg[order], scores[order], dict(L=L, centre=centre, hkl=hkl, s=s, Eo2=Eo2, G=G, Sc=Sc, mtz=mtz, st=st, sg=sg,
                                         rf=score, cen=cen, mod=mod)


def refine_orientation(R, rf, step):
    """Local refinement of an orientation on the rotation function rf (rotation_search's info['rf']): the best of
    3 x 3 x 3 rotation-vector offsets around the current one, the offset halved whenever none is better, from
    step / 2 down to 0.5 deg. Returns (R, score)."""
    grid = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T.astype(float)
    best, s0 = R, float(rf(R[None])[0])
    d = np.radians(step) / 2
    for _ in range(40):
        if d < np.radians(0.5):
            break
        cand = np.array([_rotvec(g * d) @ best for g in grid])
        sc = rf(cand)
        k = int(np.argmax(sc))
        if sc[k] > s0 + 1e-6:
            best, s0 = cand[k], float(sc[k])
        else:
            d /= 2
    return best, s0


def _complex_at(G, M, s, L, dev):
    """F_model at (M^T s) for orientation matrices M (c, 3, 3): complex (c, n)."""
    Gt = torch.as_tensor(G, device=dev)
    q = torch.einsum("cji,nj->cni", torch.as_tensor(M, device=dev, dtype=torch.float32),
                     torch.as_tensor(s, device=dev, dtype=torch.float32)) * L
    return _trilinear(Gt, q)


def translation_search(R, info, grid_step=0.33, dmin=3.5, top=5, dev=None, F_fixed=None):
    """Correlation translation function of orientation R (Cartesian, model frame -> crystal) by FFT. Returns the
    best `top` fractional positions of the model centroid with their T scores (normalised: z-scores of the map).
    F_fixed: complex structure factors of copies already placed (at info['hkl']): |F_fixed + sum_k F_k|^2 adds
    the linear terms 2 Re(F_fixed* F_k e^{2 pi i h.(S_k t + tau_k)}) (index S_k^T h)."""
    dev = dev or device()
    mtz = info["mtz"]
    hkl, s, Eo2 = info["hkl"], info["s"], info["Eo2"]
    sg = info["sg"]
    ops = list(sg.operations())
    O = np.array(mtz.cell.orth.mat.tolist())
    Sf = [np.array(op.rot, float) / gemmi.Op.DEN for op in ops]
    tau = [np.array(op.tran, float) / gemmi.Op.DEN for op in ops]
    Ms = np.array([O @ S @ np.linalg.inv(O) @ R for S in Sf])
    Fk = _complex_at(info["Gc"], Ms, s, info["Lc"], dev).cpu().numpy()                 # (nsym, n)
    # model normalised to E-values in resolution shells (the incoherent sum over the copies has mean 1 per shell):
    # without it the model's B-factors weight the low-resolution terms
    d2 = 1.0 / np.maximum((s ** 2).sum(1), 1e-12)
    _, sh = _shell_norm(np.ones(len(d2)), 1.0 / d2)
    tot = (np.abs(Fk) ** 2).sum(0)
    msh = np.bincount(sh, tot) / np.maximum(np.bincount(sh), 1)
    Fk = Fk / np.sqrt(np.maximum(msh[sh], 1e-30))[None]
    if F_fixed is not None:
        F_fixed = F_fixed / np.sqrt(np.maximum(msh[sh], 1e-30))
    w = Eo2 - Eo2.mean()
    # grid for t: fine enough for the largest pair index (S_j - S_k)^T h
    hmax = np.abs(hkl).max(0)
    n = [int(2 ** np.ceil(np.log2(max(16, 4 * hm + 2)))) for hm in hmax]
    C = np.zeros(n, complex)
    for j in range(len(ops)):
        for k in range(len(ops)):
            if j == k:
                continue
            g = hkl @ (Sf[j] - Sf[k])                       # row: h (S_j - S_k) -> coefficient of t
            ph = np.exp(2j * np.pi * (hkl @ (tau[j] - tau[k])))
            c = w * Fk[j] * np.conj(Fk[k]) * ph
            gi = np.mod(np.round(g).astype(int), n)
            np.add.at(C, (gi[:, 0], gi[:, 1], gi[:, 2]), c)
    if F_fixed is not None:
        for k in range(len(ops)):
            g = hkl @ Sf[k]
            c = 2.0 * w * np.conj(F_fixed) * Fk[k] * np.exp(2j * np.pi * (hkl @ tau[k]))
            gi = np.mod(np.round(g).astype(int), n)
            np.add.at(C, (gi[:, 0], gi[:, 1], gi[:, 2]), c)
    T = np.real(np.fft.ifftn(C)) * np.prod(n)
    # polar axes (first copy in a polar space group, e.g. b in C2): the map is constant along them; collapse them,
    # otherwise the peak list is one peak repeated along the axis
    nz = np.nonzero(np.abs(C) > 1e-12 * max(np.abs(C).max(), 1e-30))
    polar = [a for a in range(3) if len(nz[a]) and np.all(nz[a] == 0)]
    Tc = T
    for a in polar:
        Tc = Tc.max(axis=a, keepdims=True)
    z = (Tc - Tc.mean()) / max(Tc.std(), 1e-30)
    flat = np.argsort(-z.ravel())
    peaks, out = [], []
    keep = np.array([0.0 if a in polar else 1.0 for a in range(3)])
    Om = np.array(mtz.cell.orth.mat.tolist())
    # equivalent positions: the symmetry operators, and for the first copy (nothing fixed yet) the origin shifts
    # (0 / 1/2 along each axis; polar axes are collapsed above)
    origins = [np.zeros(3)] if F_fixed is not None else \
        [np.array(o, float) for o in np.array(np.meshgrid([0, .5], [0, .5], [0, .5])).reshape(3, -1).T]

    def equivalents(p):
        return [Sf[k] @ p + tau[k] + o for k in range(len(ops)) for o in origins]
    for f in flat:
        p = np.array(np.unravel_index(f, z.shape), float) / np.array(n)
        if any(np.linalg.norm(Om @ (((e - q + 0.5) % 1.0 - 0.5) * keep)) < 3.0 for q in peaks for e in equivalents(p)):
            continue
        peaks.append(p)
        out.append((p, float(z.ravel()[f])))
        if len(out) >= top:
            break
    return out


def _quick_llg(Eo, Fc, sh, cen, grid=np.linspace(0.0, 0.95, 20)):
    """sigma_A LLG of candidate structure factors Fc (m, n) against E_o (n): E_c normalised per resolution bin, D
    per bin the best of a grid; no bulk solvent or measurement error (xpg.llg, fast, for ranking placements)."""
    from .llg import _llg_terms
    Ob = np.eye(int(sh.max()) + 1)[sh]                                      # (n, bins) one-hot
    I = np.abs(Fc) ** 2
    Ec = np.sqrt(I / (np.maximum((I @ Ob) / np.maximum(Ob.sum(0), 1), 1e-30) @ Ob.T))
    best = None
    for D in grid:
        tb = _llg_terms(Eo[None], Ec, 0.0, cen[None], D) @ Ob
        best = tb if best is None else np.maximum(best, tb)
    return best.sum(1)


def rank_placements(R, info, peaks, partners, F_fixed=None, n_fixed=0):
    """Placements of orientation R at translation-function peaks, each with the given tNCS partner offsets (None:
    no partner), ranked by the fast LLG: [(LLG, position, partner offset)], best first. The copies' structure
    factors come from the model transform by phase shifts; F_fixed (copies already placed) is scaled to n_fixed
    copies of the search model."""
    mtz, hkl, s, sg = info["mtz"], info["hkl"], info["s"], info["sg"]
    ops = list(sg.operations())
    O = np.array(mtz.cell.orth.mat.tolist())
    Sf = [np.array(op.rot, float) / gemmi.Op.DEN for op in ops]
    tau = [np.array(op.tran, float) / gemmi.Op.DEN for op in ops]
    Fk = _complex_at(info["Gc"], np.array([O @ S @ np.linalg.inv(O) @ R for S in Sf]), s, info["Lc"],
                     None).cpu().numpy()                                    # (nsym, n)
    _, sh = _shell_norm(np.ones(len(hkl)), (s ** 2).sum(1))
    one = np.bincount(sh, (np.abs(Fk) ** 2).sum(0)) / np.maximum(np.bincount(sh), 1)      # one copy per shell
    Ff = 0.0
    if F_fixed is not None and n_fixed:
        mf = np.bincount(sh, np.abs(F_fixed) ** 2) / np.maximum(np.bincount(sh), 1)
        Ff = F_fixed * np.sqrt(n_fixed * one / np.maximum(mf, 1e-30))[sh]
    cand, Fc = [], []
    for p, _ in peaks:
        base = [Fk[k] * np.exp(2j * np.pi * (hkl @ (Sf[k] @ p + tau[k]))) for k in range(len(ops))]
        for tn in partners:
            f = sum(b * (1.0 + np.exp(2j * np.pi * (hkl @ (Sf[k] @ tn)))) if tn is not None else b
                    for k, b in enumerate(base))
            cand.append((p, tn))
            Fc.append(f + Ff)
    Eo2 = info["Eo2"] * (info["mod"] if partners[0] is not None else 1.0)  # a pair carries the tNCS modulation
    ll = _quick_llg(np.sqrt(np.clip(Eo2, 0, None)), np.array(Fc), sh, info["cen"])
    return sorted(((float(v), p, tn) for v, (p, tn) in zip(ll, cand)), key=lambda x: -x[0])


def _moved(info, R, t_frac, tncs=None):
    """The search model rotated by R about its centroid and moved to fractional position t (data cell); with tncs
    also its tNCS partner (same orientation, at t + tncs) as further chains."""
    if tncs is not None:
        st = _moved(info, R, t_frac)
        _add_chains(st, _moved(info, R, np.asarray(t_frac, float) + np.asarray(tncs, float)))
        return st
    st = info["st"].clone()
    mtz = info["mtz"]
    O = np.array(mtz.cell.orth.mat.tolist())
    tc = O @ np.asarray(t_frac, float)
    tr = gemmi.Transform()
    tr.mat.fromlist(np.asarray(R, float).tolist())
    tr.vec.fromlist((tc - np.asarray(R, float) @ info["centre"]).tolist())
    st[0].transform_pos_and_adp(tr)
    st.cell = mtz.cell
    st.spacegroup_hm = info["sg"].hm
    return st


def _add_chains(out, st):
    """Append the chains of st to out under unused chain names."""
    used = {ch.name for ch in out[0]}
    names = iter(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" if c not in used)
    for ch in st[0]:
        ch2 = ch.clone()
        ch2.name = next(names)
        used.add(ch2.name)
        out[0].add_chain(ch2)
    out.setup_entities()
    return out


def place_model(info, R, t_frac, path, fixed=None, tncs=None):
    """Write the search model rotated by R about its centroid and moved to fractional position t (data cell), with
    its tNCS partner if tncs; fixed: a structure of copies already placed, written with it (new chain names)."""
    st = _moved(info, R, t_frac, tncs)
    if fixed is not None:
        st = _add_chains(fixed.clone(), st)
    st.write_pdb(str(path))
    return path


def space_group_alternatives(sg):
    """Space groups of the same point group, centring and axis setting (screw-axis / enantiomorph alternatives that
    intensities cannot always tell apart), one per absence pattern (origin-shifted duplicates dropped)."""
    sg = gemmi.SpaceGroup(sg) if isinstance(sg, str) else sg
    rots = sorted(tuple(map(tuple, op.rot)) for op in sg.operations().sym_ops)
    axial = [h for i in range(3) for n in range(1, 13) for h in [tuple(n if j == i else 0 for j in range(3))]]
    out, seen = [], set()
    for g in gemmi.spacegroup_table():
        if not g.is_sohncke() or g.centring_type() != sg.centring_type() or g.point_group_hm() != sg.point_group_hm():
            continue
        if sorted(tuple(map(tuple, op.rot)) for op in g.operations().sym_ops) != rots:
            continue
        ops = g.operations()
        key = (g.number, tuple(ops.is_systematically_absent(list(h)) for h in axial))
        if key in seen:
            continue
        seen.add(key)
        out.append(gemmi.SpaceGroup(g.xhm()))
    return out


def clash_fraction(path, cutoff=3.0):
    """Packing check: fraction of the CA atoms within `cutoff` A of a CA of another molecule (other chains and all
    symmetry images, the chain's own included)."""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    n = 0
    for ch in st[0]:
        for r in ch:
            for i in range(len(r) - 1, -1, -1):
                if r[i].name != "CA":
                    del r[i]
            n += len(r)
    ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate()
    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.SameChain
    return len(cs.find_contacts(ns)) / max(n, 1)


def _angle_sym(R, Q, Sc):
    """Smallest rotation angle (deg) between orientation R and the crystal-symmetry copies S Q of Q."""
    return min(np.degrees(np.arccos(np.clip((np.trace(R.T @ (S @ Q)) - 1) / 2, -1, 1))) for S in Sc)


def _rotvec(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rigid_body_refine(data, info, R, t_frac, path, dmin=3.5, log=print, fixed=None, tncs=None, twin=None):
    """Six-parameter rigid-body refinement of a placed model by maximising the sigma_A LLG (xpg.llg) at dmin:
    rotation vector about the model centroid (rad) and shift (A). Powell, from the search solution. The target
    keeps the data side and the bulk-solvent model of the starting placement, and the copies already placed enter
    as fixed structure factors: each evaluation computes only the moving copy's (the final LLG is recomputed in
    full). tncs: the moving unit is a tNCS pair (both copies turn about their own centroids and keep the tNCS
    translation); twin: the twin law and fraction of the target."""
    from scipy.optimize import minimize
    from .llg import LLGTarget, model_fcalc, placed_model_llg
    O = np.array(info["mtz"].cell.orth.mat.tolist())
    Oi = np.linalg.inv(O)
    place_model(info, R, t_frac, path, fixed=fixed, tncs=tncs)
    tgt = LLGTarget(data, path, dmin=dmin, sg=info["sg"], tncs=info.get("tncs"), twin=twin)
    F_fix = (np.nan_to_num(model_fcalc(fixed, tgt.mtz, tgt.hkl, tgt.dmin_, solvent=False, sg=tgt.sg)[0])
             if fixed is not None else 0.0)
    cache = {}

    def f(p):
        key = tuple(np.round(p, 6))
        if key not in cache:
            st = _moved(info, _rotvec(p[:3]) @ R, np.asarray(t_frac) + Oi @ p[3:], tncs)
            Fm = model_fcalc(st, tgt.mtz, tgt.hkl, tgt.dmin_, solvent=False, sg=tgt.sg)[0]
            cache[key] = -tgt.evaluate(F_fix + np.nan_to_num(Fm))
        return cache[key]
    f0 = -f(np.zeros(6))
    r = minimize(f, np.zeros(6), method="Powell",
                 options=dict(xtol=1e-3, ftol=1e-4, maxfev=400, direc=np.diag([0.02, 0.02, 0.02, 0.5, 0.5, 0.5])))
    R2, t2 = _rotvec(r.x[:3]) @ R, np.asarray(t_frac) + Oi @ r.x[3:]
    place_model(info, R2, t2, path, fixed=fixed, tncs=tncs)
    llg = placed_model_llg(data, path, dmin=dmin, sg=info["sg"], tncs=info.get("tncs"), twin=twin)["LLG"]
    log(f"    rigid body: LLG({dmin:g} A) {f0:.1f} -> {-r.fun:.1f} ({r.nfev} evaluations; rotation "
        f"{np.degrees(np.linalg.norm(r.x[:3])):.2f} deg, shift {np.linalg.norm(r.x[3:]):.2f} A); full {llg:.1f}")
    return R2, t2, llg


def search(data, model, step=6.0, dmin=3.5, n_rot=5, n_trans=3, dev=None, log=print, workdir=None, refine=2,
           copies=1, sg=None, max_clash=0.10, tncs=None, twin=None):
    """Rotation search -> translation search for the best distinct orientations -> sigma_A LLG of each placed
    model (xpg.llg), copy after copy; at the end every copy rigid-body refined to 2 A. Returns [solution] (dict R,
    t, rot score, trans z, LLG, LLGZ: the Z of the last copy placed, pdb).
    tncs: fractional tNCS translation: copies are placed as pairs (copies // 2 pairs, then single copies);
    twin: dict(law, alpha) for the LLG."""
    import tempfile
    from pathlib import Path
    from .llg import placed_model_llg
    from .llg import model_fcalc
    Rs, sc, info = rotation_search(data, model, step, dmin, dev=dev, log=log, sg=sg, tncs=tncs)
    info["Gc"], info["Lc"], _, _ = model_transform(model, dmin, box_pad=2.5)
    info["tncs"] = tncs
    Sc = info["Sc"]

    def llg_of(pdb, dm=dmin):
        return placed_model_llg(data, pdb, dmin=dm, sg=info["sg"], tncs=tncs, twin=twin)["LLG"]

    def pick(Rs, sc, n, rf):
        """The n best distinct orientations (up to the crystal's rotations: R and S_k R score the same), each
        refined locally on the rotation function."""
        out = []
        for R, s_ in zip(Rs, sc):
            if all(_angle_sym(R, Q, Sc) > 1.5 * step for Q, _ in out):
                out.append((R, s_))
            if len(out) >= n:
                break
        ref = []
        for R, s_ in (refine_orientation(R, rf, step) for R, _ in out):
            if all(_angle_sym(R, Q, Sc) > 2.0 for Q, _ in ref):
                ref.append((R, s_))
        return ref
    rf = info["rf"]
    picked = pick(Rs, sc, max(n_rot, copies + 2), rf)
    wd = Path(workdir or tempfile.mkdtemp(prefix="xpg_mr_"))
    wd.mkdir(parents=True, exist_ok=True)
    fixed, F_fixed, best = None, None, None
    # search units: tNCS pairs first, then single copies
    units = [2] * (copies // 2) + [1] * (copies % 2) if tncs is not None else [1] * copies

    def candidates(copy, picked, n_tr, pair, skip=0):
        """Placements of the picked orientations (from the skip-th): the n_tr best of their translation peaks by the
        fast LLG, without packing clashes, by LLG; rigid-body refined best first. Each placement's Z: its fast LLG
        against that of random positions of the same orientation (the z-score of an LLG translation function)."""
        sols = []
        rnd = [(q, 0.0) for q in np.random.default_rng(0).uniform(0.0, 1.0, (300, 3))]
        what = "pair" if pair else "copy"
        partners = [np.asarray(tncs, float), -np.asarray(tncs, float)] if pair else [None]
        for i, (R, s_) in enumerate(picked[skip:], skip):
            peaks = translation_search(R, info, dmin=dmin, top=n_peaks, dev=dev, F_fixed=F_fixed)
            z_of = {tuple(np.round(p, 6)): z for p, z in peaks}
            ranked = rank_placements(R, info, peaks, partners, F_fixed, placed)
            null = np.array([r[0] for r in rank_placements(R, info, rnd, partners[:1], F_fixed, placed)])
            mu, sd = null.mean(), max(null.std(), 1e-6)
            for j, (q, tf, tn) in enumerate(ranked[:n_tr]):
                z = z_of[tuple(np.round(tf, 6))]
                sign = "" if tn is None else ("+t" if np.allclose(tn, partners[0]) else "-t")
                pdb = place_model(info, R, tf, wd / f"{what}{copy + 1}_{i}_{j}{sign}.pdb", fixed=fixed, tncs=tn)
                cf = clash_fraction(pdb)
                if cf > max_clash:
                    log(f"  {what} {copy + 1}: orientation {i + 1} placement {j + 1}{sign}: TF z {z:5.2f}, "
                        f"packing clashes {100 * cf:.0f}% -> skipped")
                    continue
                llg = llg_of(pdb)
                sols.append(dict(R=R, t=tf, rot=float(s_), tz=z, LLG=llg, pdb=str(pdb), tn=tn,
                                 LLGZ=float((q - mu) / sd)))
                log(f"  {what} {copy + 1}: orientation {i + 1} (RF {s_:.4f}) placement {j + 1}{sign}: TF z {z:5.2f}, "
                    f"LLG Z {(q - mu) / sd:5.1f}, LLG({dmin:g} A) {llg:9.1f}")
        if not sols:
            return sols
        sols.sort(key=lambda d: -d["LLG"])
        for s_ in sols[:refine]:
            s_["R"], s_["t"], s_["LLG"] = rigid_body_refine(data, info, s_["R"], s_["t"], s_["pdb"], dmin=dmin, log=log,
                                                            fixed=fixed, tncs=s_["tn"], twin=twin)
        sols.sort(key=lambda d: -d["LLG"])
        return sols

    placed, pairs, layout = 0, 0, []
    n_peaks = 40                         # translation-function peaks per orientation ranked by the fast LLG
    for copy, unit in enumerate(units):
        pair = unit == 2
        Rs_c, sc_c = Rs, sc
        if fixed is not None:
            # rotation function against the copies already placed (their intensity removed from the target): weak
            # copies rank far down the plain function more orientations and translation peaks than for the first copy:
            # the remaining copies are weaker and peaks of misfits pile up against the placed ones
            Rs_c, sc_c, inf_c = rotation_search(data, model, step, dmin, dev=dev, log=log, sg=sg, fixed=fixed,
                                                tncs=tncs)
            rf = inf_c["rf"]
            picked = pick(Rs_c, sc_c, max(n_rot, len(units) - copy + 2, 10), rf)
        n_tr = n_trans if fixed is None else max(n_trans, 4)
        sols = candidates(copy, picked, n_tr, pair)
        ok = bool(sols) and (best is None or sols[0]["LLG"] > best["LLG"])
        # the first copy always takes its best candidate: when that does not stand out from random positions (Z
        # below 8, the level wrong models do not reach), the true placement may not be among the peaks tested
        weak_first = best is None and bool(sols) and not sols[0].get("LLGZ", 0.0) >= 8.0
        if not ok or weak_first:
            # before giving up on a copy: twice the orientations and translation peaks
            wide = pick(Rs_c, sc_c, 2 * len(picked), rf)
            why = ("no placement without packing clashes" if not sols else "no LLG gain" if not ok else
                   f"best placement does not stand out (Z {sols[0].get('LLGZ', float('nan')):.1f})")
            log(f"  {'pair' if pair else 'copy'} {copy + 1}: {why}; widening the search to {len(wide)} orientations x "
                f"{2 * n_tr} translation peaks")
            more = candidates(copy, wide, 2 * n_tr, pair) + sols
            more.sort(key=lambda d: -d["LLG"])
            sols = more
            ok = bool(sols) and (best is None or sols[0]["LLG"] > best["LLG"])
        if not sols:
            log(f"  {'pair' if pair else 'copy'} {copy + 1}: no placement without packing clashes; stop at {placed} "
                f"copies")
            break
        b = sols[0]
        if not ok:
            what = "pair" if pair else "copy"
            log(f"  {what} {copy + 1}: no LLG gain ({b['LLG']:.1f} <= {best['LLG']:.1f}); stop at {placed} copies")
            break
        best = b                         # its Z: that of the last unit placed (with the others fixed)
        placed += unit
        pairs += pair
        best["copies"] = placed
        layout += [(b["R"], b["t"])] + ([(b["R"], np.asarray(b["t"]) + b["tn"])] if b["tn"] is not None else [])
        fixed = gemmi.read_structure(b["pdb"])
        fixed.setup_entities()
        if copy + 1 < len(units):
            mtz = info["mtz"]
            F_fixed, _ = model_fcalc(b["pdb"], mtz, info["hkl"], dmin, solvent=False, sg=info["sg"])
            F_fixed = np.nan_to_num(F_fixed)
    if best is None:
        raise RuntimeError("no placement without packing clashes")
    # every copy (tNCS partners on their own) rigid-body refined against the data to 2 A, the others fixed
    d_ref = max(float(info["mtz"].resolution_high()), 2.0)
    log(f"  final rigid-body refinement of each copy to {d_ref:.2f} A")
    for i in range(len(layout)):
        others = None
        for k, (Rk, tk) in enumerate(layout):
            if k != i:
                others = _moved(info, Rk, tk) if others is None else _add_chains(others, _moved(info, Rk, tk))
        R2, t2, _ = rigid_body_refine(data, info, *layout[i], wd / "final.pdb", dmin=d_ref, log=log, fixed=others,
                                      twin=twin)
        layout[i] = (R2, t2)
    best["pdb"] = str(wd / "final.pdb")
    best["LLG"] = llg_of(best["pdb"])
    best["LLG_all"] = llg_of(best["pdb"], None)
    best["sg"] = info["sg"].hm
    best["tncs_pairs"] = pairs
    log(f"  best: {best['copies']} copies, LLG({dmin:g} A) {best['LLG']:.1f}, all data {best['LLG_all']:.1f}")
    return [best]


def search_space_groups(data, model, sgs=None, copies=1, step=6.0, dmin=3.5, n_rot=4, n_trans=2, dev=None,
                        log=print, workdir=None, tncs=None, twin=None):
    """Space-group scan: one rotation search (the rotation parts are the same), then
    per alternative a translation search + LLG of the first copy (tNCS pair) for the best orientations; the full
    search (all copies, rigid-body refinement) in the best alternative. Returns (best solution, {sg: LLG of the
    first unit})."""
    import tempfile
    from pathlib import Path
    from .llg import placed_model_llg
    mtz = gemmi.read_mtz_file(str(data))
    sgs = [gemmi.SpaceGroup(s) if isinstance(s, str) else s for s in (sgs or space_group_alternatives(mtz.spacegroup))]
    if copies < 2:
        tncs = None                      # a single copy has no tNCS partner
    Rs, sc, info = rotation_search(data, model, step, dmin, dev=dev, log=log, sg=sgs[0], tncs=tncs)
    info["Gc"], info["Lc"], _, _ = model_transform(model, dmin, box_pad=2.5)
    picked = []
    for R, s_ in zip(Rs, sc):
        if all(_angle_sym(R, Q, info["Sc"]) > 1.5 * step for Q, _ in picked):
            picked.append((R, s_))
        if len(picked) >= n_rot:
            break
    picked = [refine_orientation(R, info["rf"], step) for R, _ in picked]
    wd = Path(workdir or tempfile.mkdtemp(prefix="xpg_mr_"))
    wd.mkdir(parents=True, exist_ok=True)
    table = {}
    for g in sgs:
        inf = dict(info, sg=g)
        best = -np.inf
        partners = [np.asarray(tncs, float), -np.asarray(tncs, float)] if tncs is not None else [None]
        for i, (R, _) in enumerate(picked):
            peaks = translation_search(R, inf, dmin=dmin, top=40, dev=dev)
            for j, (_, tf, tn) in enumerate(rank_placements(R, inf, peaks, partners)[:n_trans]):
                pdb = place_model(inf, R, tf, wd / f"sg_{g.hm.replace(' ', '')}_{i}_{j}.pdb", tncs=tn)
                if clash_fraction(pdb) > 0.10:
                    continue
                best = max(best, placed_model_llg(data, pdb, dmin=dmin, sg=g, tncs=tncs, twin=twin)["LLG"])
        table[g.xhm()] = float(best)
        log(f"  {g.xhm():14s} {'pair' if tncs is not None else 'copy'} 1 LLG({dmin:g} A) {best:9.1f}")
    top = max(table, key=table.get)
    log(f"  best space group {top}; full search with {copies} cop{'y' if copies == 1 else 'ies'}"
        + (" (tNCS pairs)" if tncs is not None else ""))
    sol = search(data, model, step, dmin, n_rot=max(n_rot, copies + 2), n_trans=n_trans, dev=dev, log=log,
                 workdir=wd / top.replace(" ", ""), copies=copies, sg=top, tncs=tncs, twin=twin)[0]
    return sol, table


def main(argv=None):
    ap = argparse.ArgumentParser(description="GPU molecular replacement: rotation + translation search, LLG scoring")
    ap.add_argument("data")
    ap.add_argument("model")
    ap.add_argument("--step", type=float, default=6.0)
    ap.add_argument("--dmin", type=float, default=3.5)
    ap.add_argument("--rotations", type=int, default=5)
    ap.add_argument("--translations", type=int, default=3)
    ap.add_argument("--out", default=None, help="directory for the placed models")
    ap.add_argument("--copies", type=int, default=1)
    a = ap.parse_args(argv)
    t0 = time.time()
    sols = search(a.data, a.model, a.step, a.dmin, a.rotations, a.translations, workdir=a.out, copies=a.copies)
    b = sols[0]
    print(f"best: LLG({a.dmin:g} A) {b['LLG']:.1f}, all data {b.get('LLG_all', float('nan')):.1f} (RF {b['rot']:.4f}, "
          f"TF z {b['tz']:.2f}) -> {b['pdb']}; {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

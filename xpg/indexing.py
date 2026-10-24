"""Auto-indexing using a GPU 3D-FFT of the reciprocal-space spot map.

1. spots -> reciprocal-lattice vectors p0 (phi = 0 frame)
2. maximum cell estimated from nearest-neighbour distances in reciprocal space
3. p0 density gridded and Fourier transformed (torch.fft) -> real-space lattice-vector map
4. candidate real-space vectors from map peaks, refined against the spot set
5. best non-coplanar triplet = smallest volume basis indexing (nearly) all spots
6. Niggli reduction (gemmi GruberVector), refinement, re-indexing; iterate to full resolution
7. optional search for additional lattices in the unindexed spots
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import gemmi
import numpy as np
import torch
from scipy.spatial import cKDTree

from .backend import device
from .geometry import Geometry, cell_from_Astar, real_basis
from .models import Crystal, Experiment
from .refine import refine
from .spotfind import SpotList


@dataclass
class IndexingResult:
    exp: Experiment
    hkl: np.ndarray           # per spot (0 0 0 for unindexed)
    indexed: np.ndarray       # bool per spot
    fraction: float
    rmsd_xy: float
    rmsd_phi: float
    extra_lattices: list
    domains: list = field(default_factory=list)   # weak near-copies of the main lattice (integrated as one composite)


def estimate_max_cell(p0, phi_deg, osc, percentile=0.05, multiplier=1.5, min_sep=None):
    """Max cell dimension from nearest-neighbour distances between reciprocal-space spots.

    Pairs closer than the positional uncertainty of a spot (``min_sep``, 1/A: a few pixels on the detector
    plus about one frame of rotation) are the same reflection found twice (split spots, spots straddling
    frames) and are ignored; otherwise they drive the estimate to absurd cells (and the FFT to low resolution)."""
    tree = cKDTree(p0)
    d, j = tree.query(p0, k=8)
    if min_sep is None:
        min_sep = np.zeros(len(p0))
    rad = np.linalg.norm(p0, axis=1) * np.radians(1.5 * abs(osc))
    floor = np.sqrt(np.asarray(min_sep) ** 2 + rad ** 2)
    nn = np.full(len(p0), np.nan)
    for c in range(1, d.shape[1]):
        ok = np.isnan(nn) & np.isfinite(d[:, c]) & (d[:, c] > floor)
        nn[ok] = d[ok, c]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size < 10:
        return 300.0
    lo = np.quantile(nn, percentile)
    return float(min(multiplier / lo, 2000.0))


def estimate_max_cell_detector(geom, spots, sel=None, percentile=0.05, multiplier=1.5, min_px=4.0):
    """Max cell from spot separations on the detector: distinct reflections recorded in the same frame are
    >= about lambda D / (a_max q) pixels apart (near the beam). Pairs closer than ``min_px`` are one
    reflection found twice and are ignored. More robust than reciprocal-space neighbours, which are
    dominated by split spots on fine-sliced data."""
    sel = np.arange(len(spots)) if sel is None else np.asarray(sel)
    x, y, z = spots.x[sel], spots.y[sel], spots.z[sel]
    # neighbours are looked for within ~0.5 deg of rotation (at least 0.1 frame): on finely sliced data a
    # single frame holds too few reflections and the separations (hence the cell) come out too small/large
    zscale = 50.0 * min(1.0, abs(geom.osc) / 0.5)
    pts = np.c_[x, y, z * zscale]
    d, _ = cKDTree(pts).query(pts, k=6)
    nn = np.full(len(x), np.nan)
    for c in range(1, d.shape[1]):
        ok = np.isnan(nn) & np.isfinite(d[:, c]) & (d[:, c] > min_px) & (d[:, c] < 50.0)
        nn[ok] = d[ok, c]
    nn = nn[np.isfinite(nn)]
    if nn.size < 50:
        return None
    q = np.quantile(nn, percentile)
    return float(multiplier * geom.wavelength * geom.distance / (q * geom.qx))


def _peaks_rfft(A):
    """Local maxima of A over a 3x3x3 neighbourhood (separable max filter, periodic)."""
    m = A
    for ax in (0, 1, 2):
        m = torch.maximum(m, torch.maximum(torch.roll(m, 1, ax), torch.roll(m, -1, ax)))
    return A >= m


LAST_MAX_CELL_RAW = None   # uncapped max-cell estimate of the last index_spots call (retry logic in the pipeline)
LAST_MAX_CELL_RECIP = None  # its reciprocal-space part alone (an upper bracket: split spots inflate it)


def fft_candidate_vectors(p, dmin, max_cell, n_max=None, n_peaks=60):
    """Candidate real-space lattice vectors (Angstrom) from the FFT of the spot density. The grid resolution limits
    the spots used to d >= 4 max_cell / n: a 256^3 grid with a 672 A cell estimate keeps only d >= 10.5 A -> 512^3
    (0.5 GB) beyond the default 400 A cap (the large-cell retry; d >= 5.3 A). Very long axes still smear on a 3D
    grid: fft1d_candidate_vectors."""
    if n_max is None:
        n_max = 512 if max_cell > 400.0 else 256
    n = int(2 ** np.ceil(np.log2(max(64, 4.0 * max_cell / dmin))))
    if n > n_max:
        dmin = 4.0 * max_cell / n_max
        n = n_max
        p = p[np.einsum("ij,ij->i", p, p) <= 1.0 / dmin ** 2]
    scale = n / (2.0 / dmin)
    idx = np.round(p * scale).astype(np.int64) % n
    dev = device()
    ti_ = torch.as_tensor(idx, device=dev)
    grid = torch.zeros((n, n, n), dtype=torch.float32, device=dev)
    grid.index_put_((ti_[:, 0], ti_[:, 1], ti_[:, 2]), torch.ones(len(idx), dtype=torch.float32, device=dev),
                    accumulate=True)
    F = torch.fft.rfftn(grid).abs()  # shape (n, n, n//2+1); real-space vector t = k / (n/scale)...
    F[0, 0, 0] = 0.0
    isp = _peaks_rfft(F)
    vals = torch.where(isp, F, 0.0).ravel()
    k = min(int(n_peaks * 4), vals.numel())
    top_v, top_i = torch.topk(vals, k)
    top_i = top_i.cpu().numpy()
    top_v = top_v.cpu().numpy()
    i0, i1, i2 = np.unravel_index(top_i, F.shape)
    kk = np.stack([i0, i1, i2], -1).astype(float)
    kk[kk[:, 0] > n / 2, 0] -= n
    kk[kk[:, 1] > n / 2, 1] -= n
    # real-space vector: exp(2 pi i k.idx/n) with idx = p*scale -> t = k * scale / n
    t = kk * scale / n
    L = np.linalg.norm(t, axis=1)
    ok = (L > dmin) & (L < 1.2 * max_cell) & (top_v > 0)
    return t[ok], top_v[ok], dmin


def fft1d_candidate_vectors(p, dmin, max_cell, step_deg=2.0, n_peaks=60):
    """Candidate real-space vectors from 1D FFTs of the spot projections on directions over a hemisphere (as in
    DIALS fft1d): a fine 1D grid resolves long axes that a 3D grid cannot. Returns (vectors, scores, dmin) like
    fft_candidate_vectors."""
    p = p[np.einsum("ij,ij->i", p, p) <= 1.0 / dmin ** 2]
    n_dir = int(2 * np.pi / np.radians(step_deg) ** 2)
    i = np.arange(n_dir) + 0.5
    zc = 1.0 - i / n_dir                                  # hemisphere (z >= 0), Fibonacci lattice
    ph = np.pi * (1 + 5 ** 0.5) * i
    rr = np.sqrt(np.maximum(1 - zc * zc, 0))
    U = np.stack([rr * np.cos(ph), rr * np.sin(ph), zc], 1)
    nb = int(2 ** np.ceil(np.log2(8.0 * max_cell / dmin)))  # 4 bins per period of the longest allowed axis
    span = 2.0 / dmin
    dev = device()
    pj = torch.as_tensor(p, dtype=torch.float32, device=dev)
    kmin = max(int(np.ceil(5.0 * span)), 2)               # axes >= 5 A
    kmax = min(int(1.2 * max_cell * span), nb // 2 - 1)
    best_k, best_v = [], []
    for c0 in range(0, n_dir, 256):
        Uc = torch.as_tensor(U[c0:c0 + 256], dtype=torch.float32, device=dev)
        x = (Uc @ pj.T + 1.0 / dmin) / span * nb          # (D, N) in bins
        bi = torch.clamp(x.long(), 0, nb - 1)
        H = torch.zeros((Uc.shape[0], nb), dtype=torch.float32, device=dev)
        H.scatter_add_(1, bi, torch.ones_like(x))
        F = torch.fft.rfft(H, dim=1).abs()[:, kmin:kmax + 1]
        v, k = F.max(dim=1)
        best_k.append(k.cpu().numpy() + kmin)
        best_v.append(v.cpu().numpy())
    best_k, best_v = np.concatenate(best_k), np.concatenate(best_v)
    L = best_k / span                                     # real-space length of the period (A)
    T = U * L[:, None]
    order = np.argsort(-best_v)
    keep = []
    for j in order:                                       # angular non-maximum suppression (5 deg)
        if all(abs(U[j] @ U[q]) < np.cos(np.radians(5.0)) for q in keep):
            keep.append(j)
        if len(keep) >= n_peaks:
            break
    keep = np.array(keep, int)
    return T[keep], best_v[keep], dmin


def refine_vector(t, p, n_iter=4, tol=0.2):
    """Least-squares refinement of a real-space vector so that p.t is integral. The tolerance tightens so that, with
    several lattices present, the vector is fitted to its own lattice only."""
    tols = np.linspace(tol, 0.1, n_iter)
    for it in range(n_iter):
        h = p @ t
        r = h - np.round(h)
        sel = np.abs(r) < tols[it]
        if sel.sum() < 6:
            break
        A = p[sel]
        b = np.round(h[sel])
        t_new, *_ = np.linalg.lstsq(A, b, rcond=None)
        t = t_new
    h = p @ t
    score = np.mean(np.abs(h - np.round(h)) < 0.15)
    return t, score


def _remove_multiples(ts, scores, angle_tol=5.0):
    """Keep shortest representative for collinear vectors (removes integer multiples)."""
    order = np.argsort(np.linalg.norm(ts, axis=1))
    keep = []
    for i in order:
        u = ts[i] / np.linalg.norm(ts[i])
        dup = False
        for j in keep:
            v = ts[j] / np.linalg.norm(ts[j])
            if abs(np.dot(u, v)) > np.cos(np.radians(angle_tol)):
                ratio = np.linalg.norm(ts[i]) / np.linalg.norm(ts[j])
                if abs(ratio - round(ratio)) < 0.1 or scores[j] >= scores[i] - 0.02:
                    dup = True
                    break
        if not dup:
            keep.append(i)
    return ts[keep], scores[keep]


def best_basis(ts, p, tol=0.15, max_vectors=30, rank=0):
    """Choose a non-coplanar triplet of real-space vectors maximising the number of indexed
    spots, tie-broken by minimal cell volume. Vectorised over all triplets (torch)."""
    ts = ts[:max_vectors]
    m = len(ts)
    trip = np.array(list(itertools.combinations(range(m), 3)))
    if trip.size == 0:
        return None
    B = ts[trip]                          # (T, 3, 3) rows = basis vectors
    vol = np.abs(np.linalg.det(B))
    L = np.linalg.norm(B, axis=2)
    minvol = 0.1 * L.prod(1)              # reject near-coplanar (angles too flat)
    ok = vol > minvol
    trip, B, vol = trip[ok], B[ok], vol[ok]
    if len(B) == 0:
        return None
    dev = device()
    Bj = torch.as_tensor(B, dtype=torch.float32, device=dev)
    pj = torch.as_tensor(p, dtype=torch.float32, device=dev)

    @torch.no_grad()
    def count(Bc):
        h = torch.einsum("tij,nj->tni", Bc, pj)
        r = (h - torch.round(h)).abs()
        return (r < tol).all(-1).sum(-1)

    counts = np.concatenate([count(Bj[i:i + 512]).cpu().numpy() for i in range(0, len(B), 512)])
    best = counts.max()
    good = counts >= 0.95 * best
    cand = np.flatnonzero(good)
    i = cand[np.argmin(vol[cand])]
    if rank == 0:
        return B[i], counts[i] / len(p)
    # alternatives (retries on narrow wedges, where a small spurious cell can win): distinct cell volumes,
    # ranked by count
    order = np.argsort(-counts)
    picked = [vol[i]]
    for j in order:
        if all(abs(vol[j] / v - 1) > 0.1 for v in picked):
            picked.append(vol[j])
            if len(picked) == rank + 1:
                return B[j], counts[j] / len(p)
    return None


def niggli_reduce(R):
    """Niggli-reduce a real-space basis (rows). Returns (R_reduced, M) with R_red = M @ R."""
    cell = gemmi.UnitCell(*cell_from_Astar(np.linalg.inv(R)))
    gv = gemmi.GruberVector(cell, None, track_change_of_basis=True)
    gv.niggli_reduce()
    op = gv.change_of_basis
    # gemmi Op rot (for real-space basis change) in units of DEN; new basis rows = M @ old rows
    M = np.array(op.rot, float) / gemmi.Op.DEN
    Rn = None
    # determine convention numerically: choose the transformation that reproduces the reduced cell
    red = np.array(gv.cell_parameters())
    for cand in (M, M.T, np.linalg.inv(M), np.linalg.inv(M).T):
        Rc = cand @ R
        if np.allclose(_cell_of_rows(Rc), red, atol=1e-3 * max(red[:3]) + 1e-3):
            Rn = Rc
            M = cand
            break
    if Rn is None:
        return R, np.eye(3)
    Mi = np.round(M).astype(int)
    Rn = Mi @ R
    if np.linalg.det(Rn) < 0:
        Mi = -Mi
        Rn = -Rn
    return Rn, Mi


def _cell_of_rows(R):
    a, b, c = np.linalg.norm(R, axis=1)
    al = np.degrees(np.arccos(R[1] @ R[2] / (b * c)))
    be = np.degrees(np.arccos(R[0] @ R[2] / (a * c)))
    ga = np.degrees(np.arccos(R[0] @ R[1] / (a * b)))
    return np.array([a, b, c, al, be, ga])


def _index_score(geom, spots_xyz, R, tol=0.12, n_refit=3):
    """Fraction of spots indexed by real-space basis R (rows) after linear re-fit of R."""
    x, y, z = spots_xyz
    p = geom.spot_to_p0(x, y, z)
    for _ in range(n_refit):
        h = p @ R.T
        hr = np.round(h)
        ok = np.all(np.abs(h - hr) < 0.3, axis=1) & np.any(hr != 0, axis=1)
        if ok.sum() < 12:
            return 0.0, R
        R = np.linalg.lstsq(p[ok], hr[ok], rcond=None)[0].T
    h = p @ R.T
    resid = np.abs(h - np.round(h))
    good = np.all(resid < tol, axis=1)
    return float(good.mean()), R


def beam_centre_search(geom, spots, R, dmin=4.0, n_spots=3000, verbose=True, log=print):
    """INDEX_ORIGIN / DIALS search_beam_position equivalent.

    A beam-centre error of about one reflection spacing makes the indices off by one while the lattice
    still looks well indexed. Scan beam-centre offsets over +-1.5 reflection spacings (at the
    detector, for the longest real-space axis), re-map the spots, re-fit the basis linearly and
    score the indexed fraction. Returns (dx, dy, score0, best_score, table).
    """
    d = geom.resolution_of_pixel(spots.x, spots.y)
    sel = np.flatnonzero(d >= dmin)
    if sel.size < 50:
        return 0.0, 0.0, None, None, []
    if sel.size > n_spots:
        sel = np.random.default_rng(0).choice(sel, n_spots, replace=False)
    xyz = (spots.x[sel], spots.y[sel], spots.z[sel])
    amax = np.linalg.norm(R, axis=1).max()
    spacing_px = geom.wavelength * geom.distance / amax / geom.qx
    rng_px = 1.5 * spacing_px
    step = max(spacing_px / 6.0, 0.5)
    offs = np.arange(-rng_px, rng_px + 1e-9, step)
    score0, _ = _index_score(geom, xyz, R)
    table = []
    for dx in offs:
        for dy in offs:
            g2 = geom.copy(orgx=geom.orgx + dx, orgy=geom.orgy + dy)
            sc, _ = _index_score(g2, xyz, R)
            table.append((sc, dx, dy))
    table.sort(reverse=True)
    best, bdx, bdy = table[0]
    # fine search around the best coarse point
    for dx in np.linspace(bdx - step, bdx + step, 7):
        for dy in np.linspace(bdy - step, bdy + step, 7):
            sc, _ = _index_score(geom.copy(orgx=geom.orgx + dx, orgy=geom.orgy + dy), xyz, R)
            if sc > best:
                best, bdx, bdy = sc, dx, dy
    if verbose:
        log(f"  INDEX: beam-centre / index-origin search (+-{rng_px:.1f} px, one spacing = {spacing_px:.1f} px): "
            f"indexed fraction {score0:.3f} at header position, best {best:.3f} at offset ({bdx:+.2f}, {bdy:+.2f}) px")
    return bdx, bdy, score0, best, table


def segment_offset_search(geom, spots, Astar, dmin=None, n_spots=20000, min_spots=40, log=print):
    """Per-segment translation search for multi-segment detectors (cf. REFINE(IDXREF)=SEGMENT, which
    only converges for sub-pixel errors). With the crystal orientation fixed from all spots, each segment
    is shifted over +-1.5 reflection spacings and the shift maximising the fraction of its spots indexed
    (|h - round(h)| < 0.12) is kept if it improves that fraction by > 0.1 and reaches 0.5.
    Returns (new geometry or None, list of (segment, dx, dy, before, after))."""
    if not geom.segmented:
        return None, []
    R = np.linalg.inv(Astar)
    amax = np.linalg.norm(R, axis=1).max()
    spacing_px = geom.wavelength * geom.distance / amax / geom.qx
    step = max(spacing_px / 8.0, 0.25)
    offs = np.arange(-1.5 * spacing_px, 1.5 * spacing_px + 1e-9, step)
    seg = geom.segment_of(spots.x, spots.y)
    rng = np.random.default_rng(0)
    g_new = geom.copy()
    changes = []

    def frac(g, sel):
        h = g.spot_to_p0(spots.x[sel], spots.y[sel], spots.z[sel]) @ R.T
        return float(np.mean(np.all(np.abs(h - np.round(h)) < 0.12, axis=1)))
    for k, sg in enumerate(geom.segments):
        sel = np.flatnonzero(seg == k)
        if sel.size < min_spots:
            continue
        if sel.size > n_spots // max(len(geom.segments), 1):
            sel = rng.choice(sel, n_spots // len(geom.segments), replace=False)
        f0 = frac(geom, sel)
        best = (f0, 0.0, 0.0)
        g2 = geom.copy()
        for dx in offs:
            for dy in offs:
                g2.segments[k].orgx, g2.segments[k].orgy = sg.orgx + dx, sg.orgy + dy
                f = frac(g2, sel)
                if f > best[0]:
                    best = (f, dx, dy)
        f1, dx, dy = best
        if f1 > f0 + 0.1 and f1 >= 0.5:
            g_new.segments[k].orgx, g_new.segments[k].orgy = sg.orgx + dx, sg.orgy + dy
            changes.append((k, dx, dy, f0, f1))
            log(f"  INDEX: segment {k + 1} ({sg.x1}-{sg.x2}, {sg.y1}-{sg.y2}) appears displaced by "
                f"({dx:+.2f}, {dy:+.2f}) px: indexed fraction {f0:.2f} -> {f1:.2f}")
    return (g_new if changes else None), changes


def beam_centre_search_vectors(geom, spots, vecs, n_spots=6000, dmin=3.5, log=print, verbose=True):
    """Beam-centre search scored on individual reliable lattice vectors (not a full basis).

    A beam-centre error of a few pixels shifts all reciprocal positions by a constant; the index error
    along an axis grows with its length, so long axes are lost from the FFT while short ones survive
    (indexing difference vectors is immune). With the surviving vectors (refined score
    > 0.8) the shift that makes their indices most integral is found; the FFT is then redone there.
    Returns (dx, dy, score_header, score_best)."""
    inten = np.asarray(spots.intensity, float)
    top = np.argsort(-inten)[:n_spots]
    x, y, z = spots.x[top], spots.y[top], spots.z[top]
    keep = 1.0 / np.linalg.norm(geom.spot_to_p0(x, y, z), axis=1) >= dmin
    x, y, z = x[keep], y[keep], z[keep]
    if x.size < 200 or not vecs:
        return 0.0, 0.0, None, None
    lmax = max(np.linalg.norm(v) for v in vecs)
    spacing = geom.wavelength * geom.distance / lmax / geom.qx
    rng_px = max(1.5 * spacing, 8.0)
    step = max(spacing / 6.0, 1.0)

    def score(dx, dy):
        p = geom.copy(orgx=geom.orgx + dx, orgy=geom.orgy + dy).spot_to_p0(x, y, z)
        return float(np.mean([refine_vector(v, p, n_iter=2)[1] for v in vecs]))
    s0 = score(0.0, 0.0)
    best = (s0, 0.0, 0.0)
    for dx in np.arange(-rng_px, rng_px + 1e-9, step):
        for dy in np.arange(-rng_px, rng_px + 1e-9, step):
            sc = score(dx, dy)
            if sc > best[0]:
                best = (sc, dx, dy)
    b0 = best
    for dx in np.linspace(b0[1] - step, b0[1] + step, 5):
        for dy in np.linspace(b0[2] - step, b0[2] + step, 5):
            sc = score(dx, dy)
            if sc > best[0]:
                best = (sc, dx, dy)
    if verbose:
        log(f"  INDEX: beam-centre search on {len(vecs)} reliable lattice vectors (+-{rng_px:.1f} px): integrality "
            f"{s0:.3f} at header position, {best[0]:.3f} at offset ({best[1]:+.2f}, {best[2]:+.2f}) px")
    return best[1], best[2], s0, best[0]


def _misorientation_sym(c1, c2):
    """Misorientation (deg) between two crystals modulo lattice symmetry of the first (approx.: tests
    all integer basis permutations/sign changes mapping cell 2 onto cell 1)."""
    import itertools
    R1 = np.linalg.inv(c1.Astar)
    R2 = np.linalg.inv(c2.Astar)
    best = np.nan            # no basis correspondence (different cells)
    for perm in itertools.permutations(range(3)):
        for sgn in itertools.product((1, -1), repeat=3):
            M = np.diag(sgn)[list(perm)]
            if np.linalg.det(M) < 0:
                continue
            R2m = M @ R2
            if not np.allclose(np.linalg.norm(R2m, axis=1), np.linalg.norm(R1, axis=1), rtol=0.03):
                continue
            U = np.linalg.solve(R1, R2m).T          # approx rotation R1 -> R2m
            u, _, vt = np.linalg.svd(U)
            Rot = u @ vt
            ang = float(np.degrees(np.arccos(np.clip((np.trace(Rot) - 1) / 2, -1, 1))))
            best = ang if not np.isfinite(best) else min(best, ang)
    return best


def _hnf_rows(V):
    """Integer basis (rows) of the lattice generated by integer row vectors V (Euclid-style HNF)."""
    rows = [np.array(r, np.int64) for r in V if np.any(r)]
    basis = []
    for c in range(3):
        while True:
            nz = [i for i, r in enumerate(rows) if r[c] != 0]
            if len(nz) <= 1:
                break
            nz.sort(key=lambda i: abs(rows[i][c]))
            piv = rows[nz[0]].copy()
            for i in nz[1:]:
                rows[i] = rows[i] - (rows[i][c] // piv[c]) * piv
            rows = [r for r in rows if np.any(r)]
        nz = [i for i, r in enumerate(rows) if r[c] != 0]
        if nz:
            basis.append(rows.pop(nz[0]))
    return np.array(basis)


def remove_supercell(R, p, tol=0.1, n_max=3000, log=None):
    """If the indices of all well-indexed spots lie on a proper sublattice of Z^3 (e.g. h+k+l always
    even), the basis is a supercell (an augmented lattice basis): replace it by the true cell.
    R: real-space basis rows; p: reciprocal vectors of spots. Returns corrected R (Niggli reduced)."""
    h = p @ R.T
    hr = np.round(h)
    ok = np.all(np.abs(h - hr) < tol, axis=1) & np.any(hr != 0, axis=1)
    if ok.sum() < 50:
        return R
    H = hr[ok].astype(np.int64)
    if len(H) > n_max:
        H = H[np.random.default_rng(0).choice(len(H), n_max, replace=False)]
    # statistical sublattice test: in a correct cell every class v.h mod q (q = 2, 3) is populated in
    # proportion (q-1)/q; for a supercell one class is (nearly) empty. With several lattices, stray spots
    # partly fill the empty classes, so test depletion (< 25% of the expected share), not emptiness.
    sel = np.ones(len(H), bool)
    changed = True
    while changed:
        changed = False
        for q in (2, 3):
            for v in itertools.product(range(q), repeat=3):
                v = np.array(v)
                if not v.any() or (v[np.flatnonzero(v)[0]] != 1):
                    continue
                off = (H[sel] @ v) % q != 0
                if off.any() and off.mean() < 0.25 * (q - 1) / q and sel.sum() > 50:
                    sel &= (H @ v) % q == 0
                    changed = True
    # a perfectly empty class never enters the depletion loop (nothing to remove): the generators of the occupied
    # index lattice reveal it either way
    B = _hnf_rows(H[sel])                  # generators of the occupied index lattice
    if B.shape[0] < 3:
        return R
    det = int(round(abs(np.linalg.det(B.astype(float)))))
    if det <= 1:
        return R
    # true reciprocal basis = A* B^T  ->  real basis rows R_new = (B^T)^-1 applied: R_new = inv(B).T? derive:
    # p = A* h, h = B^T k (k integer in the true lattice) -> p = (A* B^T) k
    Astar_new = np.linalg.inv(R) @ B.T.astype(float)
    Rn, _ = niggli_reduce(np.linalg.inv(Astar_new))
    if log:
        log(f"  INDEX: indexed reflections occupy only 1/{det} of the index lattice: basis was a supercell; "
            f"reduced cell now {np.array2string(cell_from_Astar(np.linalg.inv(Rn)), precision=2)}")
    return Rn


def assign_indices(Astar, p0, tol=0.3):
    R = real_basis(Astar)
    h = p0 @ R.T
    hr = np.round(h)
    ok = np.all(np.abs(h - hr) < tol, axis=1) & np.any(hr != 0, axis=1)
    return hr.astype(int), ok


def index_spots(geom: Geometry, spots: SpotList, dmin_index=None, max_cell=None, refine_flags=None,
                min_fraction=0.5, verbose=True, log=print, search_extra_lattices=True,
                known_cell=None, known_system=None, max_err_xy=3.0, max_err_phi=2.0, beam_search=True,
                max_extra_lattices=3, n_fft=20000, max_cell_cap=400.0, basis_rank=0, _vector_search_done=False,
                fft1d=False):
    if refine_flags is None:
        refine_flags = ["BEAM", "AXIS", "ORIENTATION", "CELL"]
    p0_all = geom.spot_to_p0(spots.x, spots.y, spots.z)
    d_all = 1.0 / np.linalg.norm(p0_all, axis=1)
    phi = geom.z_to_phi(spots.z)
    # use strong low/medium resolution spots for the initial FFT
    if dmin_index is None:
        dmin_index = max(np.quantile(d_all, 0.05), 3.5)
    # strong spots drive the lattice search (only the strongest spots are used): on weak,
    # low-background data most spots are faint and many are noise, which spoils both the max-cell
    # estimate and the FFT
    inten = np.asarray(spots.intensity, float)
    strong_half = inten >= np.median(inten)
    if max_cell is None:
        top = np.argsort(-inten)[:max(min(len(inten), 20000), min(len(inten), 3000))]
        px_err = 3.0 * geom.qx / geom.distance / geom.wavelength      # 3 px on the detector, in 1/A
        mc_r = estimate_max_cell(p0_all[top], phi[top], geom.osc, min_sep=px_err)
        mc_d = estimate_max_cell_detector(geom, spots, top)
        # capped (400 A, enough for nearly all protein cells; FFT then reaches ~6 A): neighbour statistics are
        # inflated by split spots / continua; the caller retries with a larger cap if the cell approaches it
        global LAST_MAX_CELL_RAW, LAST_MAX_CELL_RECIP
        LAST_MAX_CELL_RAW = float(min(mc_r, mc_d if mc_d else mc_r))
        LAST_MAX_CELL_RECIP = float(mc_r)
        max_cell = float(np.clip(LAST_MAX_CELL_RAW, 30.0, max_cell_cap))
    cand = np.flatnonzero(d_all >= dmin_index)
    if cand.size > n_fft:
        cand = cand[np.argsort(-inten[cand])[:n_fft]]
    sel = np.zeros(len(d_all), bool)
    sel[cand] = True
    p = p0_all[sel]
    if verbose:
        log(f"  INDEX: {len(spots)} spots, using the {p.shape[0]} strongest with d >= {dmin_index:.2f} A; "
            f"max cell estimate {max_cell:.1f} A")
    ts, vals, dmin_used = (fft1d_candidate_vectors(p, dmin_index, max_cell) if fft1d
                           else fft_candidate_vectors(p, dmin_index, max_cell))
    if len(ts) < 3:
        raise RuntimeError("indexing failed: too few FFT peaks")
    p = p[np.einsum("ij,ij->i", p, p) <= 1.0 / dmin_used ** 2]
    refined = [refine_vector(t, p) for t in ts]
    ts = np.array([r[0] for r in refined])
    scores = np.array([r[1] for r in refined])
    good = (scores > 0.3) & (np.linalg.norm(ts, axis=1) > dmin_used)   # degenerate refits collapse to ~0
    ts, scores = ts[good], scores[good]
    ts, scores = _remove_multiples(ts, scores)
    order = np.lexsort((np.linalg.norm(ts, axis=1), -np.round(scores, 2)))
    ts, scores = ts[order], scores[order]
    if beam_search and not _vector_search_done:
        # reliable, non-collinear vectors (score >= 0.85 x best, >= 0.6): origin-error check before the basis
        rel = []
        thr_rel = max(0.6, 0.85 * float(np.max(scores))) if len(scores) else 1.0
        for t, sc in zip(ts, scores):
            if sc < thr_rel or len(rel) == 3:
                continue
            u = t / np.linalg.norm(t)
            if all(abs(u @ (r / np.linalg.norm(r))) < 0.95 for r in rel):
                rel.append(t)
        if len(rel) >= 1:
            dxv, dyv, sv0, svb = beam_centre_search_vectors(geom, spots, rel, log=log, verbose=verbose)
            if sv0 is not None and svb > sv0 + 0.005 and np.hypot(dxv, dyv) > 1.0:
                # Integrality of single vectors barely changes for shifts near a whole reflection spacing of a short
                # axis, and a wrong origin can still give a plausible-looking (super)cell. No threshold on the gain:
                # index at both positions and keep the solution that indexes more spots (a few seconds each).
                kw = dict(refine_flags=refine_flags, min_fraction=min_fraction, log=log, search_extra_lattices=False,
                          known_cell=known_cell, known_system=known_system, max_err_xy=max_err_xy,
                          max_err_phi=max_err_phi, beam_search=True, max_extra_lattices=max_extra_lattices,
                          n_fft=n_fft, max_cell_cap=max_cell_cap, basis_rank=basis_rank, _vector_search_done=True, fft1d=fft1d)
                g_sh = geom.copy(orgx=geom.orgx + dxv, orgy=geom.orgy + dyv)
                cands = []
                for lab, g_ in (("header", geom), (f"shifted ({dxv:+.2f}, {dyv:+.2f}) px", g_sh)):
                    try:
                        cands.append((lab, index_spots(g_, spots, dmin_index, max_cell, verbose=False, **kw)))
                    except RuntimeError:
                        pass
                if cands:
                    if verbose:
                        log("  INDEX: beam centre candidates: " + "; ".join(
                            f"{lab}: {int(r.indexed.sum())} spots indexed, rmsd {r.rmsd_xy:.2f} px, cell "
                            f"{np.array2string(r.exp.crystal.cell, precision=1)}" for lab, r in cands))
                    # the indexed count saturates at the acceptance limit (MAXIMUM_ERROR_OF_SPOT_POSITION): a wrong
                    # index origin indexes nearly as many spots with a larger positional rmsd. Counts within 2%: the
                    # clearly tighter fit (> 20%) wins.
                    n_best = max(int(r.indexed.sum()) for _, r in cands)
                    close = [c for c in cands if int(c[1].indexed.sum()) >= 0.98 * n_best]
                    if len(close) == 2 and max(r.rmsd_xy for _, r in close) > 1.2 * min(r.rmsd_xy for _, r in close):
                        lab, _ = min(close, key=lambda c: c[1].rmsd_xy)
                    else:
                        lab, _ = max(cands, key=lambda c: (int(c[1].indexed.sum()), -c[1].rmsd_xy))
                    g_best = g_sh if lab.startswith("shifted") else geom
                    if g_best is g_sh and verbose:
                        log(f"  INDEX: WARNING: header beam centre appears off by ({dxv:+.2f}, {dyv:+.2f}) px; "
                            f"re-indexing there")
                    return index_spots(g_best, spots, dmin_index, max_cell, refine_flags, min_fraction, verbose, log,
                                       search_extra_lattices, known_cell, known_system, max_err_xy, max_err_phi,
                                       beam_search=True, max_extra_lattices=max_extra_lattices, n_fft=n_fft,
                                       max_cell_cap=max_cell_cap, basis_rank=basis_rank, _vector_search_done=True, fft1d=fft1d)
    bb = best_basis(ts, p, rank=basis_rank)
    if bb is None:
        raise RuntimeError("indexing failed: no basis found")
    R, frac0 = bb
    if np.linalg.det(R) < 0:
        R = -R
    # re-fit the basis on its own (tightly indexed) spots: other lattices must not bias the cell
    for tol_fit in (0.15, 0.1, 0.1):
        h = p @ R.T
        ok_fit = np.all(np.abs(h - np.round(h)) < tol_fit, axis=1) & np.any(np.round(h) != 0, axis=1)
        if ok_fit.sum() < 20:
            break
        R_new = np.linalg.lstsq(p[ok_fit], np.round(h[ok_fit]), rcond=None)[0].T
        # keep the previous basis if the re-fit degenerates (spots on too few lattice planes)
        if abs(np.linalg.det(R_new)) < 1e-3 * abs(np.linalg.det(R)) or not np.all(np.isfinite(R_new)):
            break
        R = R_new
    R, M = niggli_reduce(R)
    R = remove_supercell(R, p, log=log if verbose else None)
    if beam_search:
        dx, dy, sc0, scb, _ = beam_centre_search(geom, spots, R, log=log, verbose=verbose)
        # accept a shifted origin only if it is clearly better (real misindexing gives a large gain;
        # sub-pixel improvements are left to the refinement)
        # only for a convincing single-lattice solution: with several lattices (or a poor basis) a low
        # indexed fraction would otherwise "improve" by moving the beam (report but do not act)
        if sc0 is not None and scb >= 0.5 and scb > sc0 + 0.1 and np.hypot(dx, dy) > 0.5:
            if verbose:
                log(f"  INDEX: WARNING: header beam centre appears wrong; using ORGX/ORGY shifted by "
                    f"({dx:+.2f}, {dy:+.2f}) px and re-indexing")
            geom = geom.copy(orgx=geom.orgx + dx, orgy=geom.orgy + dy)
            return index_spots(geom, spots, dmin_index, max_cell, refine_flags, min_fraction, verbose, log,
                               search_extra_lattices, known_cell, known_system, max_err_xy, max_err_phi,
                               beam_search=False, max_extra_lattices=max_extra_lattices, n_fft=n_fft,
                               max_cell_cap=max_cell_cap, basis_rank=basis_rank, fft1d=fft1d)
    Astar = np.linalg.inv(R)
    if verbose:
        log(f"  INDEX: initial basis indexes {100 * frac0:.1f}% of low-res spots; "
            f"reduced cell {np.array2string(cell_from_Astar(Astar), precision=2)}")
    exp = Experiment(geom.copy(), Crystal.from_Astar(Astar, "a"))
    # progressive refinement: extend resolution
    dsteps = sorted({dmin_index, max(dmin_index * 0.75, d_all.min()), d_all.min()}, reverse=True)
    for dcut in dsteps:
        use = d_all >= dcut
        p0u = exp.geom.spot_to_p0(spots.x[use], spots.y[use], spots.z[use])
        hkl, ok = assign_indices(exp.crystal.Astar, p0u, tol=0.2 if dcut == dsteps[0] else 0.15)
        idx = np.flatnonzero(use)[ok]
        res = refine(exp, hkl[ok], spots.x[idx], spots.y[idx], spots.z[idx], refine_flags)
        exp = res.exp
    Rchk = remove_supercell(real_basis(exp.crystal.Astar), exp.geom.spot_to_p0(spots.x, spots.y, spots.z),
                            log=log if verbose else None)
    if not np.allclose(Rchk, real_basis(exp.crystal.Astar)):
        exp = Experiment(exp.geom, Crystal.from_Astar(np.linalg.inv(Rchk), "a"))
    # final indexing: nearest hkl, accepted if the refined prediction lies within the position / spindle limits
    # (MAXIMUM_ERROR_OF_SPOT_POSITION / SPINDLE_POSITION) -- a physical criterion in detector space
    p0f = exp.geom.spot_to_p0(spots.x, spots.y, spots.z)
    hkl, ok = assign_indices(exp.crystal.Astar, p0f, tol=0.5)
    res = refine(exp, hkl[ok], spots.x[ok], spots.y[ok], spots.z[ok], refine_flags,
                 max_err_xy=max_err_xy, max_err_phi=max_err_phi)
    exp = res.exp
    p0f = exp.geom.spot_to_p0(spots.x, spots.y, spots.z)
    hkl, ok = assign_indices(exp.crystal.Astar, p0f, tol=0.5)
    res = refine(exp, hkl[ok], spots.x[ok], spots.y[ok], spots.z[ok], [], reject=True, n_reject_cycles=1,
                 max_err_xy=max_err_xy, max_err_phi=max_err_phi, robust=False)
    final_ok = np.zeros(len(spots), bool)
    final_ok[np.flatnonzero(ok)[res.inliers]] = True
    # success criterion on the stronger half of the spots (weak ones include noise; the spot list
    # likewise contains only the strongest)
    frac = float(final_ok[strong_half].mean()) if strong_half.any() else float(final_ok.mean())
    if verbose:
        log(f"  INDEX: {final_ok.sum()}/{len(spots)} spots indexed ({100 * final_ok.mean():.1f}%; "
            f"{100 * frac:.1f}% of the stronger half), "
            f"rmsd {res.rmsd_xy:.3f} px, {res.rmsd_phi:.4f} deg; cell "
            f"{np.array2string(exp.crystal.cell, precision=3)}")
    extra, domains = [], []
    if search_extra_lattices:
        extra, domains = search_extra_lattices_fn(exp, spots, final_ok, dmin_index, max_cell, max_err_xy, max_err_phi,
                                                  n_fft, max_extra_lattices, verbose, log)
    hkl_out = np.where(final_ok[:, None], hkl, 0)
    if frac < min_fraction and verbose:
        log(f"  INDEX WARNING: only {100 * frac:.1f}% of spots indexed (< {100 * min_fraction:.0f}%)")
    out = IndexingResult(exp, hkl_out, final_ok, frac, res.rmsd_xy, res.rmsd_phi, extra)
    out.domains = domains
    out.dmin_index, out.n_fft, out.max_cell = dmin_index, n_fft, max_cell
    return out


def classify_unindexed(res: IndexingResult, spots: SpotList, ice_rings=()):
    """Classify spots not claimed by any lattice relative to the main lattice:
    near-integer fractional indices (split / missed Bragg spots), half-integer (superstructure, modulation,
    doubled cell), ice-ring d-spacings, unexplained (other crystals, salt, noise). Returns rows
    (label, count, fraction of unindexed, chance fraction or None)."""
    claimed = res.indexed.copy()
    for e in res.extra_lattices:
        m = getattr(e, "spot_mask", None)
        if m is not None and m.size == claimed.size:
            claimed |= m
    U = ~claimed
    n = int(U.sum())
    if n == 0:
        return claimed, []
    g = res.exp.geom
    p0 = g.spot_to_p0(spots.x[U], spots.y[U], spots.z[U])
    fr = np.abs(p0 @ real_basis(res.exp.crystal.Astar).T)
    fr = np.abs(fr - np.round(fr))
    near = np.all(fr < 0.15, axis=1)
    half = np.all((fr < 0.15) | (np.abs(fr - 0.5) < 0.1), axis=1) & np.any(np.abs(fr - 0.5) < 0.1, axis=1) & ~near
    s = np.linalg.norm(p0, axis=1)
    ice = np.zeros(n, bool)
    for r in ice_rings:
        ice |= np.abs(s - 1.0 / r) < 0.004
    ice &= ~near & ~half
    rest = ~(near | half | ice)
    rows = [("near-integer indices (split / missed Bragg spots)", int(near.sum()), near.mean(), 0.3 ** 3),
            ("half-integer indices (superstructure / modulation)", int(half.sum()), half.mean(), 0.5 ** 3 - 0.3 ** 3),
            ("ice-ring d-spacings", int(ice.sum()), ice.mean(), None),
            ("unexplained (other crystals / noise)", int(rest.sum()), rest.mean(), None)]
    return claimed, rows


_UNIMODULAR = None


def _match_basis(Astar_sub, Astar_main):
    """Integer basis change T (det +1) putting a lattice of the same crystal form into the main lattice's setting:
    the Niggli-reduced basis, then the unimodular matrix with entries in {-1, 0, 1} whose metric is closest to the
    main lattice's. Returns T with new real basis rows = T @ old rows (indices: h_new = h_old @ T.T)."""
    global _UNIMODULAR
    if _UNIMODULAR is None:
        m = np.array(list(itertools.product((-1, 0, 1), repeat=9)), int).reshape(-1, 3, 3)
        _UNIMODULAR = m[np.round(np.linalg.det(m)).astype(int) == 1]
    Rs = real_basis(Astar_sub)
    Rn, Mr = niggli_reduce(Rs)
    Gm = real_basis(Astar_main) @ real_basis(Astar_main).T
    R2 = np.einsum("nij,jk->nik", _UNIMODULAR, Rn)
    G2 = np.einsum("nij,nkj->nik", R2, R2)
    cost = np.abs(G2 - Gm).sum(axis=(1, 2))
    return _UNIMODULAR[int(np.argmin(cost))] @ Mr


def search_extra_lattices_fn(exp, spots, final_ok, dmin_index, max_cell, max_err_xy=3.0, max_err_phi=2.0, n_fft=20000,
                             max_extra_lattices=3, verbose=True, log=print):
    """Iterative search for further lattices (split / multiple crystals) and weak near-copy domains among the spots
    the main lattice does not index. Returns (extra lattices, domains)."""
    extra = []
    # iterative search for further lattices (split / multiple crystals) in the unindexed spots
    claimed = final_ok.copy()
    domains = []
    c_main = cell_from_Astar(np.linalg.inv(niggli_reduce(real_basis(exp.crystal.Astar))[0]))
    # rotation width of the main lattice's spots (median frame extent of its stronger indexed spots, deg): a
    # near-copy domain closer than ~1.5x this has overlapping spots (composite integration); a farther
    # one has resolved spots and is integrated as a lattice of its own whatever its share of the spots
    ok_w = final_ok & (spots.intensity >= np.median(spots.intensity[final_ok])) if final_ok.any() else final_ok
    w_rot = (float(np.median(spots.zmax[ok_w] - spots.zmin[ok_w] + 1)) * abs(exp.geom.osc)
             if ok_w.sum() >= 20 else np.inf)
    for k in range(max_extra_lattices + 6):
        if len(extra) >= max_extra_lattices:
            break
        rest_idx = np.flatnonzero(~claimed)
        if rest_idx.size < max(200, 0.05 * len(spots)):
            break
        try:
            sub = index_spots(exp.geom, spots.subset(rest_idx), dmin_index, max_cell,
                              refine_flags=["ORIENTATION", "CELL"], verbose=False, search_extra_lattices=False,
                              beam_search=False, max_err_xy=max_err_xy, max_err_phi=max_err_phi, n_fft=n_fft)
        except Exception:
            break
        # same crystal form required: reduced cells must agree (3% lengths, 3 deg angles); otherwise
        # noise / contaminant spots can be "indexed" by a huge cell
        c_sub = cell_from_Astar(np.linalg.inv(niggli_reduce(real_basis(sub.exp.crystal.Astar))[0]))
        same_cell = bool(np.all(np.abs(c_sub[:3] - c_main[:3]) / c_main[:3] < 0.03) and
                         np.all(np.abs(np.abs(c_sub[3:] - 90) - np.abs(c_main[3:] - 90)) < 3.0))
        n_ix = int(sub.indexed.sum())
        mis_main = _misorientation_sym(exp.crystal, sub.exp.crystal)
        resolved = same_cell and np.isfinite(mis_main) and mis_main > 1.5 * w_rot
        strong = (sub.fraction > 0.3 or resolved) and n_ix > max(100, 0.05 * final_ok.sum())
        # weak near-copy of the main lattice (split / mosaic crystal: domains within a few degrees): part of the
        # same crystal, integrated as one composite spot with the main lattice -- its spots are not
        # foreign; only strong domains are integrated as separate lattices
        near_copy = same_cell and np.isfinite(mis_main) and mis_main < 3.0 and n_ix > max(100, 0.02 * len(spots))
        if near_copy and not strong:
            full = np.zeros(len(spots), bool)
            full[rest_idx[sub.indexed]] = True
            sub.spot_mask = full
            domains.append(sub)
            claimed |= full
            if verbose:
                log(f"  INDEX: weak domain {len(domains)} ({n_ix} spots, {100 * sub.fraction:.0f}% of the remaining "
                    f"stronger half) misoriented {mis_main:.2f} deg, cell {np.array2string(sub.exp.crystal.cell, precision=2)}"
                    f" (spot rotation width {w_rot:.2f} deg): integrated with the main lattice (composite)")
            continue
        if not strong:
            break
        if not same_cell:
            if verbose:
                log(f"  INDEX: candidate extra lattice with cell {np.array2string(c_sub, precision=1)} rejected "
                    f"(different from the main reduced cell)")
            break
        # duplicate of an existing lattice (same orientation within 1 deg, any lattice symmetry)?
        mis = [_misorientation_sym(L.crystal, sub.exp.crystal) for L in [exp] + [e.exp for e in extra]]
        if np.nanmin(np.r_[mis, 99.0]) < min(1.0, 1.5 * w_rot):
            claimed[rest_idx[sub.indexed]] = True
            continue
        # same setting as the main lattice
        T = _match_basis(sub.exp.crystal.Astar, exp.crystal.Astar)
        if not np.array_equal(T, np.eye(3, dtype=int)):
            sub.exp = Experiment(sub.exp.geom, Crystal.from_Astar(np.linalg.inv(T @ real_basis(sub.exp.crystal.Astar)),
                                                                  sub.exp.crystal.system))
            sub.hkl = sub.hkl @ T.T
        # spot indices relative to the full list
        full = np.zeros(len(spots), bool)
        full[rest_idx[sub.indexed]] = True
        sub.spot_mask = full
        extra.append(sub)
        claimed |= full
        if verbose:
            rot = _misorientation_sym(exp.crystal, sub.exp.crystal)
            log(f"  INDEX: lattice {k + 2} indexes {sub.indexed.sum()} of the remaining spots; cell "
                f"{np.array2string(sub.exp.crystal.cell, precision=2)}; misorientation {rot:.2f} deg "
                f"(spot rotation width {w_rot:.2f} deg)")
    return extra, domains

"""Bravais-lattice determination and conventional settings.

The lattice holohedry of the reduced cell is found with gemmi (Le Page-type search for
twofold axes, obliquity tolerance). All subgroups of its rotation group are enumerated and
each is brought to its conventional (ITA reference) setting by constructing the basis from
the symmetry axes. The result is a list of candidate lattices with a metric-distortion penalty,
like the table of lattice characters printed by INDEX, but complete for all subgroups
(it also gives the candidate point groups for symmetry determination in CORRECT).

Conventions: real-space basis rows R; new basis R' = P^T R (columns of P are new axes in terms
of old); Miller indices transform as h' = P^T h; symmetry operators as R' = P^-1 R P.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import gemmi
import numpy as np

from .geometry import cell_from_metric

# lattice system letter used by models.CELL_PARAMS
_SYSTEM_LETTER = {"triclinic": "a", "monoclinic": "m", "orthorhombic": "o", "tetragonal": "t",
                  "trigonal": "h", "hexagonal": "h", "cubic": "c"}


@dataclass
class LatticeCandidate:
    bravais: str            # e.g. 'mC'
    point_group: str        # e.g. '2', '222', '422'
    space_group: str        # lowest symmorphic space group of this point group + centring
    P: np.ndarray           # integer transformation, reduced -> conventional (columns)
    cell: np.ndarray        # conventional cell from the (unconstrained) reduced metric
    penalty: float          # metric distortion (degrees-equivalent)
    system: str             # models.CELL_PARAMS key
    ops: list               # proper rotation matrices (conventional basis, act on frac. coords)

    @property
    def reindex(self):
        """Matrix M with h_conv = M @ h_reduced."""
        return self.P.T


def _order(R):
    M = np.eye(3, dtype=int)
    for k in range(1, 7):
        M = M @ R
        if np.array_equal(M, np.eye(3, dtype=int)):
            return k
    return 0


def _int_inv(R):
    return np.round(np.linalg.inv(R)).astype(int)


def _rotations(hm):
    """Set of the rotation matrices (as bytes) of a space group's symmetry operations."""
    return {(np.array(op.rot, int) // gemmi.Op.DEN).tobytes() for op in gemmi.SpaceGroup(hm).operations().sym_ops}


def _closure(gens):
    G = {np.eye(3, dtype=int).tobytes(): np.eye(3, dtype=int)}
    frontier = list(gens)
    while frontier:
        new = []
        for A in frontier:
            for B in list(G.values()):
                for C in (A @ B, B @ A):
                    k = C.tobytes()
                    if k not in G:
                        G[k] = C
                        new.append(C)
        frontier = new
        if len(G) > 48:
            break
    return list(G.values())


def _small_vectors(nmax=3):
    r = range(-nmax, nmax + 1)
    v = np.array([u for u in itertools.product(r, r, r) if any(u)])
    # primitive vectors only (gcd 1)
    g = np.gcd.reduce(np.abs(v), axis=1)
    return v[g == 1]


_VECS = _small_vectors(4)


def _axis(R, G):
    """Shortest integer lattice vector along the rotation axis of R."""
    fixed = _VECS[np.all(_VECS @ R.T == _VECS, axis=1)]
    if len(fixed) == 0:
        return None
    L = np.einsum("ni,ij,nj->n", fixed, G, fixed)
    return fixed[np.argmin(L)]


# Angular tolerance for 'perpendicular' when constructing conventional axes from symmetry axes.
# Matches the lattice-symmetry tolerance (2 deg; up to ~3-5 deg is common).
PERP_TOL = np.sin(np.radians(2.0))


def _is_perp(u, v, G, tol=PERP_TOL):
    return abs(u @ G @ v) <= tol * np.sqrt((u @ G @ u) * (v @ G @ v))


def _perp_vectors(u, G, tol=PERP_TOL):
    """Integer vectors (approximately) perpendicular to integer vector u, sorted by length."""
    d = _VECS @ G @ u
    L = np.einsum("ni,ij,nj->n", _VECS, G, _VECS)
    ok = np.abs(d) <= tol * np.sqrt(L * (u @ G @ u))
    v, L = _VECS[ok], L[ok]
    return v[np.argsort(L, kind="stable")]


def _centring_vectors(P):
    """Fractional centring translations (in the new basis) produced by basis change P."""
    Pi = np.linalg.inv(P)
    det = int(round(abs(np.linalg.det(P))))
    pts = set()
    for x in itertools.product(range(det), repeat=3):
        f = Pi @ np.array(x, float)
        f = np.round(f - np.floor(f + 1e-6), 6) % 1.0
        pts.add(tuple(np.round(f, 4)))
    pts.discard((0.0, 0.0, 0.0))
    return sorted(pts)


def _centring_symbol(P):
    det = int(round(abs(np.linalg.det(P))))
    if det == 1:
        return "P"
    c = _centring_vectors(P)
    s = set(c)
    if det == 2:
        if (0.5, 0.5, 0.0) in s:
            return "C"
        if (0.0, 0.5, 0.5) in s:
            return "A"
        if (0.5, 0.0, 0.5) in s:
            return "B"
        if (0.5, 0.5, 0.5) in s:
            return "I"
    if det == 4 and {(0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)} <= s:
        return "F"
    if det == 3:
        if (round(2 / 3, 4), round(1 / 3, 4), round(1 / 3, 4)) in s:
            return "R"     # obverse
        if (round(1 / 3, 4), round(2 / 3, 4), round(1 / 3, 4)) in s:
            return "Rrev"
    return "?"


def _conventional_basis(ops, G):
    """Return (P, system, point_group) for proper rotation group ``ops`` in the reduced basis."""
    orders = [_order(R) for R in ops]
    n = len(ops)
    twos = [R for R, o in zip(ops, orders) if o == 2]
    threes = [R for R, o in zip(ops, orders) if o == 3]
    fours = [R for R, o in zip(ops, orders) if o == 4]
    sixes = [R for R, o in zip(ops, orders) if o == 6]
    I = np.eye(3, dtype=int)
    if n == 1:
        return I, "triclinic", "1"
    if n == 2:  # monoclinic: b along the 2-fold
        b = _axis(twos[0], G)
        perp = _perp_vectors(b, G)
        best = None
        for i in range(min(len(perp), 12)):
            for j in range(i + 1, min(len(perp), 12)):
                a, c = perp[i], perp[j]
                P = np.array([a, b, c]).T
                det = round(np.linalg.det(P))
                if det == 0 or abs(det) > 2:
                    continue
                if det < 0:
                    P = np.array([a, -b, c]).T
                    P = np.array([c, b, a]).T if np.linalg.det(np.array([c, b, a]).T) > 0 else P
                if np.linalg.det(P) < 0:
                    continue
                cen = _centring_symbol(P)
                if cen not in ("P", "C"):
                    continue
                Gc = P.T @ G @ P
                cell = cell_from_metric(Gc)
                score = abs(cell[4] - 90.0) + (0 if cell[4] >= 90 else 0.01)
                key = (0 if cen == "P" else 1, score, cell[0] + cell[2])
                if best is None or key < best[0]:
                    best = (key, P)
        if best is None:
            return None
        P = best[1]
        # standard: beta >= 90
        cell = cell_from_metric(P.T @ G @ P)
        if cell[4] < 90:
            # (a,b,c) -> (-a,-b,c): stays right-handed and C-centred, beta -> 180 - beta
            P = np.array([-P[:, 0], -P[:, 1], P[:, 2]]).T
        cell = cell_from_metric(P.T @ G @ P)
        if _centring_symbol(P) == "P" and cell[0] > cell[2] + 1e-6:
            # primitive monoclinic: conventional a <= c ((a,b,c) -> (c,-b,a) keeps beta and handedness)
            P = np.array([P[:, 2], -P[:, 1], P[:, 0]]).T
        return P, "monoclinic", "2"
    if n == 4 and len(twos) == 3:  # orthorhombic 222
        ax = [_axis(R, G) for R in twos]
        P = np.array(ax).T
        if np.linalg.det(P) < 0:
            P[:, 2] *= -1
        cen = _centring_symbol(P)
        perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        if cen in ("A", "B"):
            for pm in perms:
                Q = P[:, pm]
                if _centring_symbol(Q) == "C":
                    P = Q
                    break
            cen = "C"
        Gc = P.T @ G @ P
        L = np.diag(Gc)
        if cen in ("P", "I", "F"):
            order = np.argsort(L)
        else:  # C: keep c, sort a<b
            order = [0, 1, 2] if L[0] <= L[1] else [1, 0, 2]
        P = P[:, order]
        if np.linalg.det(P) < 0:
            P[:, 2] *= -1
        return P, "orthorhombic", "222"
    if fours and n in (4, 8) and not threes:  # tetragonal
        c = _axis(fours[0], G)
        R4 = fours[0]
        cand = _perp_vectors(c, G)
        if n == 8:
            two_perp = [_axis(R, G) for R in twos if _is_perp(_axis(R, G), c, G)]
            two_perp = [t for t in two_perp if t is not None]
            if two_perp:
                cand = np.array(sorted(two_perp, key=lambda t: t @ G @ t))
        for a in cand:
            b = R4 @ a
            P = np.array([a, b, c]).T
            if np.linalg.det(P) < 0:
                b = _int_inv(R4) @ a      # opposite rotation sense (R^-1, not R^T: basis is not orthonormal)
                P = np.array([a, b, c]).T
            if np.linalg.det(P) > 0 and _centring_symbol(P) in ("P", "I"):
                return P, "tetragonal", "422" if n == 8 else "4"
        return None
    if (threes or sixes) and n in (3, 6, 12) and len(threes) <= 2 + (2 if sixes else 0) and not \
            (n == 12 and len(threes) == 8):  # trigonal / hexagonal
        Rk = sixes[0] if sixes else threes[0]
        c = _axis(Rk, G)
        R3 = [R for R in threes if np.array_equal(_axis(R, G), c) or np.array_equal(_axis(R, G), -c)]
        if not R3:
            R3 = [Rk @ Rk]
        perp = _perp_vectors(c, G)
        if n in (6, 12) and len(twos) >= 3:
            tp = [_axis(R, G) for R in twos]
            tp = [t for t in tp if t is not None and _is_perp(t, c, G)]
            if tp:
                perp = np.array(sorted(tp, key=lambda t: t @ G @ t) + list(perp))
        for a in perp:
            for R in R3 + [_int_inv(R) for R in R3]:
                b = R @ a
                P = np.array([a, b, c]).T
                if np.linalg.det(P) <= 0:
                    continue
                Gc = P.T @ G @ P
                cell = cell_from_metric(Gc)
                if abs(cell[5] - 120) > 5:
                    continue
                cen = _centring_symbol(P)
                if cen == "Rrev":
                    P = np.array([-a, -b, c]).T
                    cen = _centring_symbol(P)
                if cen in ("P", "R"):
                    if sixes:
                        pg = "622" if n == 12 else "6"
                    else:
                        pg = "32" if n == 6 else "3"
                    if pg == "32" and cen == "P":
                        # two orientations on a primitive hexagonal lattice: 2-folds along a (P321) or perpendicular
                        # to a (P312); same point group, different Laue class (-3m1 / -31m)
                        Pi = np.linalg.inv(P)
                        conv = {np.round(Pi @ R @ P).astype(int).tobytes() for R in ops}
                        if conv != _rotations("P 3 2 1"):
                            pg = "312"
                    return P, ("hexagonal" if sixes else "trigonal"), pg
        return None
    if threes and len(threes) == 8:  # cubic
        if fours:
            ax = []
            for R in fours:
                u = _axis(R, G)
                if not any(abs(abs(u @ G @ v) - np.sqrt((u @ G @ u) * (v @ G @ v))) < 1e-3 * np.sqrt((u @ G @ u) * (v @ G @ v)) for v in ax):
                    ax.append(u)
        else:
            ax = [_axis(R, G) for R in twos]
        if len(ax) < 3:
            return None
        P = np.array(ax[:3]).T
        if np.linalg.det(P) < 0:
            P[:, 2] *= -1
        return P, "cubic", "432" if fours else "23"
    return None


_SG_FOR = {  # (point group, centring) -> lowest symmorphic space group (reference setting)
    ("1", "P"): "P 1", ("2", "P"): "P 1 2 1", ("2", "C"): "C 1 2 1",
    ("222", "P"): "P 2 2 2", ("222", "C"): "C 2 2 2", ("222", "I"): "I 2 2 2", ("222", "F"): "F 2 2 2",
    ("4", "P"): "P 4", ("4", "I"): "I 4", ("422", "P"): "P 4 2 2", ("422", "I"): "I 4 2 2",
    ("3", "P"): "P 3", ("3", "R"): "R 3", ("32", "P"): "P 3 2 1", ("312", "P"): "P 3 1 2", ("32", "R"): "R 3 2",
    ("6", "P"): "P 6", ("622", "P"): "P 6 2 2",
    ("23", "P"): "P 2 3", ("23", "I"): "I 2 3", ("23", "F"): "F 2 3",
    ("432", "P"): "P 4 3 2", ("432", "I"): "I 4 3 2", ("432", "F"): "F 4 3 2",
}
_BRAVAIS = {"triclinic": "a", "monoclinic": "m", "orthorhombic": "o", "tetragonal": "t",
            "trigonal": "h", "hexagonal": "h", "cubic": "c"}


def _penalty(cell, system):
    """Metric distortion: angle deviations (deg) + relative length deviations (in 'degrees')."""
    a, b, c, al, be, ga = cell
    k = 57.2958  # relative length deviation -> deg-equivalent
    if system == "triclinic":
        return 0.0
    if system == "monoclinic":
        return abs(al - 90) + abs(ga - 90)
    if system == "orthorhombic":
        return abs(al - 90) + abs(be - 90) + abs(ga - 90)
    if system == "tetragonal":
        return abs(al - 90) + abs(be - 90) + abs(ga - 90) + k * abs(a - b) / (a + b) * 2
    if system in ("trigonal", "hexagonal"):
        return abs(al - 90) + abs(be - 90) + abs(ga - 120) + k * abs(a - b) / (a + b) * 2
    if system == "cubic":
        m = (a + b + c) / 3
        return abs(al - 90) + abs(be - 90) + abs(ga - 90) + k * (abs(a - m) + abs(b - m) + abs(c - m)) / m
    return 99.0


def lattice_candidates(reduced_cell, max_obliq=2.0, max_penalty=None):
    """All Bravais-lattice candidates compatible with the reduced (primitive) cell."""
    uc = gemmi.UnitCell(*reduced_cell)
    G = np.array(uc.metric_tensor().as_mat33().tolist(), float) if hasattr(uc, "metric_tensor") else None
    from .geometry import metric_from_cell
    G = metric_from_cell(reduced_cell)
    gops = gemmi.find_lattice_symmetry(uc, "P", max_obliq)
    rots = []
    seen = set()
    for op in gops.sym_ops:
        R = np.array(op.rot, int) // gemmi.Op.DEN
        if round(np.linalg.det(R)) != 1:
            continue
        k = R.tobytes()
        if k not in seen:
            seen.add(k)
            rots.append(R)
    # subgroups: closures of all pairs (any finite rotation group is 2-generated)
    subgroups = {}
    for A, B in itertools.combinations_with_replacement(rots, 2):
        H = _closure([A, B])
        key = frozenset(R.tobytes() for R in H)
        subgroups.setdefault(key, H)
    subgroups.setdefault(frozenset([np.eye(3, dtype=int).tobytes()]), [np.eye(3, dtype=int)])
    out = []
    for key, H in subgroups.items():
        res = _conventional_basis(H, G)
        if res is None:
            continue
        P, system, pg = res
        P = np.round(P).astype(int)
        cen = _centring_symbol(P)
        if (pg, cen) not in _SG_FOR:
            continue
        cell = cell_from_metric(P.T @ G @ P)
        pen = _penalty(cell, system)
        Pi = np.linalg.inv(P)
        ops_conv = [np.round(Pi @ R @ P).astype(int) for R in H]
        bravais = _BRAVAIS[system] + cen
        sg = _SG_FOR[(pg, cen)]
        # sanity check with gemmi: the conventional ops must belong to the reference group
        refm = {(np.array(op.rot, int) // gemmi.Op.DEN).tobytes() for op in gemmi.SpaceGroup(sg).operations().sym_ops}
        if not all(R.tobytes() in refm for R in ops_conv):
            continue
        out.append(LatticeCandidate(bravais, pg, sg, P, cell, pen, _SYSTEM_LETTER[system], ops_conv))
    # unique by (bravais, point group, P)
    uniq = {}
    for c in out:
        k = (c.bravais, c.point_group, c.P.tobytes())
        uniq.setdefault(k, c)
    out = sorted(uniq.values(), key=lambda c: (-len(c.ops), c.penalty))
    if max_penalty is not None:
        out = [c for c in out if c.penalty <= max_penalty]
    return out


def best_lattice_per_bravais(cands):
    """For each Bravais type keep the lowest-penalty candidate ."""
    best = {}
    for c in cands:
        k = (c.bravais, c.point_group)
        if k not in best or c.penalty < best[k].penalty:
            best[k] = c
    order = ["aP", "mP", "mC", "oP", "oC", "oI", "oF", "tP", "tI", "hP", "hR", "cP", "cI", "cF"]
    return sorted(best.values(), key=lambda c: (order.index(c.bravais) if c.bravais in order else 99,
                                                len(c.ops)))

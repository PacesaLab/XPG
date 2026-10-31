"""Vectorised reciprocal-space symmetry helpers built on gemmi."""
from __future__ import annotations

import gemmi
import numpy as np


def as_spacegroup(sg) -> gemmi.SpaceGroup:
    if isinstance(sg, gemmi.SpaceGroup):
        return sg
    if isinstance(sg, (int, np.integer)):
        return gemmi.find_spacegroup_by_number(int(sg))
    return gemmi.SpaceGroup(sg)


def op_matrix(op) -> np.ndarray:
    """Rotation part of a gemmi Op acting on hkl row vectors (h' = h @ R): integer when exact, else the exact
    fractional matrix (twin laws / alternative indexings of centred cells, e.g. the trigonal 3-fold in a C-centred
    orthohexagonal cell has elements +-1/2, +-3/2). Integer division would silently give a different operator."""
    r = np.array(op.rot, dtype=np.int64)
    return r // gemmi.Op.DEN if not (r % gemmi.Op.DEN).any() else r / gemmi.Op.DEN


def apply_rot(h: np.ndarray, R: np.ndarray) -> np.ndarray:
    """h @ R as integer indices (reflections obeying the lattice centring map to integers under fractional
    lattice operators)."""
    h2 = h @ R
    return h2 if np.issubdtype(h2.dtype, np.integer) else np.rint(h2).astype(np.int64)


def centring_reindex(T, sg, tol=0.05):
    """Index transformation h' = M h from an approximate T (e.g. inv(A*_alt) A*), exact for the reflections the
    centring of sg allows, or None. Between centred settings of one lattice M has fractions 1/n, n the number of
    centring translations (two C cells of a hexagonal lattice: +-1/2, +-3/2); M is accepted when it is within tol of T
    and maps every allowed index in -2..2 (a generating set of that lattice for every centring) to integers. Integer
    dtype when M is integral."""
    sg = as_spacegroup(sg)
    cen = np.array(sg.operations().cen_ops, dtype=np.int64)
    n = len(cen)
    M = np.round(n * np.asarray(T, float)) / n
    if np.abs(M - T).max() > tol:
        return None
    g = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), -1).reshape(-1, 3)
    g = g[((g @ cen.T) % gemmi.Op.DEN == 0).all(axis=1)]
    hp = g @ M.T
    if np.abs(hp - np.rint(hp)).max() > 1e-9:
        return None
    return np.rint(M).astype(np.int64) if np.allclose(M, np.rint(M)) else M


def rot_matrices(sg) -> np.ndarray:
    """Point-group rotation matrices (nops, 3, 3) acting on hkl row vectors: h' = h @ R."""
    sg = as_spacegroup(sg)
    rots = []
    seen = set()
    for op in sg.operations().sym_ops:
        R = np.array(op.rot, dtype=np.int64) // gemmi.Op.DEN
        key = R.tobytes()
        if key not in seen:
            seen.add(key)
            rots.append(R)
    return np.array(rots)


def hkl_key(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, np.int64) + 1024
    return (h[..., 0] << 22) | (h[..., 1] << 11) | h[..., 2]


def key_hkl(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, np.int64)
    return np.stack([(k >> 22) & 2047, (k >> 11) & 2047, k & 2047], -1) - 1024


def to_asu(hkl, sg, anomalous=False):
    """Map hkl to the standard reciprocal ASU (vectorised).

    Returns (hasu (N,3), isym (N,), plus (N,) bool): ISYM follows the MTZ convention
    (odd = I(+) / h related by a proper rotation, even = Friedel mate).
    When ``anomalous`` is True, Friedel mates map to the same asu hkl and are distinguished
    by ``plus``; the same holds when False (the flag is just informative then).
    """
    sg = as_spacegroup(sg)
    hkl = np.asarray(hkl, np.int64)
    R = rot_matrices(sg)
    eq = np.einsum("nj,kji->kni", hkl, R)  # (nops, N, 3)
    keys = np.concatenate([hkl_key(eq), hkl_key(-eq)], 0)  # (2 nops, N)
    canon = keys.max(0)
    uk, inv = np.unique(canon, return_inverse=True)
    asu = gemmi.ReciprocalAsu(sg)
    ops = sg.operations()
    ua = np.array([asu.to_asu(list(map(int, h)), ops)[0] for h in key_hkl(uk)], np.int64).reshape(-1, 3)
    hasu = ua[inv]
    akey = hkl_key(hasu)
    kp = hkl_key(eq) == akey[None]
    km = hkl_key(-eq) == akey[None]
    # MTZ convention: first matching operator, (+) before (-) for each operator -> minimum ISYM
    big = 1 << 30
    ip = np.where(kp.any(0), 2 * kp.argmax(0) + 1, big)
    im = np.where(km.any(0), 2 * km.argmax(0) + 2, big)
    isym = np.minimum(ip, im)
    return hasu, isym.astype(np.int32), (isym % 2 == 1)


def reflection_props(hasu, sg):
    sg = as_spacegroup(sg)
    ops = sg.operations()
    h = np.ascontiguousarray(hasu, dtype=np.int32)
    centric = np.asarray(ops.centric_flag_array(h), bool)
    eps = np.asarray(ops.epsilon_factor_without_centering_array(h), float)
    absent = np.asarray(ops.systematic_absences(h), bool)
    return centric, eps, absent


def d_spacing(hkl, cell):
    uc = cell if isinstance(cell, gemmi.UnitCell) else gemmi.UnitCell(*cell)
    return np.asarray(uc.calculate_d_array(np.ascontiguousarray(hkl, dtype=np.int32)), float)

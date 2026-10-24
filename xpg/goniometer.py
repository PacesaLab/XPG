"""Multi-axis goniometers (kappa, chi) and detector two-theta from image headers.

The data model has a single rotation axis; for a multi-axis goniometer the effective rotation axis is the
scanned axis carried by all axes it is mounted on (closer to the lab), at their fixed angles:

    axis_lab = R(outer_1, a_1) ... R(outer_n, a_n) v_scan

(inner axes only change the crystal orientation, which indexing absorbs into U). A detector two-theta
swing about an axis through the crystal rotates the detector rigidly: DIRECTION_OF_DETECTOR_X/Y-AXIS are
rotated while ORGX, ORGY and DETECTOR_DISTANCE (foot of the perpendicular from the crystal) stay the same.

NeXus (Eiger master files): the chain is followed from ``/entry/sample/depends_on`` through the
``depends_on`` attributes of NXtransformations (vectors in the McStas frame: z along the beam, y up).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rot(v, deg):
    v = np.asarray(v, float)
    u = v / np.linalg.norm(v)
    a = np.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


@dataclass
class Axis:
    name: str
    vector: np.ndarray
    values: np.ndarray            # degrees (per frame, or one value)
    kind: str = "rotation"

    @property
    def start(self):
        return float(self.values[0])

    @property
    def increment(self):
        return float(np.median(np.diff(self.values))) if self.values.size > 1 else 0.0


@dataclass
class GonioChain:
    axes: list = field(default_factory=list)          # outer (lab) -> inner (sample)

    def scan_index(self):
        incs = [abs(a.increment) if a.kind == "rotation" else 0.0 for a in self.axes]
        return int(np.argmax(incs)) if incs and max(incs) > 1e-6 else None

    def effective_axis(self, scan=None):
        i = self.scan_index() if scan is None else scan
        if i is None:
            return None
        R = np.eye(3)
        for a in self.axes[:i]:
            if a.kind == "rotation":
                R = R @ rot(a.vector, a.start)
        v = R @ np.asarray(self.axes[i].vector, float)
        return v / np.linalg.norm(v)

    def describe(self):
        return ", ".join(f"{a.name}={a.start:.2f}" + (f" (scan {a.increment:+.3f}/frame)" if abs(a.increment) > 1e-6 else "")
                         for a in self.axes)


def _attr(ds, key, default=None):
    v = ds.attrs.get(key, default)
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in "SO" and v.size == 1:
        v = v.ravel()[0]
        v = v.decode() if isinstance(v, bytes) else v
    return v


def _resolve(h5, path, base):
    if path in (None, ".", ""):
        return None
    path = str(path)
    if not path.startswith("/"):
        path = base.rsplit("/", 1)[0] + "/" + path
    return path if path in h5 else None


def _follow(h5, start):
    """Datasets of an NXtransformations chain from ``start`` towards the lab (inner -> outer)."""
    out, seen = [], set()
    cur = start
    while cur is not None and cur not in seen:
        seen.add(cur)
        ds = h5[cur]
        out.append((cur, ds))
        cur = _resolve(h5, _attr(ds, "depends_on"), cur)
    return out


def read_nexus_goniometer(h5):
    """GonioChain (outer -> inner) from a NeXus file, or None if the chain is absent."""
    dep = None
    try:
        dep = h5["/entry/sample/depends_on"][()]
        dep = dep.decode() if isinstance(dep, bytes) else str(dep)
    except Exception:
        pass
    start = dep if dep and dep in h5 else None
    if start is None:
        for cand in ("/entry/sample/transformations/phi", "/entry/sample/transformations/omega"):
            if cand in h5:
                start = cand
                break
    if start is None:
        return None
    axes = []
    for path, ds in _follow(h5, start):
        kind = str(_attr(ds, "transformation_type", "rotation")).lower()
        vec = _attr(ds, "vector")
        if vec is None:
            continue
        vals = np.atleast_1d(np.asarray(ds[()], float))
        units = str(_attr(ds, "units", "deg")).lower()
        if kind == "rotation" and units.startswith("rad"):
            vals = np.degrees(vals)
        axes.append(Axis(path.rsplit("/", 1)[-1], np.asarray(vec, float), vals, kind))
    return GonioChain(list(reversed(axes))) if axes else None


def read_nexus_detector_axes(h5, det="/entry/instrument/detector"):
    """(fast, slow, R_two_theta, two_theta_deg) in the McStas frame, or None.

    fast/slow are the module pixel directions with all *rotations* of their depends_on chains applied
    (translations only move the origin and are handled by beam centre + distance)."""
    mod = None
    for name in ("module", "detector_module", "module_1"):
        if f"{det}/{name}/fast_pixel_direction" in h5:
            mod = f"{det}/{name}"
            break
    if mod is None:
        return None
    vecs = []
    R2 = np.eye(3)
    tth = 0.0
    for key in ("fast_pixel_direction", "slow_pixel_direction"):
        ds = h5[f"{mod}/{key}"]
        v = np.asarray(_attr(ds, "vector"), float)
        R = np.eye(3)
        chain = _follow(h5, _resolve(h5, _attr(ds, "depends_on"), f"{mod}/{key}"))
        for path, d in reversed(chain):            # outer -> inner
            if str(_attr(d, "transformation_type", "")).lower() == "rotation":
                ang = float(np.atleast_1d(d[()])[0])
                if str(_attr(d, "units", "deg")).lower().startswith("rad"):
                    ang = np.degrees(ang)
                Ri = rot(np.asarray(_attr(d, "vector"), float), ang)
                R = R @ Ri
                if "theta" in path.lower():
                    R2, tth = Ri, ang
        vecs.append(R @ v)
    return vecs[0], vecs[1], R2, tth


def cbf_kappa_chain(h, alpha=50.0, direction="+y"):
    """omega-kappa-phi chain for a Pilatus/Eiger mini-CBF header (frame: rotation axis +x,
    beam +z). Assumes a standard mini-kappa (alpha 50 deg, kappa axis tilted from the omega axis
    towards ``direction``), the convention used by DIALS' KappaGoniometer."""
    c, s = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
    kv = {"+y": (c, s, 0), "-y": (c, -s, 0), "+z": (c, 0, s), "-z": (c, 0, -s)}[direction]
    inc = float(h.get("angle_increment") or 0.0)
    osc_axis = str(h.get("oscillation_axis", "OMEGA")).upper()
    axes = []
    for name, vec in (("omega", (1, 0, 0)), ("kappa", kv), ("phi", (1, 0, 0))):
        v0 = float(h.get(name) or 0.0)
        scanned = name.upper() in osc_axis or (name == "omega" and not any(n in osc_axis for n in ("PHI", "KAPPA")))
        vals = np.array([v0, v0 + inc]) if scanned else np.array([v0])
        axes.append(Axis(name, np.array(vec, float), vals))
    return GonioChain(axes)

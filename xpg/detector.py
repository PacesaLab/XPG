"""Detector model: multi-segment (multi-panel) geometry and geometric-distortion tables (XPG conventions).

SEGMENT= semantics (multi-panel keyword of the legacy input): the pixel (IX, IY) of a segment is at

    d = QX (IX - ORGXS) EDS(:,1) + QY (IY - ORGYS) EDS(:,2) + FS EDS(:,3)

in the *detector* system, whose axes are ED1, ED2, ED3 and whose origin is ``-QX ORGX ED1 - QY ORGY ED2
+ F ED3`` away from the crystal, i.e. the lab position is

    X = (d1 - QX ORGX) ED1 + (d2 - QY ORGY) ED2 + (d3 + F) ED3.

A detector without SEGMENT= is the single segment ORGXS = ORGYS = FS = 0, EDS = identity.

Every segment therefore maps continuous pixel coordinates affinely to the lab:
``X = P0_k + (x - ORGX) Bx_k + (y - ORGY) By_k`` (reference point P0 at the global origin pixel keeps
float32 kernels precise). Points are assigned to segments by their raw pixel number, and segment
gaps are untrusted.

X-GEO_CORR= / Y-GEO_CORR= (distortion tables of the legacy input): "The geometrically corrected coordinates of a pixel at IX,IY are found
by adding table_value(IX,IY)/100" — applied to raw coordinates before the segment mapping
(bilinear between pixel centres). The inverse (for predictions) is a fixed-point iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


@dataclass
class Segment:
    x1: int                       # SEGMENT= x1 x2 y1 y2 (inclusive pixel numbers, 1-based)
    x2: int
    y1: int
    y2: int
    orgx: float = 0.0             # SEGMENT_ORGX (pixels)
    orgy: float = 0.0             # SEGMENT_ORGY
    distance: float = 0.0         # SEGMENT_DISTANCE (mm)
    ex: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))   # DIRECTION_OF_SEGMENT_X-AXIS
    ey: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))   # DIRECTION_OF_SEGMENT_Y-AXIS
    refine: tuple = ()            # REFINE_SEGMENT= (POSITION / ORIENTATION)

    def local(self, qx, qy):
        """(c, ex, ey): detector-system offset vector and orthonormal in-plane axes."""
        ex = _unit(self.ex)
        ey = _unit(np.asarray(self.ey, float) - np.dot(self.ey, ex) * ex)
        ez = np.cross(ex, ey)
        c = -qx * self.orgx * ex - qy * self.orgy * ey + self.distance * ez
        return c, ex, ey

    @classmethod
    def from_local(cls, rect, c, ex, ey, qx, qy, refine=()):
        ex = _unit(ex)
        ey = _unit(ey - np.dot(ey, ex) * ex)
        ez = np.cross(ex, ey)
        return cls(*rect, orgx=float(-(c @ ex) / qx), orgy=float(-(c @ ey) / qy), distance=float(c @ ez),
                   ex=ex, ey=ey, refine=tuple(refine))

    @property
    def rect(self):
        return (self.x1, self.x2, self.y1, self.y2)

    @property
    def centre(self):
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)


def default_segment(nx, ny):
    return Segment(1, max(nx, 1), 1, max(ny, 1))


def local_arrays(segments, qx, qy):
    """Stacked detector-system (C, EX, EY), each (K, 3)."""
    loc = [s.local(qx, qy) for s in segments]
    return tuple(np.array([l[i] for l in loc]) for i in range(3))


def panel_affine(C, EX, EY, orgx, orgy, dist, ed1, ed2, qx, qy, xp=np):
    """Lab-frame affine maps of all segments: (P0, Bx, By), each (K, 3), with
    X(x, y) = P0 + (x - orgx) Bx + (y - orgy) By."""
    ed3 = xp.cross(ed1, ed2)
    E = xp.stack([ed1, ed2, ed3], axis=0)         # rows
    Bx = qx * (EX @ E)
    By = qy * (EY @ E)
    origin = xp.stack([-qx * orgx * xp.ones(()), -qy * orgy * xp.ones(()), dist * xp.ones(())])
    A = (C + origin) @ E
    P0 = A + orgx * Bx + orgy * By
    return P0, Bx, By


def panel_affine_t(C, EX, EY, orgx, orgy, dist, ed1, ed2, qx, qy):
    """torch version of ``panel_affine`` (differentiable). Batched over leading dimensions of orgx / orgy / dist
    (...) and ed1 / ed2 (..., 3): P0, Bx, By of shape (..., segments, 3)."""
    import torch
    ed3 = torch.linalg.cross(ed1, ed2)
    E = torch.stack([ed1, ed2, ed3], -2)
    Bx = qx * (EX @ E)
    By = qy * (EY @ E)
    one = torch.ones((), dtype=E.dtype, device=E.device)
    origin = torch.stack([-qx * orgx * one, -qy * orgy * one, dist * one], -1)
    A = (C + origin[..., None, :]) @ E
    P0 = A + (orgx * one)[..., None, None] * Bx + (orgy * one)[..., None, None] * By
    return P0, Bx, By


@lru_cache(maxsize=16)
def _segmap_cached(rects, nx, ny):
    m = np.full((ny, nx), -1, np.int16)
    for k, (x1, x2, y1, y2) in enumerate(rects):
        m[max(y1 - 1, 0):min(y2, ny), max(x1 - 1, 0):min(x2, nx)] = k
    m.setflags(write=False)
    return m


def segment_map(segments, nx, ny):
    """(ny, nx) int16 array: segment index of each pixel, -1 in gaps."""
    return _segmap_cached(tuple(s.rect for s in segments), int(nx), int(ny))


def lookup_segment(segmap, x, y, xp=np):
    """Segment index of raw continuous coordinates (gaps -> -1)."""
    ix = xp.clip(xp.floor(x - 0.5).astype(xp.int32), 0, segmap.shape[1] - 1)
    iy = xp.clip(xp.floor(y - 0.5).astype(xp.int32), 0, segmap.shape[0] - 1)
    return segmap[iy, ix]


def bilinear_table(tab, x, y, xp=np):
    """Bilinear interpolation of a per-pixel table (value of array element ix at x = ix + 1)."""
    ny, nx = tab.shape
    fx = xp.clip(x - 1.0, 0, nx - 1.000001)
    fy = xp.clip(y - 1.0, 0, ny - 1.000001)
    ix = xp.floor(fx).astype(xp.int32)
    iy = xp.floor(fy).astype(xp.int32)
    ix1 = xp.minimum(ix + 1, nx - 1)
    iy1 = xp.minimum(iy + 1, ny - 1)
    tx, ty = fx - ix, fy - iy
    return (tab[iy, ix] * (1 - tx) * (1 - ty) + tab[iy, ix1] * tx * (1 - ty)
            + tab[iy1, ix] * (1 - tx) * ty + tab[iy1, ix1] * tx * ty)


def apply_geo_corr(geo, x, y, xp=np):
    if geo is None:
        return x, y
    return x + bilinear_table(geo[0], x, y, xp), y + bilinear_table(geo[1], x, y, xp)


def invert_geo_corr(geo, xc, yc, n_iter=4):
    if geo is None:
        return xc, yc
    x, y = xc, yc
    for _ in range(n_iter):
        x = xc - bilinear_table(geo[0], x, y)
        y = yc - bilinear_table(geo[1], x, y)
    return x, y


def read_geo_corr(xpath, ypath, nx, ny):
    """X-GEO_CORR / Y-GEO_CORR tables (CBF, values in 1/100 pixel) -> pixel offsets (float32)."""
    from .io.cbf import read_cbf
    out = []
    for pth in (xpath, ypath):
        if pth is None:
            out.append(np.zeros((ny, nx), np.float32))
            continue
        t, _ = read_cbf(pth)
        t = t.astype(np.float32) / 100.0
        if t.shape != (ny, nx):
            # tables sampled on a coarser grid: expand by the integer factor (nearest)
            fy, fx = ny / t.shape[0], nx / t.shape[1]
            iy = np.minimum((np.arange(ny) / fy).astype(int), t.shape[0] - 1)
            ix = np.minimum((np.arange(nx) / fx).astype(int), t.shape[1] - 1)
            t = t[np.ix_(iy, ix)]
        out.append(t)
    return tuple(out)


def parse_segments(d, qx_default=None):
    """Segments from a parsed legacy input ``{KEY: [[values], ...]}`` honouring keyword order: the
    SEGMENT_* / DIRECTION_OF_SEGMENT_* / REFINE_SEGMENT keywords after SEGMENT= belong to it."""
    segs = []
    order = d.get("__order__", [])
    cur = None
    for key, vals in order:
        if key == "SEGMENT":
            cur = Segment(*(int(float(v)) for v in vals[:4]))
            segs.append(cur)
        elif cur is None:
            continue
        elif key == "SEGMENT_ORGX":
            cur.orgx = float(vals[0])
        elif key == "SEGMENT_ORGY":
            cur.orgy = float(vals[0])
        elif key == "SEGMENT_DISTANCE":
            cur.distance = float(vals[0])
        elif key == "DIRECTION_OF_SEGMENT_X-AXIS":
            cur.ex = np.array([float(v) for v in vals[:3]])
        elif key == "DIRECTION_OF_SEGMENT_Y-AXIS":
            cur.ey = np.array([float(v) for v in vals[:3]])
        elif key == "REFINE_SEGMENT":
            cur.refine = tuple(v.upper() for v in vals)
    return segs


def format_segments(segments):
    v3 = lambda a: " ".join(f"{x:.7f}" for x in a)
    lines = []
    for s in segments:
        lines += [f" SEGMENT= {s.x1} {s.x2} {s.y1} {s.y2}",
                  f"  SEGMENT_ORGX= {s.orgx:.4f} SEGMENT_ORGY= {s.orgy:.4f} SEGMENT_DISTANCE= {s.distance:.4f}",
                  f"  DIRECTION_OF_SEGMENT_X-AXIS= {v3(_unit(s.ex))}",
                  f"  DIRECTION_OF_SEGMENT_Y-AXIS= {v3(_unit(s.ey))}"]
        if s.refine:
            lines.append(f"  REFINE_SEGMENT= {' '.join(s.refine)}")
    return lines

"""Diffraction geometry in the XPG conventions (Kabsch 1988, 2010).

Conventions
-----------
* Right-handed laboratory frame. ``S0 = beam_direction / wavelength`` (|S0| = 1/lambda).
* Detector: pixel coordinates (x, y) are continuous: the pixel with 0-based
  array index ``ix`` has its centre at ``x = ix + 1`` (spans ``[ix+0.5, ix+1.5)``).  The lab position of detector point (x, y) is
  ``X = QX (x-ORGX) ED1 + QY (y-ORGY) ED2 + F ED3`` with ``ED3 = ED1 x ED2``.
* Rotation: crystal at spindle angle phi relates to phi=0 by a right-handed rotation
  about ``ROTATION_AXIS``: ``p(phi) = R(axis, phi) p0``.
* Frames: frame ``i`` covers the continuous frame coordinate z in ``[i-1, i)``, and the
  spindle at z is ``phi(z) = STARTING_ANGLE + OSC * (z - (STARTING_FRAME - 1))``.
* Crystal: ``A*`` is a 3x3 matrix whose *columns* are a*, b*, c* in the lab frame at
  phi = 0; ``p0 = A* @ hkl``. The real-space basis (rows a, b, c) is ``inv(A*)``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from .config import Params

DEG = np.pi / 180.0
PIX0 = 1.0  # coordinate of the centre of array element 0 (XPG convention)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rotation_matrix(axis, angle):
    """Right-handed rotation matrix about unit ``axis`` by ``angle`` (radians). numpy."""
    u = unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return c * np.eye(3) + s * ux + (1 - c) * np.outer(u, u)


def _skew_t(u):
    """Cross-product matrices of (..., 3) vectors, shape (..., 3, 3)."""
    z = torch.zeros_like(u[..., 0])
    return torch.stack([torch.stack([z, -u[..., 2], u[..., 1]], -1),
                        torch.stack([u[..., 2], z, -u[..., 0]], -1),
                        torch.stack([-u[..., 1], u[..., 0], z], -1)], -2)


def rotvec_to_matrix_t(r):
    """Rotation vector(s) (axis * angle, shape (..., 3)) to rotation matrices; differentiable at 0. torch."""
    th2 = (r * r).sum(-1)[..., None, None]
    th = torch.sqrt(th2 + 1e-30)
    k = _skew_t(r)
    small = th2 < 1e-12
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / (th2 + 1e-30))
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    return eye + a * k + b * (k @ k)


# ---------------------------------------------------------------------------
# Unit cell helpers
# ---------------------------------------------------------------------------

def cell_to_B(cell):
    """Busing & Levy B matrix (columns = reciprocal axes in a crystal-fixed Cartesian frame)."""
    a, b, c, al, be, ga = cell
    al, be, ga = np.radians([al, be, ga])
    # real-space basis, a along x, b in xy plane
    ax = np.array([a, 0.0, 0.0])
    bx = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 1e-12))
    real = np.array([ax, bx, [cx, cy, cz]])  # rows a, b, c
    return np.linalg.inv(real)  # columns a*, b*, c*


def real_basis(Astar):
    """Rows a, b, c of the real-space basis for reciprocal basis columns ``Astar``."""
    return np.linalg.inv(Astar)


def cell_from_Astar(Astar):
    R = real_basis(Astar)
    a, b, c = (np.linalg.norm(R[i]) for i in range(3))
    al = np.degrees(np.arccos(np.dot(R[1], R[2]) / (b * c)))
    be = np.degrees(np.arccos(np.dot(R[0], R[2]) / (a * c)))
    ga = np.degrees(np.arccos(np.dot(R[0], R[1]) / (a * b)))
    return np.array([a, b, c, al, be, ga])


def cell_from_metric(G):
    a, b, c = np.sqrt(np.diag(G))
    al = np.degrees(np.arccos(G[1, 2] / (b * c)))
    be = np.degrees(np.arccos(G[0, 2] / (a * c)))
    ga = np.degrees(np.arccos(G[0, 1] / (a * b)))
    return np.array([a, b, c, al, be, ga])


def metric_from_cell(cell):
    a, b, c, al, be, ga = cell
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    return np.array([[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]])


def U_from_Astar(Astar, cell=None):
    """Orientation U such that Astar = U @ B(cell) (polar decomposition)."""
    if cell is None:
        cell = cell_from_Astar(Astar)
    M = Astar @ np.linalg.inv(cell_to_B(cell))
    u, _, vt = np.linalg.svd(M)
    U = u @ vt
    if np.linalg.det(U) < 0:
        raise ValueError("left-handed basis")
    return U


# ---------------------------------------------------------------------------
# Experimental geometry
# ---------------------------------------------------------------------------

_SMAP = {}   # one cached full-detector S map (Geometry.pixel_s_map)


@dataclass
class Geometry:
    wavelength: float
    beam: np.ndarray                 # unit incident beam direction
    axis: np.ndarray                 # unit rotation axis
    orgx: float
    orgy: float
    distance: float
    ed1: np.ndarray
    ed2: np.ndarray
    qx: float
    qy: float
    nx: int
    ny: int
    phi0: float                      # degrees, at start of starting_frame
    osc: float                       # degrees per frame
    starting_frame: int
    polarization_fraction: float = 0.5
    polarization_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    sensor_thickness: float = 0.0
    mu: float = 0.0                  # linear attenuation coefficient of sensor (1/mm)
    segments: list | None = None     # SEGMENT= definitions (None: single panel)
    geo_corr: tuple | None = None    # (dx, dy) per-pixel geometric corrections (pixels), X/Y-GEO_CORR

    @classmethod
    def from_params(cls, p: Params) -> "Geometry":
        ed1 = unit(p.ed1)
        ed2 = unit(p.ed2 - np.dot(p.ed2, ed1) * ed1)
        geo = None
        if p.x_geo_corr or p.y_geo_corr:
            from .detector import read_geo_corr
            rp = lambda f: None if f is None else (f if Path(f).is_absolute() else str(Path(p.workdir) / f))
            geo = read_geo_corr(rp(p.x_geo_corr), rp(p.y_geo_corr), p.nx, p.ny)
        return cls(wavelength=p.wavelength, beam=unit(p.beam_direction), axis=unit(p.rotation_axis),
                   orgx=p.orgx, orgy=p.orgy, distance=p.distance, ed1=ed1, ed2=ed2, qx=p.qx, qy=p.qy,
                   nx=p.nx, ny=p.ny, phi0=p.starting_angle, osc=p.osc, starting_frame=p.phi_start_frame,
                   polarization_fraction=p.polarization_fraction,
                   polarization_normal=unit(p.polarization_normal),
                   sensor_thickness=p.sensor_thickness,
                   mu=(p.silicon_mu if p.silicon_mu else sensor_mu(p.wavelength, p.sensor_material))
                   if p.sensor_thickness > 0 else 0.0,
                   segments=[copy.deepcopy(s_) for s_ in p.segments] or None, geo_corr=geo)

    def copy(self, **kw) -> "Geometry":
        if "segments" not in kw and self.segments is not None:
            kw["segments"] = [copy.deepcopy(s_) for s_ in self.segments]
        return replace(self, **kw)

    @property
    def ed3(self):
        return np.cross(self.ed1, self.ed2)

    @property
    def s0(self):
        return self.beam / self.wavelength

    # --- conversions ---
    def z_to_phi(self, z):
        """Continuous frame coordinate -> spindle angle (degrees)."""
        return self.phi0 + self.osc * (np.asarray(z) - (self.starting_frame - 1))

    def phi_to_z(self, phi):
        return (np.asarray(phi) - self.phi0) / self.osc + (self.starting_frame - 1)

    @property
    def has_parallax(self):
        return self.sensor_thickness > 0 and self.mu > 0

    # --- detector panels (see detector.py) ---
    @property
    def segmented(self):
        return bool(self.segments)

    def segment_list(self):
        from .detector import default_segment
        return self.segments if self.segments else [default_segment(self.nx, self.ny)]

    def panels(self):
        """Lab affine maps (P0, Bx, By), each (K, 3): X = P0 + (x-orgx) Bx + (y-orgy) By."""
        from .detector import local_arrays, panel_affine
        C, EX, EY = local_arrays(self.segment_list(), self.qx, self.qy)
        return panel_affine(C, EX, EY, self.orgx, self.orgy, self.distance, unit(self.ed1), unit(self.ed2),
                            self.qx, self.qy)

    def segmap(self):
        from .detector import segment_map
        return segment_map(self.segments, self.nx, self.ny) if self.segmented else None

    def segment_of(self, x, y):
        """Segment index of raw pixel coordinates (0 for a single panel, -1 in segment gaps)."""
        x = np.asarray(x, float)
        if not self.segmented:
            return np.zeros(x.shape, int)
        from .detector import lookup_segment
        return lookup_segment(self.segmap(), x, np.asarray(y, float)).astype(int)

    def corrected_xy(self, x, y):
        from .detector import apply_geo_corr
        return apply_geo_corr(self.geo_corr, np.asarray(x, float), np.asarray(y, float))

    def _panel_axes(self, k):
        P0, Bx, By = self.panels()
        u1 = Bx[k] / self.qx
        u2 = By[k] / self.qy
        return P0[k], Bx[k], By[k], u1, u2, np.cross(u1, u2)

    def _plane_lab(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if not self.segmented and self.geo_corr is None:
            return self.qx * (x[..., None] - self.orgx) * self.ed1 + self.qy * (y[..., None] - self.orgy) * self.ed2 \
                + self.distance * self.ed3
        k = np.maximum(self.segment_of(x, y), 0)
        xc, yc = self.corrected_xy(x, y)
        P0, Bx, By, *_ = self._panel_axes(k)
        return P0 + (xc - self.orgx)[..., None] * Bx + (yc - self.orgy)[..., None] * By

    def pixel_to_lab(self, x, y):
        """Lab coordinates (mm) of the sensor-surface entry point of the ray recorded at pixel (x, y),
        i.e. observed position corrected for geometric distortion, segment geometry and parallax."""
        if not self.has_parallax:
            return self._plane_lab(x, y)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        k = np.maximum(self.segment_of(x, y), 0)
        xc, yc = self.corrected_xy(x, y)
        P0, Bx, By, u1, u2, un = self._panel_axes(k)
        lab = lambda a, b: P0 + (a - self.orgx)[..., None] * Bx + (b - self.orgy)[..., None] * By
        X = lab(xc, yc)
        for _ in range(4):
            e = X / np.linalg.norm(X, axis=-1, keepdims=True)
            d1, d2 = parallax_offset_local(e, u1, u2, un, self.sensor_thickness, self.mu)
            X = lab(xc - d1 / self.qx, yc - d2 / self.qy)
        return X

    def pixel_to_s(self, x, y):
        """Diffracted beam wavevector S (|S| = 1/lambda) for detector points."""
        X = self.pixel_to_lab(x, y)
        return X / (np.linalg.norm(X, axis=-1, keepdims=True) * self.wavelength)

    def s_to_pixel(self, S, return_segment=False):
        """Raw pixel coordinates where rays S are recorded (nan if they miss all segments / the plane)."""
        S = np.asarray(S, float)
        from .detector import invert_geo_corr
        P0, Bx, By = self.panels()
        segs = self.segment_list()
        x = np.full(S.shape[:-1], np.nan)
        y = np.full(S.shape[:-1], np.nan)
        kk = np.full(S.shape[:-1], -1)
        for k in range(len(segs)):
            n = np.cross(Bx[k], By[k])
            sn = S @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (P0[k] @ n) / sn
            P = t[..., None] * S - P0[k]
            xk = self.orgx + (P @ Bx[k]) / (Bx[k] @ Bx[k])
            yk = self.orgy + (P @ By[k]) / (By[k] @ By[k])
            ok = np.isfinite(t) & (t > 0)
            if self.has_parallax:
                u1, u2 = Bx[k] / self.qx, By[k] / self.qy
                e = S / np.linalg.norm(S, axis=-1, keepdims=True)
                with np.errstate(invalid="ignore"):
                    d1, d2 = parallax_offset_local(e, u1, u2, np.cross(u1, u2), self.sensor_thickness, self.mu)
                xk = xk + d1 / self.qx
                yk = yk + d2 / self.qy
            xk, yk = invert_geo_corr(self.geo_corr, xk, yk)
            if self.segmented:
                s_ = segs[k]
                ok &= (xk >= s_.x1 - 0.5) & (xk < s_.x2 + 0.5) & (yk >= s_.y1 - 0.5) & (yk < s_.y2 + 0.5)
            ok &= np.isnan(x)
            x = np.where(ok, xk, x)
            y = np.where(ok, yk, y)
            kk = np.where(ok, k, kk)
        if return_segment:
            return x, y, kk
        return x, y

    def panel_normal_at(self, x, y):
        """Unit sensor normal (lab) of the segment containing raw pixel (x, y)."""
        if not self.segmented:
            return np.broadcast_to(self.ed3, np.shape(x) + (3,))
        _, _, _, _, _, un = self._panel_axes(np.maximum(self.segment_of(x, y), 0))
        return un

    def spot_to_p0(self, x, y, z):
        """Reciprocal-lattice vector at phi=0 for an observed spot (x, y, z)."""
        S = self.pixel_to_s(x, y)
        p = S - self.s0
        phi = np.radians(self.z_to_phi(z))
        R = rotation_matrices_np(self.axis, -phi)
        return np.einsum("nij,nj->ni", R, p)

    def resolution_of_pixel(self, x, y):
        S = self.pixel_to_s(x, y)
        return 1.0 / np.linalg.norm(S - self.s0, axis=-1)

    def _fingerprint(self):
        parts = []
        for name in self.__dataclass_fields__:
            v = getattr(self, name)
            arrs = [v] if isinstance(v, np.ndarray) else \
                list(v) if isinstance(v, (tuple, list)) and v and isinstance(v[0], np.ndarray) else None
            if arrs is not None:      # content (small arrays exactly, large ones by sum + strided sample)
                parts.append((name, tuple((a.shape, a.tobytes() if a.size <= 64 else
                                           (float(np.nansum(a)), a.ravel()[::max(1, a.size // 4096)].tobytes()))
                                          for a in arrs)))
            else:
                parts.append((name, repr(v)))
        return tuple(parts)

    def pixel_s_map(self):
        """S of every pixel centre, (ny, nx, 3), cached for the current geometry: BACKGROUND, the background template and
        MASK each need the full-detector map (with parallax ~4 s per evaluation on an Eiger 16M)."""
        key = self._fingerprint()
        if _SMAP.get("key") != key:
            _SMAP.clear()
            ys, xs = np.mgrid[0:self.ny, 0:self.nx]
            S = self.pixel_to_s(xs + PIX0, ys + PIX0)
            S.flags.writeable = False
            _SMAP["S"] = S
            _SMAP["key"] = key
        return _SMAP["S"]

    def resolution_map(self, step=1):
        if step == 1:
            return 1.0 / np.linalg.norm(self.pixel_s_map() - self.s0, axis=-1)
        ys, xs = np.mgrid[0:self.ny:step, 0:self.nx:step]
        return self.resolution_of_pixel(xs + PIX0, ys + PIX0)

    def max_resolution_at_edge(self):
        corners = np.array([[0.5, 0.5], [self.nx + .5, 0.5], [0.5, self.ny + .5], [self.nx + .5, self.ny + .5]], float)
        d = self.resolution_of_pixel(corners[:, 0], corners[:, 1])
        return d.min()

    def resolution_at_edge_inscribed(self):
        """Resolution at the closest detector edge from the beam (complete circle)."""
        xb, yb = self.beam_pixel()
        pts = np.array([[0.5, yb], [self.nx + .5, yb], [xb, 0.5], [xb, self.ny + .5]], float)
        return self.resolution_of_pixel(pts[:, 0], pts[:, 1]).max()

    def beam_pixel(self):
        return self.s_to_pixel(self.s0[None, :])[0][0], self.s_to_pixel(self.s0[None, :])[1][0]

    def absorption_factors(self, S, air=True):
        """Multiplicative factor by which recorded intensities are attenuated: sensor efficiency
        (if the thickness is known) times air transmission along the crystal-detector path."""
        e = S / np.linalg.norm(S, axis=-1, keepdims=True)
        f = np.ones(e.shape[:-1])
        if self.segmented:
            x, y = self.s_to_pixel(S)
            n = self.panel_normal_at(np.nan_to_num(x, nan=1.0), np.nan_to_num(y, nan=1.0))
            cosn = np.einsum("...k,...k->...", e, n)
        else:
            cosn = e @ self.ed3
        if self.sensor_thickness > 0 and self.mu > 0:
            f = f * -np.expm1(-self.mu * self.sensor_thickness / np.clip(cosn, 1e-3, 1.0))
        if air:
            mu_air = mu_linear(self.wavelength, "AIR")
            if self.segmented:
                path = np.linalg.norm(self.pixel_to_lab(np.nan_to_num(x, nan=1.0), np.nan_to_num(y, nan=1.0)), axis=-1)
            else:
                path = self.distance / np.clip(e @ self.ed3, 1e-3, 1.0)
            f = f * np.exp(-mu_air * path)
        return f

    def lp_factor(self, S):
        """Return (zeta, polarization P) for diffracted wavevectors S (..., 3).

        Integrated intensities are corrected as I_corr = I_raw * |zeta| / P.
        """
        e = S / np.linalg.norm(S, axis=-1, keepdims=True)
        e0 = self.beam
        zeta = (np.cross(e, e0) @ self.axis)
        n = self.polarization_normal - np.dot(self.polarization_normal, e0) * e0
        n = n / np.linalg.norm(n)
        emain = np.cross(n, e0)
        f = self.polarization_fraction
        P = f * (1 - (e @ emain) ** 2) + (1 - f) * (1 - (e @ n) ** 2)
        return zeta, P


def rotation_matrices_np(axis, angles):
    u = unit(axis)
    angles = np.asarray(angles, float)
    c = np.cos(angles)[..., None, None]
    s = np.sin(angles)[..., None, None]
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return c * np.eye(3) + s * ux + (1 - c) * np.outer(u, u)


# Photo-absorption mass attenuation coefficients (cm^2/g) vs energy (keV), from NIST XCOM
# (photoelectric only: scattered photons may escape the sensor) and the
# total attenuation of air. Interpolated log-log; edges of CdTe included explicitly.
_MU_TABLES = {
    "SI": (2.33, [(5, 242.0), (6, 145.5), (8, 62.0), (10, 32.87), (15, 9.76), (20, 4.06), (30, 1.15),
                  (40, 0.476), (50, 0.240), (60, 0.137), (80, 0.0566), (100, 0.0287)]),
    "CDTE": (5.85, [(8, 250.0), (10, 132.0), (15, 44.0), (20, 21.0), (25, 12.0), (26.70, 10.1), (26.72, 36.0),
                    (31.80, 23.5), (31.82, 40.5), (40, 22.5), (50, 12.5), (60, 7.6), (80, 3.5), (100, 1.95)]),
    "GAAS": (5.32, [(8, 37.0), (10, 20.5), (11.0, 16.0), (11.1, 95.0), (11.8, 77.0), (11.9, 150.0), (15, 80.0),
                    (20, 37.0), (30, 12.0), (40, 5.3), (60, 1.7), (100, 0.4)]),
    "AIR": (1.205e-3, [(5, 39.31), (6, 22.70), (8, 9.921), (10, 5.120), (15, 1.614), (20, 0.7779), (30, 0.3538),
                       (40, 0.2485), (50, 0.2080), (60, 0.1875), (80, 0.1662), (100, 0.1541)]),
}


def mu_linear(wavelength, material="Si"):
    """Linear attenuation coefficient (1/mm) of a material at the given wavelength (A)."""
    key = str(material).upper().replace(" ", "")
    if key in ("SILICON",):
        key = "SI"
    if key not in _MU_TABLES:
        return 0.0
    rho, tab = _MU_TABLES[key]
    E = 12.398419843320026 / wavelength
    e = np.array([t[0] for t in tab])
    m = np.array([t[1] for t in tab])
    mu_rho = float(np.exp(np.interp(np.log(E), np.log(e), np.log(m))))
    return mu_rho * rho / 10.0


def sensor_mu(wavelength, material="Si"):
    return mu_linear(wavelength, material)


def parallax_offset_local(e, u1, u2, un, t, mu, xp=np):
    """Parallax shift of the absorption-weighted mean depth (sensor thickness t, attenuation mu) along the panel axes
    u1, u2 of each point (..., 3): the mean interaction depth along the ray e projected on the panel."""
    dot = lambda a, b: xp.sum(a * b, axis=-1)
    cosa = xp.clip(dot(e, un), 1e-3, 1.0)
    L = t / cosa
    muL = mu * L
    lbar = 1.0 / mu - L * xp.exp(-muL) / xp.maximum(-xp.expm1(-muL), 1e-12)
    return lbar * dot(e, u1), lbar * dot(e, u2)


# ---------------------------------------------------------------------------
# Reflection generation / prediction
# ---------------------------------------------------------------------------

def generate_hkl(Astar, dmin, dmax=1e9):
    """All Miller indices with dmin <= d < dmax (excluding 000)."""
    R = real_basis(Astar)
    hmax = [int(np.ceil(np.linalg.norm(R[i]) / dmin)) for i in range(3)]
    h = np.arange(-hmax[0], hmax[0] + 1)
    k = np.arange(-hmax[1], hmax[1] + 1)
    l = np.arange(-hmax[2], hmax[2] + 1)
    H = np.stack(np.meshgrid(h, k, l, indexing="ij"), -1).reshape(-1, 3)
    p = H @ Astar.T
    d2inv = np.einsum("ij,ij->i", p, p)
    keep = (d2inv <= 1.0 / dmin ** 2) & (d2inv > 1.0 / dmax ** 2)
    return H[keep].astype(np.int32)


def predict_phi(geom: Geometry, p0):
    """Spindle angles (radians, in (-pi, pi]) at which p0 vectors are in diffracting condition.

    Returns (phi_a, phi_b, valid). Solve S0 . R(phi) p0 = -|p0|^2 / 2.
    """
    m = geom.axis
    s0 = geom.s0
    p0 = np.asarray(p0, float)
    par = (p0 @ m)[:, None] * m
    perp = p0 - par
    mxp = np.cross(m, p0)
    a = perp @ s0
    b = mxp @ s0
    c = -0.5 * np.einsum("ij,ij->i", p0, p0) - par @ s0
    r = np.hypot(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = c / r
    valid = np.abs(ratio) <= 1.0
    base = np.arctan2(b, a)
    delta = np.arccos(np.clip(ratio, -1, 1))
    return base + delta, base - delta, valid


def predict_reflections(geom: Geometry, Astar, dmin, dmax=1e9, z_range=None, hkl=None):
    """Predict reflections (rotation method).

    Returns a dict of arrays: hkl, x, y, z (frame coordinate), phi (deg), S (lab, at diffraction),
    zeta, polarization. ``z_range`` = (zmin, zmax) continuous frame coordinates.
    """
    if hkl is None:
        hkl = generate_hkl(Astar, dmin, dmax)
    p0 = hkl @ Astar.T
    pa, pb, valid = predict_phi(geom, p0)
    out = {k: [] for k in ("hkl", "phi", "S")}
    if z_range is None:
        z_range = (geom.starting_frame - 1, geom.starting_frame - 1 + 360.0 / abs(geom.osc))
    phimin, phimax = sorted(geom.z_to_phi(np.array(z_range, float)))
    for ph in (pa, pb):
        ph = np.degrees(ph[valid])
        # bring into [phimin, phimin+360)
        ph = phimin + np.mod(ph - phimin, 360.0)
        for shift in (0.0, 360.0):  # handle sweeps longer than 360
            phs = ph + shift
            ok = (phs >= phimin) & (phs < phimax)
            if not ok.any():
                continue
            H = hkl[valid][ok]
            P0 = p0[valid][ok]
            R = rotation_matrices_np(geom.axis, np.radians(phs[ok]))
            S = np.einsum("nij,nj->ni", R, P0) + geom.s0
            out["hkl"].append(H)
            out["phi"].append(phs[ok])
            out["S"].append(S)
    if not out["hkl"]:
        return {"hkl": np.zeros((0, 3), np.int32), "phi": np.zeros(0), "S": np.zeros((0, 3)),
                "x": np.zeros(0), "y": np.zeros(0), "z": np.zeros(0), "zeta": np.zeros(0),
                "polarization": np.zeros(0)}
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["x"], res["y"] = geom.s_to_pixel(res["S"])
    res["z"] = geom.phi_to_z(res["phi"])
    res["zeta"], res["polarization"] = geom.lp_factor(res["S"])
    ok = np.isfinite(res["x"])
    res = {k: v[ok] for k, v in res.items()}
    return res



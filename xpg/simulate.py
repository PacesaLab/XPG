"""Synthetic rotation-data simulator.

Spots are 3D Gaussians: sigma_D (beam divergence) on the detector and a rocking curve of
width sigma_M/|zeta| in phi; intensities follow Wilson statistics with an (optionally
anisotropic) B-factor, Lorentz/polarisation, a smooth phi-dependent scale and Poisson noise."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.special import erf

from .config import Params, format_xds_inp
from .geometry import PIX0, Geometry, mu_linear, cell_to_B, predict_reflections, rotation_matrix, unit
from .io.cbf import write_cbf


@dataclass
class SimSpec:
    cell: tuple = (61.2, 72.5, 84.3, 90.0, 90.0, 90.0)
    space_group: str = "P 21 21 21"
    wavelength: float = 0.9795
    nx: int = 1024
    ny: int = 1100
    qx: float = 0.172
    distance: float = 160.0
    orgx: float = 520.3
    orgy: float = 540.7
    osc: float = 0.5
    nframes: int = 180
    phi0: float = 0.0
    sigma_d: float = 0.035          # deg
    sigma_m: float = 0.12           # deg
    wilson_b: float = 18.0
    aniso_b: tuple = (0.0, 0.0, 0.0)  # extra B along a*, b*, c* (orthogonal approx.)
    anom: float = 0.0               # r.m.s. Bijvoet ratio |I+ - I-| / <I> (approx.)
    dispersive: float = 0.0         # r.m.s. fractional dispersive change (same for I+ and I-; MAD remote vs peak)
    sensor_thickness: float = 0.0   # mm; > 0 enables parallax + efficiency + air absorption
    rocking_split: float = 0.0      # > 0: bimodal rocking curve, two components at +-split*sigma_m
    spot_tail: float = 0.0          # fraction of each spot in a broad component (mosaic-block halo: non-Gaussian tails)
    spot_tail_scale: float = 3.0    # width of that component relative to the core (detector plane and rocking curve)
    structure_seed: int | None = None  # intensities (structure); default = seed. Same value = same crystal form
    twin_law: str = ""              # e.g. "k,h,-l" (merohedral twin operator)
    twin_fraction: float = 0.0
    ncs_op: str = ""                # rotational NCS pseudo-symmetry operator on hkl (e.g. "-h,-k,l"; not in the space group)
    ncs_c: float = 0.0              # NCS coupling at s = 0: F(h) = sqrt(1-c) G(h) + sqrt(c) G(R h), CC(I(h), I(Rh)) = 4c(1-c)
    ncs_b: float = 0.0              # imperfect NCS: c(s) = ncs_c exp(-ncs_b s^2 / 2) (A^2)
    tncs_vector: tuple = ()         # e.g. (0.5, 0.5, 0.0): pseudo-translation
    tncs_strength: float = 0.0      # I *= 1 + s cos(2 pi h.t)
    extra_lattice_rot: float = 0.0  # > 0: second crystal (same structure) rotated by this angle (deg)
    extra_lattice_scale: float = 0.6
    extra_lattice_sigma_m: float = 0.0  # > 0: the second crystal's reflecting range (deg; default: sigma_m)
    extra_lattice_sigma_d: float = 0.0  # > 0: its beam divergence / spot width (deg; default: sigma_d)
    extra_lattice_noniso: float = 0.0   # > 0: the second crystal's intensities are this fraction of an unrelated
                                        # structure's (non-isomorphous crystal; 0.5 -> CC ~0.7 to the first)
    sensor_material: str = "Si"
    mean_i0: float = 1.5e4          # mean counts at low resolution
    background: float = 2.0
    ring_d: float = 3.9             # resolution of a diffuse solvent ring
    ring_amp: float = 3.0
    ice_powder: float = 0.0         # counts per pixel at the strongest hexagonal-ice powder ring (on every frame)
    ice_spots: int = 0              # spots per frame from ice crystallites on the ice rings (spotty ice)
    ice_spot_counts: float = 3000.0  # mean counts of an ice spot
    dmin: float = 1.6
    scale_decay: float = 0.25       # fractional loss over sweep
    gaps: bool = True
    seed: int = 1
    beam_direction: tuple = (0.0, 0.0, 1.0)
    rotation_axis: tuple = (1.0, 0.0, 0.0)
    polarization_fraction: float = 0.99
    U: np.ndarray | None = None
    extra_dark_rect: list = field(default_factory=list)  # (x0,x1,y0,y1) beamstop-like shadows
    segments: list | None = None    # true detector segments (xpg.detector.Segment); gaps are masked
    geo_corr: tuple | None = None   # true geometric distortion (dx, dy) tables in pixels


def make_geometry(spec: SimSpec) -> Geometry:
    return Geometry(wavelength=spec.wavelength, beam=unit(spec.beam_direction), axis=unit(spec.rotation_axis),
                    orgx=spec.orgx, orgy=spec.orgy, distance=spec.distance, ed1=np.array([1.0, 0, 0]),
                    ed2=np.array([0, 1.0, 0]), qx=spec.qx, qy=spec.qx, nx=spec.nx, ny=spec.ny,
                    phi0=spec.phi0, osc=spec.osc, starting_frame=1,
                    polarization_fraction=spec.polarization_fraction,
                    polarization_normal=np.array([0, 1.0, 0]), sensor_thickness=spec.sensor_thickness,
                    mu=mu_linear(spec.wavelength, spec.sensor_material) if spec.sensor_thickness > 0 else 0.0,
                    segments=spec.segments, geo_corr=spec.geo_corr)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _hash_uniform(key, salt):
    """Deterministic uniforms in (0,1) from integer keys (consistent for any reflection subset)."""
    with np.errstate(over="ignore"):   # uint64 arithmetic wraps modulo 2^64 (intended)
        x = np.asarray(key, np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64((salt * 0xBF58476D1CE4E5B9) % 2 ** 64)
        x ^= x >> np.uint64(31)
        x = x * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(29)
    return (x >> np.uint64(11)).astype(np.float64) / float(2 ** 53) * (1 - 2e-12) + 1e-12


def _base_intensities(spec: SimSpec, hkl):
    from .symops import to_asu, reflection_props, d_spacing
    hasu, _, plus = to_asu(hkl, spec.space_group)
    key = ((hasu[:, 0].astype(np.int64) + 512) * 1048576 + (hasu[:, 1] + 512) * 1024 + (hasu[:, 2] + 512)).astype(np.uint64)
    ss = spec.seed if spec.structure_seed is None else spec.structure_seed
    u1 = _hash_uniform(key, ss + 1)
    u2 = _hash_uniform(key, ss + 2)
    u3 = _hash_uniform(key, ss + 3)
    g = np.sqrt(-2 * np.log(u2)) * np.cos(2 * np.pi * u3)          # standard normal (Box-Muller)
    s2 = 1.0 / d_spacing(hasu, spec.cell) ** 2
    centric, eps, absent = reflection_props(hasu, spec.space_group)
    # Wilson: <I> = eps * K * exp(-B s^2 / 2), s = 1/d ; anisotropy along reciprocal axes
    rec = np.array([1.0 / spec.cell[0], 1.0 / spec.cell[1], 1.0 / spec.cell[2]])
    hs = hasu * rec
    aniso = np.exp(-0.5 * (spec.aniso_b[0] * hs[:, 0] ** 2 + spec.aniso_b[1] * hs[:, 1] ** 2
                           + spec.aniso_b[2] * hs[:, 2] ** 2))
    mean = spec.mean_i0 * eps * np.exp(-0.5 * spec.wilson_b * s2) * aniso
    I = np.where(centric, g ** 2 * mean, -np.log(u1) * mean)       # centric chi^2_1, acentric exponential
    if spec.anom > 0:
        ga = np.sqrt(-2 * np.log(_hash_uniform(key, ss + 4))) * np.cos(2 * np.pi * _hash_uniform(key, ss + 5))
        sgn = np.where(plus, 1.0, -1.0)
        I = np.where(centric, I, I * np.clip(1.0 + spec.anom * ga * sgn, 0.05, None))
    if spec.dispersive > 0:
        gd = np.sqrt(-2 * np.log(_hash_uniform(key, ss + 6))) * np.cos(2 * np.pi * _hash_uniform(key, ss + 7))
        I = I * np.clip(1.0 + spec.dispersive * gd, 0.05, None)
    I[absent] = 0.0
    return I, hasu


def _complex_g(spec: SimSpec, hkl):
    """Unit-variance complex Gaussian per reflection, consistent under the space group (keyed by the asu index)."""
    from .symops import to_asu
    hasu, _, _ = to_asu(hkl, spec.space_group)
    key = ((hasu[:, 0].astype(np.int64) + 512) * 1048576 + (hasu[:, 1] + 512) * 1024 + (hasu[:, 2] + 512)).astype(np.uint64)
    ss = (spec.seed if spec.structure_seed is None else spec.structure_seed) + 100
    a = np.sqrt(-2 * np.log(_hash_uniform(key, ss + 1))) * np.cos(2 * np.pi * _hash_uniform(key, ss + 2))
    b = np.sqrt(-2 * np.log(_hash_uniform(key, ss + 3))) * np.cos(2 * np.pi * _hash_uniform(key, ss + 4))
    return (a + 1j * b) / np.sqrt(2.0)


def ncs_intensities(spec: SimSpec, hkl, I):
    """Rotational NCS pseudo-symmetry: acentric intensities replaced by |sqrt(1-c) G(h) + sqrt(c) G(Rh)|^2 x their
    Wilson mean (Wilson statistics stay those of an untwinned crystal; mates correlate at 4c(1-c), decaying with s)."""
    import gemmi
    from .symops import apply_rot, op_matrix, reflection_props, d_spacing, to_asu
    R = op_matrix(gemmi.Op(spec.ncs_op))
    hasu, _, _ = to_asu(hkl, spec.space_group)
    centric, eps, absent = reflection_props(hasu, spec.space_group)
    s2 = 1.0 / d_spacing(hasu, spec.cell) ** 2
    c = spec.ncs_c * np.exp(-0.5 * spec.ncs_b * s2)
    F = np.sqrt(1 - c) * _complex_g(spec, hkl) + np.sqrt(c) * _complex_g(spec, apply_rot(hkl, R))
    rec = np.array([1.0 / spec.cell[0], 1.0 / spec.cell[1], 1.0 / spec.cell[2]])
    hs = hasu * rec
    aniso = np.exp(-0.5 * (spec.aniso_b[0] * hs[:, 0] ** 2 + spec.aniso_b[1] * hs[:, 1] ** 2 + spec.aniso_b[2] * hs[:, 2] ** 2))
    mean = spec.mean_i0 * eps * np.exp(-0.5 * spec.wilson_b * s2) * aniso
    out = np.where(centric | absent, I, np.abs(F) ** 2 * mean)
    out[absent] = 0.0
    return out


def true_intensities(spec: SimSpec, hkl: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Wilson-distributed intensities, consistent under the space-group symmetry; optional
    merohedral twinning I = (1-a) I(h) + a I(T h). Returns (I_true for each hkl, asu hkl)."""
    I, hasu = _base_intensities(spec, hkl)
    if spec.ncs_op and spec.ncs_c > 0:
        I = ncs_intensities(spec, hkl, I)
    if spec.tncs_strength > 0 and len(spec.tncs_vector) == 3:
        I = I * (1 + spec.tncs_strength * np.cos(2 * np.pi * (hasu @ np.array(spec.tncs_vector, float))))
    if spec.twin_fraction > 0 and spec.twin_law:
        import gemmi
        from .symops import apply_rot, op_matrix
        R = op_matrix(gemmi.Op(spec.twin_law))
        I2, _ = _base_intensities(spec, apply_rot(hkl, R))
        I = (1 - spec.twin_fraction) * I + spec.twin_fraction * I2
    return I, hasu


def _open_eiger(outdir, spec, mask):
    import h5py
    import hdf5plugin
    data_name = "sim_data_000001.h5"
    fd = h5py.File(outdir / data_name, "w")
    ds = fd.create_dataset("/entry/data/data", shape=(spec.nframes, spec.ny, spec.nx), dtype=np.uint32,
                           chunks=(1, spec.ny, spec.nx), **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    ds.attrs["image_nr_low"] = 1
    ds.attrs["image_nr_high"] = spec.nframes
    fm = h5py.File(outdir / "sim_master.h5", "w")
    fm["/entry/data/data_000001"] = h5py.ExternalLink(data_name, "/entry/data/data")
    det = fm.create_group("/entry/instrument/detector")
    det["x_pixel_size"] = spec.qx * 1e-3
    det["y_pixel_size"] = spec.qx * 1e-3
    det["beam_center_x"] = spec.orgx
    det["beam_center_y"] = spec.orgy
    det["detector_distance"] = spec.distance * 1e-3
    det["sensor_thickness"] = 450e-6
    det["sensor_material"] = "Si"
    det["description"] = "Dectris EIGER2 Si (simulated)"
    dsp = det.create_group("detectorSpecific")
    dsp["x_pixels_in_detector"] = spec.nx
    dsp["y_pixels_in_detector"] = spec.ny
    dsp["countrate_correction_count_cutoff"] = 2 ** 20
    dsp["pixel_mask"] = (~mask).astype(np.uint32)
    fm["/entry/instrument/beam/incident_wavelength"] = spec.wavelength
    gon = fm.create_group("/entry/sample/goniometer")
    gon["omega"] = spec.phi0 + spec.osc * np.arange(spec.nframes)
    gon["omega_range_average"] = spec.osc
    _nexus_classes(fm)
    fm.close()
    return fd, ds


def _nexus_classes(fm):
    for path, cls in [("/entry", "NXentry"), ("/entry/data", "NXdata"), ("/entry/instrument", "NXinstrument"),
                      ("/entry/instrument/detector", "NXdetector"), ("/entry/instrument/beam", "NXbeam"),
                      ("/entry/sample", "NXsample"), ("/entry/sample/goniometer", "NXtransformations")]:
        if path in fm:
            fm[path].attrs["NX_class"] = np.bytes_(cls)


def simulate(outdir: str | Path, spec: SimSpec | None = None, write_images=True, verbose=True, fmt="cbf"):
    spec = spec or SimSpec()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    U = spec.U if spec.U is not None else random_rotation(rng)
    Astar = U @ cell_to_B(spec.cell)
    geom = make_geometry(spec)
    pred = predict_reflections(geom, Astar, spec.dmin, z_range=(-5.0, spec.nframes + 5.0))
    I_true, hasu = true_intensities(spec, pred["hkl"], rng)
    pred["lattice"] = np.zeros(len(I_true), int)
    if spec.extra_lattice_rot > 0:
        ax = unit(rng.normal(size=3))
        A2 = rotation_matrix(ax, np.radians(spec.extra_lattice_rot)) @ Astar
        p2 = predict_reflections(geom, A2, spec.dmin, z_range=(-5.0, spec.nframes + 5.0))
        I2, h2 = true_intensities(spec, p2["hkl"], rng)
        if spec.extra_lattice_noniso > 0:
            other = replace(spec, structure_seed=(spec.seed if spec.structure_seed is None else spec.structure_seed) + 7919)
            I2 = (1 - spec.extra_lattice_noniso) * I2 + spec.extra_lattice_noniso * true_intensities(other, p2["hkl"], rng)[0]
        p2["lattice"] = np.ones(len(I2), int)
        pred = {k: np.concatenate([pred[k], p2[k]]) for k in pred}
        I_true = np.concatenate([I_true, I2 * spec.extra_lattice_scale])
        hasu = np.concatenate([hasu, h2])
    zeta = np.abs(pred["zeta"])                       # Lorentz zeta = |m.(e x e0)|
    e1 = np.cross(pred["S"], geom.s0)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    zeta_u = np.abs(e1 @ geom.axis)                   # Kabsch zeta = |m.e1| (rocking width)
    keep = (zeta > 1e-3) & (zeta_u > 0.03)
    pred = {k: v[keep] for k, v in pred.items()}
    I_true, hasu, zeta, zeta_u = I_true[keep], hasu[keep], zeta[keep], zeta_u[keep]
    phi_frac = (pred["phi"] - spec.phi0) / (spec.nframes * spec.osc)
    scale = 1.0 - spec.scale_decay * np.clip(phi_frac, 0, 1)
    counts = I_true * pred["polarization"] / zeta * scale
    if spec.sensor_thickness > 0:
        counts = counts * geom.absorption_factors(pred["S"])
    Xv = geom.pixel_to_lab(pred["x"], pred["y"])
    Xlen = np.linalg.norm(Xv, axis=-1)
    cosa = (Xv @ geom.ed3) / Xlen                     # obliquity: footprint stretched radially
    second = pred["lattice"] == 1
    sd_l = np.where(second & (spec.extra_lattice_sigma_d > 0), spec.extra_lattice_sigma_d, spec.sigma_d)
    sm_l = np.where(second & (spec.extra_lattice_sigma_m > 0), spec.extra_lattice_sigma_m, spec.sigma_m)
    sig_px = Xlen * np.radians(sd_l) / spec.qx
    sig_rad = sig_px / cosa
    rdir = np.c_[pred["x"] - spec.orgx, pred["y"] - spec.orgy]
    rn = np.linalg.norm(rdir, axis=1, keepdims=True)
    rdir = np.where(rn > 1e-6, rdir / np.maximum(rn, 1e-6), np.array([[1.0, 0.0]]))
    sig_z = sm_l / zeta_u / spec.osc                  # in frames
    # static background and mask
    ys, xs = np.mgrid[0:spec.ny, 0:spec.nx]
    dmap = geom.resolution_of_pixel(xs + PIX0, ys + PIX0)
    bg = spec.background * (1.0 + 0.5 * np.exp(-((1 / dmap) / 0.25) ** 2)) \
        + spec.ring_amp * np.exp(-0.5 * ((1 / dmap - 1 / spec.ring_d) / 0.02) ** 2)
    zeta_px, pol_px = geom.lp_factor(geom.pixel_to_s(xs + PIX0, ys + PIX0))
    if spec.ice_powder > 0:      # powder rings of hexagonal ice (relative intensities approximate)
        from .ice import ICE_RINGS
        w_ice = (1.0, 0.6, 0.8, 0.3, 0.4, 0.25, 0.05, 0.25, 0.15, 0.05)
        bg = bg + spec.ice_powder * sum(w * np.exp(-0.5 * ((1 / dmap - 1 / d) / 0.0015) ** 2)
                                        for d, w in zip(ICE_RINGS, w_ice))
    if spec.ice_spots > 0:       # pixels on each ice ring, for the crystallite spots
        from .ice import ICE_RINGS
        ice_rng = np.random.default_rng(spec.seed + 4242)
        ice_pix = [np.flatnonzero(np.abs(1 / dmap - 1 / d).ravel() < 0.001) for d in ICE_RINGS]
        ice_pix = [a for a in ice_pix if a.size]
    bg = bg * pol_px
    mask = np.ones((spec.ny, spec.nx), bool)
    if geom.segmented:
        mask &= geom.segmap() >= 0
    elif spec.gaps:
        for y0 in range(195, spec.ny, 212):
            mask[y0:y0 + 17, :] = False
        for x0 in range(487, spec.nx, 494):
            mask[:, x0:x0 + 7] = False
    shadow = np.ones_like(bg)
    for (x0, x1, y0, y1) in spec.extra_dark_rect:
        shadow[y0:y1, x0:x1] = 0.02
    bg *= shadow
    # spots rendered within +-4 sigma (at least +-6 px; large spots: long distance, small pixels)
    half = int(np.clip(np.ceil(4.0 * np.quantile(sig_rad, 0.99)), 6, 40)) if sig_rad.size else 6
    offs = np.arange(-half, half + 1)
    template = str(outdir / "sim_?????.cbf")
    eiger = _open_eiger(outdir, spec, mask) if (write_images and fmt == "eiger") else None
    kt = spec.spot_tail_scale if spec.spot_tail > 0 else 1.0
    half_t = int(np.clip(np.ceil(3.5 * kt * np.quantile(sig_rad, 0.95)), half, max(30, half))) if spec.spot_tail > 0 else half
    offs_t = np.arange(-half_t, half_t + 1)
    for f in range(1, spec.nframes + 1):
        z0, z1 = f - 1.0, float(f)
        zc = pred["z"]
        if spec.spot_tail > 0:          # broad component: wider rocking curve and footprint, no sub-pixel sampling
            actt = (zc + 5 * kt * sig_z > z0) & (zc - 5 * kt * sig_z < z1)
            if actt.any():
                szt = kt * sig_z[actt]
                fzt = 0.5 * (erf((z1 - zc[actt]) / (np.sqrt(2) * szt)) - erf((z0 - zc[actt]) / (np.sqrt(2) * szt)))
                x, y, rd = pred["x"][actt], pred["y"][actt], rdir[actt]
                st, sr = kt * sig_px[actt], kt * sig_rad[actt]
                ix0 = np.floor(x - PIX0 + 0.5).astype(int)
                iy0 = np.floor(y - PIX0 + 0.5).astype(int)
                px = ix0[:, None] + offs_t[None, :]
                py = iy0[:, None] + offs_t[None, :]
                dxs = (px + PIX0 - x[:, None])[:, None, :]
                dys = (py + PIX0 - y[:, None])[:, :, None]
                ur = dxs * rd[:, 0, None, None] + dys * rd[:, 1, None, None]
                ut = -dxs * rd[:, 1, None, None] + dys * rd[:, 0, None, None]
                gt_ = np.exp(-0.5 * ((ur / sr[:, None, None]) ** 2 + (ut / st[:, None, None]) ** 2)) / (2 * np.pi * sr * st)[:, None, None]
                val = (spec.spot_tail * counts[actt] * fzt)[:, None, None] * gt_ * \
                    shadow[np.clip(iy0, 0, spec.ny - 1), np.clip(ix0, 0, spec.nx - 1)][:, None, None]
                PY = np.broadcast_to(py[:, :, None], val.shape)
                PX = np.broadcast_to(px[:, None, :], val.shape)
                ok = (PX >= 0) & (PX < spec.nx) & (PY >= 0) & (PY < spec.ny)
                tail_img = np.zeros_like(bg)
                np.add.at(tail_img, (PY[ok], PX[ok]), val[ok])
            else:
                tail_img = 0.0
        act = (zc + 5 * sig_z > z0) & (zc - 5 * sig_z < z1)
        img = bg.copy()
        if spec.spot_tail > 0:
            img = img + tail_img
        if act.any():
            c = counts[act] * (1.0 - spec.spot_tail)
            if spec.rocking_split > 0:
                sz = 0.5 * sig_z[act]
                fz = 0.0
                for sgn, wgt in ((-1, 0.65), (1, 0.35)):     # asymmetric split crystal
                    zcomp = zc[act] + sgn * spec.rocking_split * sig_z[act]
                    fz = fz + wgt * 0.5 * (erf((z1 - zcomp) / (np.sqrt(2) * sz)) - erf((z0 - zcomp) / (np.sqrt(2) * sz)))
            else:
                fz = 0.5 * (erf((z1 - zc[act]) / (np.sqrt(2) * sig_z[act])) - erf((z0 - zc[act]) / (np.sqrt(2) * sig_z[act])))
            x, y, st, sr, rd = pred["x"][act], pred["y"][act], sig_px[act], sig_rad[act], rdir[act]
            ix0 = np.floor(x - PIX0 + 0.5).astype(int)   # array index of the pixel containing x
            iy0 = np.floor(y - PIX0 + 0.5).astype(int)
            px = ix0[:, None] + offs[None, :]
            py = iy0[:, None] + offs[None, :]
            # elliptical Gaussian (radial sigma sr, tangential st), 4x4 sub-pixel sampling
            sub = (np.arange(4) + 0.5) / 4 - 0.5
            g2 = 0.0
            for sy in sub:
                for sx_ in sub:
                    dxs = (px + PIX0 + sx_ - x[:, None])[:, None, :]
                    dys = (py + PIX0 + sy - y[:, None])[:, :, None]
                    ur = dxs * rd[:, 0, None, None] + dys * rd[:, 1, None, None]
                    ut = -dxs * rd[:, 1, None, None] + dys * rd[:, 0, None, None]
                    g2 = g2 + np.exp(-0.5 * ((ur / sr[:, None, None]) ** 2 + (ut / st[:, None, None]) ** 2))
            g2 = g2 / 16.0 / (2 * np.pi * sr * st)[:, None, None]
            val = (c * fz)[:, None, None] * g2 * shadow[np.clip(iy0, 0, spec.ny - 1), np.clip(ix0, 0, spec.nx - 1)][:, None, None]
            PY = np.broadcast_to(py[:, :, None], val.shape)
            PX = np.broadcast_to(px[:, None, :], val.shape)
            ok = (PX >= 0) & (PX < spec.nx) & (PY >= 0) & (PY < spec.ny)
            np.add.at(img, (PY[ok], PX[ok]), val[ok])
        if spec.ice_spots > 0 and ice_pix:    # crystallites: 2D Gaussian spots (sigma 1 px) on one frame each
            ring = ice_rng.integers(0, len(ice_pix), spec.ice_spots)
            at = np.array([ice_pix[r][ice_rng.integers(0, ice_pix[r].size)] for r in ring])
            cy, cx = at // spec.nx, at % spec.nx
            amp = ice_rng.exponential(spec.ice_spot_counts, at.size)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    yy, xx = cy + dy, cx + dx
                    ok = (yy >= 0) & (yy < spec.ny) & (xx >= 0) & (xx < spec.nx)
                    np.add.at(img, (yy[ok], xx[ok]), amp[ok] * np.exp(-0.5 * (dx * dx + dy * dy)) / (2 * np.pi))
        frame = rng.poisson(img).astype(np.int32)
        frame[~mask] = -1
        if eiger is not None:
            fr = frame.astype(np.int64)
            fr[fr < 0] = 2 ** 32 - 1
            eiger[1][f - 1] = fr.astype(np.uint32)
        elif write_images:
            write_cbf(template.replace("?????", f"{f:05d}"), frame,
                      dict(qx=spec.qx, qy=spec.qx, wavelength=spec.wavelength, distance=spec.distance,
                           orgx=spec.orgx, orgy=spec.orgy, start_angle=spec.phi0 + (f - 1) * spec.osc,
                           osc=spec.osc))
        if verbose and f % 20 == 0:
            print(f"  simulated frame {f}/{spec.nframes}")
    if eiger is not None:
        eiger[0].close()
    p = Params(name_template=template, data_range=(1, spec.nframes), nx=spec.nx, ny=spec.ny, qx=spec.qx,
               qy=spec.qx, orgx=spec.orgx, orgy=spec.orgy, distance=spec.distance, wavelength=spec.wavelength,
               osc=spec.osc, starting_angle=spec.phi0, starting_frame=1, overload=1048500, min_valid_pixel=0,
               polarization_fraction=spec.polarization_fraction, sensor_thickness=spec.sensor_thickness,
               sensor_material=spec.sensor_material,
               beam_direction=np.array(spec.beam_direction, float), rotation_axis=np.array(spec.rotation_axis, float),
               trusted_region=(0.0, 1.5), detector="PILATUS")
    p.spot_range = [(1, spec.nframes)]
    p.background_range = (1, min(spec.nframes, 10))
    p.name_template = "sim_master.h5" if fmt == "eiger" else "sim_?????.cbf"
    if fmt == "eiger":
        p.detector, p.overload = "EIGER", 2 ** 20
    from .inputs import params_to_toml
    (outdir / "xpg.toml").write_text(params_to_toml(p))
    (outdir / "XDS.INP").write_text(format_xds_inp(p))     # legacy input, for the import / conversion tests
    np.savez_compressed(outdir / "truth.npz", hkl=pred["hkl"], hasu=hasu, I_true=I_true, counts=counts, lattice=pred["lattice"],
                        x=pred["x"], y=pred["y"], z=pred["z"], Astar=Astar, U=U)
    (outdir / "truth.json").write_text(json.dumps({
        "cell": spec.cell, "space_group": spec.space_group, "orgx": spec.orgx, "orgy": spec.orgy,
        "distance": spec.distance, "wavelength": spec.wavelength, "sigma_d": spec.sigma_d,
        "sigma_m": spec.sigma_m, "Astar": Astar.tolist()}, indent=1))
    return p, Astar

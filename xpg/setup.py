"""Build the processing parameters from image headers (CBF, SMV or Eiger/NeXus HDF5).

From the headers: detector type, geometry, wavelength,
oscillation, data range and standard untrusted regions (module gaps) are filled in.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np

from .config import Params
from .io.images import CBFImageSet, SMVImageSet, is_smv, open_images


def _template_from_file(path: str) -> str:
    base = os.path.basename(path)
    m = list(re.finditer(r"\d+", base))
    if not m:
        return path
    last = m[-1]
    return os.path.join(os.path.dirname(path), base[:last.start()] + "?" * len(last.group()) + base[last.end():])


def params_from_images(images: str, workdir=".") -> Params:
    images = os.path.abspath(images)
    if os.path.isdir(images):
        cands = sorted(glob.glob(os.path.join(images, "*_master.h5"))) or sorted(glob.glob(os.path.join(images, "*.cbf*"))) \
            or sorted(glob.glob(os.path.join(images, "*.img")))
        if not cands:
            raise FileNotFoundError(f"no images found in {images}")
        images = cands[0]
    if images.endswith(".h5") or images.endswith(".nxs"):
        iset = open_images(images)
        h = iset.header
        p = Params(name_template=images, detector="EIGER")
        p.nx, p.ny = iset.nx, iset.ny
        p.qx = h.get("qx") or 0.075
        p.qy = h.get("qy") or 0.075
        p.orgx = float(h.get("orgx") or iset.nx / 2)
        p.orgy = float(h.get("orgy") or iset.ny / 2)
        p.distance = float(h.get("distance") or 200.0)
        p.wavelength = float(h.get("wavelength") or 1.0)
        p.sensor_thickness = float(h.get("sensor_thickness") or 0.45)
        p.sensor_material = "CdTe" if "cdte" in str(h.get("sensor_material", "")).lower() else "Si"
        osc = None
        start = 0.0
        for ax in ("omega", "phi", "chi", "kappa"):
            inc = h.get(f"{ax}_increment")
            if inc and abs(inc) > 1e-6:
                osc = inc
                start = float(np.atleast_1d(h[ax])[0])
                break
        p.osc = float(osc or 0.1)
        p.starting_angle = start
        geo_note = _nexus_geometry(iset._h5, p, h)
        p.data_range = (iset.first, iset.last)
        p.overload = float(h.get("count_cutoff") or 2 ** 31 - 2)
        if h.get("pixel_mask") is None and (iset.nx + 10) % 1040 == 0:
            p.untrusted_rectangles = eiger_gaps(iset.nx, iset.ny)
    elif is_smv(images if "?" not in images else sorted(glob.glob(images.replace("?", "[0-9]")))[0]):
        tmpl = images if "?" in images else _template_from_file(images)
        p, geo_note = _smv_params(SMVImageSet(tmpl), tmpl)
    else:
        tmpl = images if "?" in images else _template_from_file(images)
        iset = CBFImageSet(tmpl)
        h = iset.header
        p = Params(name_template=tmpl, detector="PILATUS")
        p.nx, p.ny = iset.nx, iset.ny
        ps = h.get("pixel_size") or (172e-6, 172e-6)
        p.qx, p.qy = ps[0] * 1000, ps[1] * 1000
        bx, by = h.get("beam_xy") or (p.nx / 2, p.ny / 2)
        p.orgx, p.orgy = bx, by
        p.distance = (h.get("distance") or 0.2) * 1000
        p.wavelength = h.get("wavelength") or 1.0
        p.osc = h.get("angle_increment") or 0.1
        p.starting_angle = h.get("start_angle") or 0.0
        p.sensor_thickness = (h.get("sensor_thickness") or 0.00045) * 1000
        p.data_range = (iset.first, iset.last)
        p.overload = h.get("count_cutoff") or 1048500
        if "EIGER" in str(h.get("detector", "")).upper():
            p.detector = "EIGER"
            p.untrusted_rectangles = eiger_gaps(p.nx, p.ny)
        geo_note = _cbf_geometry(h, p)
    p.starting_frame = p.data_range[0]
    p.spot_range = [p.data_range]
    nb = max(1, int(round(5.0 / max(p.osc, 1e-6))))
    p.background_range = (p.data_range[0], min(p.data_range[1], p.data_range[0] + nb - 1))
    p.trusted_region = (0.0, 1.41)
    p.polarization_fraction = 0.99
    p.workdir = Path(workdir)
    p.beam_direction = np.array([0.0, 0.0, 1.0])
    p.min_valid_pixel = 1 if p.detector == "ADSC" else 0     # ADSC: 0 marks a dead pixel
    p.extra["XPG_GEOMETRY_NOTE"] = geo_note
    return p


def _smv_params(iset, tmpl):
    """SMV (ADSC-style) header: CCD, no parallax. ADSC's BEAM_CENTER_X / _Y are mm along the slow / fast pixel
    direction (ORGX = BEAM_CENTER_Y / QX, ORGY = BEAM_CENTER_X / QY; DIALS reads them the same way); a
    detector block's <NAME>SPATIAL_BEAM_POSITION (pixels, fast slow) takes precedence. A header IMAGE_PEDESTAL is
    subtracted as OFFSET (the background variance model assumes no constant offset); none is assumed without the
    key."""
    h = iset.header
    p = Params(name_template=tmpl, detector="ADSC")
    p.nx, p.ny = iset.nx, iset.ny
    p.qx = p.qy = float(h.get("PIXEL_SIZE") or 0.1)
    det = (h.get("DETECTOR_NAMES") or "").split()
    sbp = h.get(f"{det[0]}SPATIAL_BEAM_POSITION") if det else None
    notes = []
    if sbp:
        p.orgx, p.orgy = (float(v) for v in sbp.split()[:2])
        notes.append(f"beam position from {det[0]}SPATIAL_BEAM_POSITION")
    elif h.get("BEAM_CENTER_X") and h.get("BEAM_CENTER_Y"):
        p.orgx, p.orgy = float(h["BEAM_CENTER_Y"]) / p.qx, float(h["BEAM_CENTER_X"]) / p.qy
    else:
        p.orgx, p.orgy = p.nx / 2, p.ny / 2
    p.distance = float(h.get("DISTANCE") or 200.0)
    p.wavelength = float(h.get("WAVELENGTH") or 1.0)
    p.osc = float(h.get("OSC_RANGE") or 0.1)
    p.starting_angle = float(h.get("OSC_START", h.get("PHI")) or 0.0)
    p.sensor_thickness = 0.0                        # phosphor-coupled CCD: no parallax / sensor absorption model
    p.data_range = (iset.first, iset.last)
    sat = float(h.get("count_cutoff") or 65535)
    p.overload = min(sat, 65000.0) if sat >= 65535 else sat      # the usual ADSC value (non-linear near saturation)
    if "IMAGE_PEDESTAL" in h:
        p.offset = float(h["IMAGE_PEDESTAL"])
    p.rotation_axis = np.array([1.0, 0.0, 0.0])
    tth = float(h.get("TWOTHETA") or 0.0)
    if abs(tth) > 1e-3:
        from .goniometer import rot
        R = rot([1.0, 0.0, 0.0], tth)
        p.ed1, p.ed2, p.two_theta = R @ p.ed1, R @ p.ed2, tth
        notes.append(f"detector two-theta {tth:.2f} deg (swing about the horizontal axis assumed)")
    return p, "; ".join(notes)


# NeXus vectors are in the McStas frame (x left, y up, z along the beam). The frame used here
# (detector x = fast = +x, y = slow = +y, beam +z) is McStas rotated by 180 deg about z.
_MCSTAS_TO_LAB = np.diag([-1.0, -1.0, 1.0])


def _nexus_geometry(h5, p, h):
    """Rotation axis (multi-axis goniometers), detector axes and two-theta from a NeXus master file."""
    from .goniometer import read_nexus_detector_axes, read_nexus_goniometer, rot
    notes = []
    p.rotation_axis = np.array([1.0, 0.0, 0.0])          # default: horizontal axis (most beamlines)
    try:
        chain = read_nexus_goniometer(h5)
    except Exception:
        chain = None
    if chain is not None and chain.scan_index() is not None:
        i = chain.scan_index()
        p.rotation_axis = _MCSTAS_TO_LAB @ chain.effective_axis()
        p.osc = chain.axes[i].increment
        p.starting_angle = chain.axes[i].start
        notes.append(f"goniometer {chain.describe()}; effective rotation axis {np.round(p.rotation_axis, 5).tolist()}")
    try:
        dax = read_nexus_detector_axes(h5)
    except Exception:
        dax = None
    if dax is not None:
        fast, slow, _, tth = dax
        p.ed1 = _MCSTAS_TO_LAB @ (fast / np.linalg.norm(fast))
        p.ed2 = _MCSTAS_TO_LAB @ (slow / np.linalg.norm(slow))
        p.two_theta = tth
        if abs(tth) > 1e-3:
            notes.append(f"detector two-theta {tth:.2f} deg (from transformation chain)")
    elif h.get("two_theta") and abs(h["two_theta"]) > 1e-3:
        # no module vectors: swing about the horizontal axis through the crystal
        R = rot([1.0, 0.0, 0.0], h["two_theta"])
        p.ed1, p.ed2, p.two_theta = R @ p.ed1, R @ p.ed2, h["two_theta"]
        notes.append(f"detector two-theta {h['two_theta']:.2f} deg about the horizontal axis (assumed)")
    return "; ".join(notes)


def _cbf_geometry(h, p):
    from .goniometer import cbf_kappa_chain, rot
    notes = []
    p.rotation_axis = np.array([1.0, 0.0, 0.0])
    kappa, chi = float(h.get("kappa") or 0.0), float(h.get("chi") or 0.0)
    if abs(kappa) > 1e-3:
        osc_axis = str(h.get("oscillation_axis", "OMEGA")).upper()
        h2 = dict(h)
        scan = "phi" if "PHI" in osc_axis else "omega"
        h2[scan] = h.get("start_angle", h2.get(scan))
        chain = cbf_kappa_chain(h2, alpha=float(h.get("alpha") or 50.0) or 50.0)
        p.rotation_axis = chain.effective_axis()
        notes.append(f"mini-kappa {chain.describe()} (alpha 50 deg assumed); effective rotation axis "
                     f"{np.round(p.rotation_axis, 5).tolist()} - verify for this beamline")
    elif abs(chi) > 1e-3 and "PHI" in str(h.get("oscillation_axis", "")).upper():
        # phi on a chi circle (rotation about the beam-perpendicular vertical-plane axis)
        p.rotation_axis = rot([0.0, 0.0, 1.0], chi) @ np.array([1.0, 0.0, 0.0])
        notes.append(f"phi scan at chi {chi:.2f}; effective rotation axis {np.round(p.rotation_axis, 5).tolist()}")
    tth = float(h.get("two_theta") or 0.0)
    if abs(tth) > 1e-3:
        R = rot(p.rotation_axis if abs(kappa) < 1e-3 else [1.0, 0.0, 0.0], tth)
        p.ed1, p.ed2, p.two_theta = R @ p.ed1, R @ p.ed2, tth
        notes.append(f"detector two-theta {tth:.2f} deg (swing about the horizontal axis assumed)")
    return "; ".join(notes)


def eiger_gaps(nx, ny):
    """Inter-module gaps for standard Eiger/Eiger2 module layouts (1030x514 modules)."""
    rects = []
    mx, my, gx, gy = 1030, 514, 10, 37
    x = mx
    while x < nx:
        rects.append((x, x + gx + 1, 0, ny + 1))
        x += mx + gx
    y = my
    while y < ny:
        rects.append((0, nx + 1, y, y + gy + 1))
        y += my + gy
    return rects

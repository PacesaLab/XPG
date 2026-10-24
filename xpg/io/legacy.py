"""Legacy-format text files: XPARM.XDS / GXPARM.XDS, INTEGRATE.HKL, XDS_ASCII.HKL (read; written with --export-legacy),
so that other programs can read XPG output directly.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from ..detector import Segment
from ..geometry import Geometry, real_basis
from ..models import Crystal, Experiment


def write_xparm(path, exp: Experiment, space_group_number=1, title="XPARM.XDS"):
    g, c = exp.geom, exp.crystal
    R = real_basis(c.Astar)
    cell = c.cell
    s0 = g.s0
    L = [
        f" {title:12s} VERSION XPG {__version__}",
        f"{g.starting_frame:6d}{g.phi0:14.4f}{g.osc:10.4f}{g.axis[0]:10.6f}{g.axis[1]:10.6f}{g.axis[2]:10.6f}",
        f"{g.wavelength:15.6f}{s0[0]:15.6f}{s0[1]:15.6f}{s0[2]:15.6f}",
        f"{space_group_number:6d}{cell[0]:12.4f}{cell[1]:12.4f}{cell[2]:12.4f}{cell[3]:8.3f}{cell[4]:8.3f}{cell[5]:8.3f}",
        *[f"{R[i, 0]:15.6f}{R[i, 1]:15.6f}{R[i, 2]:15.6f}" for i in range(3)],
        f"{len(g.segment_list()):10d}{g.nx:10d}{g.ny:10d}{g.qx:12.6f}{g.qy:12.6f}",
        f"{g.orgx:15.6f}{g.orgy:15.6f}{g.distance:15.6f}",
        *[f"{v[0]:15.6f}{v[1]:15.6f}{v[2]:15.6f}" for v in (g.ed1, g.ed2, g.ed3)],
    ]
    for k, sg in enumerate(g.segment_list()):
        _, ex, ey = sg.local(g.qx, g.qy)
        L.append(f"{k + 1:10d}{sg.x1:10d}{sg.x2:10d}{sg.y1:10d}{sg.y2:10d}")
        L.append(f"{sg.orgx:8.2f}{sg.orgy:8.2f}{sg.distance:8.2f}" + "".join(f"{v:9.5f}" for v in (*ex, *ey)))
    Path(path).write_text("\n".join(L) + "\n")


def read_xparm(path, base: Geometry | None = None):
    """Read a (G)XPARM.XDS file (new format). Returns (Experiment, space_group_number)."""
    t = Path(path).read_text().split("\n")
    nums = [list(map(float, l.split())) for l in t[1:] if l.strip()]
    f0, phi0, osc, *axis = nums[0]
    wl, *s0 = nums[1]
    sg = int(nums[2][0])
    R = np.array(nums[3:6])
    nseg, nx, ny, qx, qy = nums[6]
    orgx, orgy, dist = nums[7]
    ed1, ed2 = np.array(nums[8]), np.array(nums[9])
    s0 = np.array(s0)
    segs = []
    for k in range(int(nseg)):
        a, b = nums[11 + 2 * k], nums[12 + 2 * k]
        segs.append(Segment(*(int(v) for v in a[1:5]), orgx=b[0], orgy=b[1], distance=b[2],
                            ex=np.array(b[3:6]), ey=np.array(b[6:9])))
    trivial = len(segs) == 1 and segs[0].rect == (1, int(nx), 1, int(ny)) and \
        abs(segs[0].orgx) + abs(segs[0].orgy) + abs(segs[0].distance) < 1e-6 and \
        np.allclose(segs[0].ex, [1, 0, 0]) and np.allclose(segs[0].ey, [0, 1, 0])
    g = Geometry(wavelength=wl, beam=s0 / np.linalg.norm(s0), axis=np.array(axis) / np.linalg.norm(axis),
                 orgx=orgx, orgy=orgy, distance=dist, ed1=ed1, ed2=ed2, qx=qx, qy=qy, nx=int(nx), ny=int(ny),
                 phi0=phi0, osc=osc, starting_frame=int(f0), segments=None if trivial or not segs else segs)
    if base is not None:
        g.polarization_fraction = base.polarization_fraction
        g.polarization_normal = base.polarization_normal
        g.sensor_thickness, g.mu = base.sensor_thickness, base.mu
        g.geo_corr = base.geo_corr
    Astar = np.linalg.inv(R)
    return Experiment(g, Crystal.from_Astar(Astar, "a")), sg


def _header_common(g: Geometry, cell, R, sgnum, data_range, sigma_m, sigma_d, template, p=None):
    s0 = g.s0 * g.wavelength
    L = [
        f"!NAME_TEMPLATE_OF_DATA_FRAMES={template}",
        f"!DATA_RANGE={data_range[0]:8d}{data_range[1]:8d}",
        f"!ROTATION_AXIS={g.axis[0]:10.6f}{g.axis[1]:10.6f}{g.axis[2]:10.6f}",
        f"!OSCILLATION_RANGE={g.osc:10.6f}",
        f"!STARTING_ANGLE={g.phi0:10.3f}",
        f"!STARTING_FRAME={g.starting_frame:8d}",
    ]
    return L, [
        f"!SPACE_GROUP_NUMBER={sgnum:5d}",
        f"!UNIT_CELL_CONSTANTS={cell[0]:10.3f}{cell[1]:10.3f}{cell[2]:10.3f}{cell[3]:8.3f}{cell[4]:8.3f}{cell[5]:8.3f}",
        f"!UNIT_CELL_A-AXIS={R[0, 0]:10.3f}{R[0, 1]:10.3f}{R[0, 2]:10.3f}",
        f"!UNIT_CELL_B-AXIS={R[1, 0]:10.3f}{R[1, 1]:10.3f}{R[1, 2]:10.3f}",
        f"!UNIT_CELL_C-AXIS={R[2, 0]:10.3f}{R[2, 1]:10.3f}{R[2, 2]:10.3f}",
        f"!REFLECTING_RANGE_E.S.D.={sigma_m:10.3f}",
        f"!BEAM_DIVERGENCE_E.S.D.={sigma_d:10.3f}",
        f"!X-RAY_WAVELENGTH={g.wavelength:10.6f}",
        f"!INCIDENT_BEAM_DIRECTION={s0[0]:10.6f}{s0[1]:10.6f}{s0[2]:10.6f}",
        f"!FRACTION_OF_POLARIZATION={g.polarization_fraction:8.3f}",
        f"!POLARIZATION_PLANE_NORMAL={g.polarization_normal[0]:10.6f}{g.polarization_normal[1]:10.6f}{g.polarization_normal[2]:10.6f}",
        "!DETECTOR=PILATUS",
        f"!NX={g.nx:6d}  NY={g.ny:6d}    QX={g.qx:10.6f}  QY={g.qy:10.6f}",
        f"!ORGX={g.orgx:10.2f}  ORGY={g.orgy:10.2f}",
        f"!DETECTOR_DISTANCE={g.distance:10.3f}",
        f"!DIRECTION_OF_DETECTOR_X-AXIS={g.ed1[0]:10.5f}{g.ed1[1]:10.5f}{g.ed1[2]:10.5f}",
        f"!DIRECTION_OF_DETECTOR_Y-AXIS={g.ed2[0]:10.5f}{g.ed2[1]:10.5f}{g.ed2[2]:10.5f}",
    ]


def _frame_safe(z):
    """Frame coordinate for one-decimal output that stays in the reflection's own frame (z in [n-1, n) is frame
    n; printed as n.0 it would be read as frame n+1, e.g. a near-empty batch past the end of DATA_RANGE)."""
    z = np.asarray(z, float)
    return np.minimum(z, np.floor(z) + 0.94)


def write_legacy_hkl(path, exp: Experiment, sgnum, hkl, I, sig, xd, yd, zd, rlp, peak, corr, psi=None,
                    friedel=True, data_range=(1, 1), sigma_m=0.1, sigma_d=0.03, template="", dmin=None,
                    variance_model=(1.0, 0.0)):
    g, c = exp.geom, exp.crystal
    R = real_basis(c.Astar)
    today = datetime.date.today().strftime("%d-%b-%Y")
    head = [
        f"!FORMAT=XDS_ASCII    MERGE=FALSE    FRIEDEL'S_LAW={'TRUE' if friedel else 'FALSE'}",
        f"!OUTPUT_FILE=XDS_ASCII.HKL        DATE={today}",
        f"!Generated by SCALE   (VERSION XPG {__version__})",
        "!PROFILE_FITTING= TRUE ",
    ]
    a, b = _header_common(g, c.cell, R, sgnum, data_range, sigma_m, sigma_d, template)
    dmin = dmin if dmin is not None else 0.0
    head += a + [f"!INCLUDE_RESOLUTION_RANGE={50.0:10.3f}{dmin:10.3f}"] + b
    head += [f"!VARIANCE_MODEL={variance_model[0]:11.3E}{variance_model[1]:11.3E}",
             "!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=12",
             "!ITEM_H=1", "!ITEM_K=2", "!ITEM_L=3", "!ITEM_IOBS=4", "!ITEM_SIGMA(IOBS)=5",
             "!ITEM_XD=6", "!ITEM_YD=7", "!ITEM_ZD=8", "!ITEM_RLP=9", "!ITEM_PEAK=10", "!ITEM_CORR=11",
             "!ITEM_PSI=12", "!END_OF_HEADER"]
    if psi is None:
        psi = np.zeros(len(I))
    zd = _frame_safe(zd)
    with open(path, "w") as f:
        f.write("\n".join(head) + "\n")
        for i in range(len(I)):
            f.write(f"{hkl[i, 0]:6d}{hkl[i, 1]:6d}{hkl[i, 2]:6d}{I[i]:11.3E}{sig[i]:11.3E}"
                    f"{xd[i]:8.1f}{yd[i]:8.1f}{zd[i]:9.1f}{rlp[i]:10.5f}{int(peak[i]):4d}{int(corr[i]):4d}{psi[i]:8.2f}\n")
        f.write("!END_OF_DATA\n")


def write_integrate_hkl(path, exp: Experiment, refl, data_range, sigma_m, sigma_d, template=""):
    """INTEGRATE.HKL-like file (LP-corrected profile-fitted intensities, P1 indices)."""
    g, c = exp.geom, exp.crystal
    R = real_basis(c.Astar)
    head = ["!OUTPUT_FILE=INTEGRATE.HKL    DATE=" + datetime.date.today().strftime("%d-%b-%Y"),
            f"!Generated by XPG {__version__}"]
    # space group of the indexed lattice (centred cells must not be declared P1: readers would take the conventional
    # cell as primitive and find a doubled cell)
    try:
        import gemmi
        sgnum = gemmi.SpaceGroup(c.space_group).number if c.space_group else 1
    except Exception:
        sgnum = 1
    a, b = _header_common(g, c.cell, R, sgnum, data_range, sigma_m, sigma_d, template)
    head += a + b
    head += ["!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=21",
             "!H,K,L,IOBS,SIGMA,XCAL,YCAL,ZCAL,RLP,PEAK,CORR,MAXC,", "!             XOBS,YOBS,ZOBS,ALF0,BET0,ALF1,BET1,PSI,ISEG",
             "!Items are separated by a blank and can be read in free-format",
             "!END_OF_HEADER"]
    n = len(refl["I"])
    # beam directions as the format defines them: spherical angles (deg) in the laboratory frame, ALF = atan2(y, x),
    # BET = angle from the z axis; readers derive each reflection's scattering angle (resolution) from them
    u0 = np.asarray(g.s0, float) / np.linalg.norm(g.s0)
    alf0, bet0 = np.degrees(np.arctan2(u0[1], u0[0])), np.degrees(np.arccos(np.clip(u0[2], -1, 1)))
    if "S" in refl:
        u1 = np.asarray(refl["S"], float)
        u1 = u1 / np.linalg.norm(u1, axis=1, keepdims=True)
        alf1, bet1 = np.degrees(np.arctan2(u1[:, 1], u1[:, 0])), np.degrees(np.arccos(np.clip(u1[:, 2], -1, 1)))
    else:
        alf1 = bet1 = np.zeros(n)
    xo = np.where(np.isfinite(refl["xo"]), refl["xo"], refl["x"])
    yo = np.where(np.isfinite(refl["yo"]), refl["yo"], refl["y"])
    zo = _frame_safe(np.where(np.isfinite(refl["zo"]), refl["zo"], refl["z"]))
    zc = _frame_safe(refl["z"])
    h = np.asarray(refl["hkl"], int)
    with open(path, "w") as f:
        f.write("\n".join(head) + "\n")
        for i in range(n):
            f.write(f" {h[i, 0]} {h[i, 1]} {h[i, 2]} {refl['I'][i]:.3E} {refl['sig'][i]:.3E} {refl['x'][i]:.1f} "
                    f"{refl['y'][i]:.1f} {zc[i]:.1f} {refl['rlp'][i]:.5f} {float(refl['peak'][i]):.5f} "
                    f"{int(np.nan_to_num(refl['corr'][i])):d} 0 {xo[i]:.1f} {yo[i]:.1f} {zo[i]:.1f} {alf0:.2f} {bet0:.2f} "
                    f"{alf1[i]:.2f} {bet1[i]:.2f} 0.00 1\n")
        f.write("!END_OF_DATA\n")


def read_legacy_hkl(path):
    """Read XDS_ASCII.HKL / INTEGRATE.HKL into (header dict, data array)."""
    hdr = {}
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("!"):
                if "=" in line:
                    for part in line[1:].split("  "):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            hdr[k.strip()] = v.strip()
                continue
            if line.strip():
                rows.append(line.split())
    return hdr, np.array(rows, float) if rows else np.zeros((0, 12))

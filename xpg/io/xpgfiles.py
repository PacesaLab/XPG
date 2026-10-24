"""XPG's own file formats.

* ``xpg_geometry*.json``: experimental geometry and crystal model (beam, rotation axis, detector, scan, unit cell
  and orientation, space group). Written next to every unmerged MTZ so that it is self-contained for re-scaling /
  merging (``xpg merge``), and as the indexing / refined model of a run.
* ``xpg_spots.txt``: strong spots (x, y, frame, intensity[, h, k, l]).
* unmerged MTZ reading for the merge step (original indices from M/ISYM, frame from ROT and the geometry).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "xpg-geometry/1"


def geometry_dict(exp, space_group_number=1, data_range=None, friedel=True, merged=False):
    g, c = exp.geom, exp.crystal
    d = dict(format=FORMAT,
             wavelength=float(g.wavelength), beam=[float(v) for v in g.beam], rotation_axis=[float(v) for v in g.axis],
             detector=dict(orgx=float(g.orgx), orgy=float(g.orgy), distance=float(g.distance),
                           x_axis=[float(v) for v in g.ed1], y_axis=[float(v) for v in g.ed2],
                           pixel_x=float(g.qx), pixel_y=float(g.qy), nx=int(g.nx), ny=int(g.ny),
                           sensor_thickness=float(g.sensor_thickness), mu=float(g.mu)),
             polarization=dict(fraction=float(g.polarization_fraction), normal=[float(v) for v in g.polarization_normal]),
             scan=dict(phi_start=float(g.phi0), osc=float(g.osc), starting_frame=int(g.starting_frame),
                       frames=[int(v) for v in data_range] if data_range is not None else None),
             crystal=dict(space_group_number=int(space_group_number), cell=[float(v) for v in c.cell],
                          a_star=np.asarray(c.Astar, float).tolist(), system=str(c.system)),
             friedel_law=bool(friedel), merged=bool(merged))
    return d


def write_geometry(path, exp, space_group_number=1, data_range=None, friedel=True, merged=False):
    Path(path).write_text(json.dumps(geometry_dict(exp, space_group_number, data_range, friedel, merged), indent=1) + "\n")


def read_geometry(path):
    """-> (Geometry, A* (3x3), dict). Detector segments / distortion tables are not stored (single-panel model)."""
    from ..geometry import Geometry, unit
    d = json.loads(Path(path).read_text())
    if int(str(d.get("format", FORMAT)).rsplit("/", 1)[-1]) > int(FORMAT.rsplit("/", 1)[-1]):
        raise ValueError(f"{path}: {d['format']} is newer than this XPG reads ({FORMAT})")
    det, sc = d["detector"], d["scan"]
    g = Geometry(wavelength=d["wavelength"], beam=unit(np.array(d["beam"])), axis=unit(np.array(d["rotation_axis"])),
                 orgx=det["orgx"], orgy=det["orgy"], distance=det["distance"], ed1=unit(np.array(det["x_axis"])),
                 ed2=unit(np.array(det["y_axis"])), qx=det["pixel_x"], qy=det["pixel_y"], nx=det["nx"], ny=det["ny"],
                 phi0=sc["phi_start"], osc=sc["osc"], starting_frame=sc["starting_frame"],
                 polarization_fraction=d.get("polarization", {}).get("fraction", 0.5),
                 polarization_normal=np.array(d.get("polarization", {}).get("normal", [0.0, 1.0, 0.0])),
                 sensor_thickness=det.get("sensor_thickness", 0.0), mu=det.get("mu", 0.0))
    return g, np.array(d["crystal"]["a_star"], float), d


def sidecar(mtz_path):
    """Geometry file belonging to an unmerged MTZ: <stem>_geometry.json, else xpg_geometry.json in the directory."""
    p = Path(mtz_path)
    for c in (p.with_name(p.stem.replace("_unmerged", "") + "_geometry.json"), p.with_name("xpg_geometry.json")):
        if c.exists():
            return c
    return None


def write_spots(path, x, y, z, intensity, hkl=None):
    """xpg_spots.txt: one strong spot per line; frame coordinate z as in the pipeline (0 = start of the first frame)."""
    cols = "x_px y_px frame intensity" + (" h k l" if hkl is not None else "")
    with open(path, "w") as f:
        f.write(f"# XPG strong spots: {cols}\n")
        for i in range(len(x)):
            line = f"{x[i]:10.2f} {y[i]:10.2f} {z[i]:10.2f} {intensity[i]:12.0f}"
            if hkl is not None:
                line += f" {int(hkl[i][0]):5d} {int(hkl[i][1]):5d} {int(hkl[i][2]):5d}"
            f.write(line + "\n")


def read_unmerged_mtz(path):
    """Unmerged MTZ (M/ISYM, BATCH, I, SIGI, XDET, YDET, ROT, LP) -> dict with original hkl, I, sig, x, y, z (frame
    coordinate from ROT and the geometry), rlp, plus the geometry from the sidecar JSON (None if absent)."""
    import gemmi
    m = gemmi.read_mtz_file(str(path))
    m.switch_to_original_hkl()
    a = np.array(m, copy=False)
    lab = [c.label for c in m.columns]
    col = lambda k, default=None: a[:, lab.index(k)].astype(float) if k in lab else default
    hkl = a[:, :3].astype(int)
    out = dict(hkl=hkl, I=col("I"), sig=col("SIGI"), x=col("XDET", np.zeros(len(a))), y=col("YDET", np.zeros(len(a))),
               rot=col("ROT"), rlp=col("LP", np.ones(len(a))), spacegroup=m.spacegroup, cell=np.array(m.cell.parameters))
    sc = sidecar(path)
    if sc is not None:
        g, Astar, d = read_geometry(sc)
        out.update(geom=g, Astar=Astar, meta=d)
        out["z"] = g.phi_to_z(out["rot"]) if out["rot"] is not None else np.zeros(len(a))
    else:
        out.update(geom=None, Astar=None, meta={})
        out["z"] = np.zeros(len(a))
    return out

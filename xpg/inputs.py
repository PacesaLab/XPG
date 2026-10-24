"""XPG input file (``xpg.toml``) and conversion of other programs' input files.

``xpg.toml`` sections and keys (all optional except ``data.images``; units: pixels, mm, degrees, Angstrom):

    [data]        images (template with ???? or an HDF5 master file), frames = [first, last], exclude_frames = [[a, b]],
                  spot_frames, background_frames, starting_frame, starting_angle, oscillation
    [beam]        wavelength, direction, polarization_fraction, polarization_normal
    [goniometer]  rotation_axis
    [detector]    type, nx, ny, pixel_x, pixel_y, beam_x, beam_y, distance, detector_x_axis, detector_y_axis,
                  sensor_material, sensor_thickness, overload, min_valid_pixel, trusted_region, trusted_values,
                  untrusted_rectangles, untrusted_ellipses, untrusted_quadrilaterals, gain, dark_current_image, offset,
                  sensor_mu, x_distortion, y_distortion, two_theta
    [crystal]     space_group (number), unit_cell, friedels_law, reindex, reference_data, free_r_reference, max_cell
    [processing]  stages, resolution = [low, high], exclude_resolution, refine_index, refine_integrate, refine_scale,
                  and any other processing parameter by its name (see config.Params)

The full reference is docs/FORMAT.md. An XDS.INP is read through the keyword parser (config.read_xds_inp) and can be
converted (``xpg convert``).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .config import Params

# toml key -> Params attribute, per section (keys not listed map to the attribute of the same name)
SECTIONS = {
    "data": {"images": "name_template", "frames": "data_range", "exclude_frames": "exclude_data_range",
             "spot_frames": "spot_range", "background_frames": "background_range", "starting_frame": "starting_frame",
             "starting_angle": "starting_angle", "oscillation": "osc"},
    "beam": {"wavelength": "wavelength", "direction": "beam_direction", "polarization_fraction": "polarization_fraction",
             "polarization_normal": "polarization_normal"},
    "goniometer": {"rotation_axis": "rotation_axis"},
    "detector": {"type": "detector", "nx": "nx", "ny": "ny", "pixel_x": "qx", "pixel_y": "qy", "beam_x": "orgx",
                 "beam_y": "orgy", "distance": "distance", "detector_x_axis": "ed1", "detector_y_axis": "ed2",
                 "sensor_material": "sensor_material", "sensor_thickness": "sensor_thickness", "overload": "overload",
                 "min_valid_pixel": "min_valid_pixel", "trusted_region": "trusted_region",
                 "trusted_values": "value_range_trusted", "untrusted_rectangles": "untrusted_rectangles",
                 "untrusted_ellipses": "untrusted_ellipses", "untrusted_quadrilaterals": "untrusted_quadrilaterals",
                 "gain": "gain", "dark_current_image": "dark_current_image", "offset": "offset",
                 "sensor_mu": "silicon_mu", "x_distortion": "x_geo_corr", "y_distortion": "y_geo_corr",
                 "two_theta": "two_theta"},
    "crystal": {"space_group": "space_group_number", "unit_cell": "unit_cell", "friedels_law": "friedels_law",
                "reindex": "reidx", "reference_data": "reference_data_set", "free_r_reference": "free_r_reference",
                "max_cell": "max_cell"},
    "processing": {"stages": "jobs", "resolution": "include_resolution", "exclude_resolution": "exclude_resolution",
                   "refine_index": "refine_idxref", "refine_integrate": "refine_integrate", "refine_scale": "refine_correct"},
}
FORMAT = 1           # layout version, written as "format = 1"; files without it are read as version 1
_ALWAYS = {"name_template", "data_range", "wavelength", "nx", "ny", "qx", "qy", "orgx", "orgy", "distance", "osc",
           "starting_angle", "rotation_axis", "beam_direction", "ed1", "ed2", "detector"}
_SKIP = {"workdir", "extra", "segments", "jobs", "lib"}


def _attr_to_key():
    out = {}
    for sec, m in SECTIONS.items():
        for k, a in m.items():
            out[a] = (sec, k)
    return out


def _coerce(attr, value, default):
    """TOML value -> the type of the Params field."""
    if value is None:
        return None
    if isinstance(default, np.ndarray) or attr in ("ed1", "ed2", "beam_direction", "rotation_axis", "polarization_normal"):
        return np.array(value, float)
    if attr == "reidx":
        return np.array(value, float).reshape(3, 4) if np.size(value) == 12 else np.array(value, float)
    if attr in ("data_range", "background_range", "include_resolution", "trusted_region", "value_range_trusted",
                "fixed_scale_range", "unit_cell"):
        return tuple(value)
    if attr in ("exclude_data_range", "spot_range", "exclude_resolution", "untrusted_rectangles", "untrusted_ellipses",
                "untrusted_quadrilaterals"):
        return [tuple(v) for v in value]
    if attr in ("jobs", "refine_idxref", "refine_integrate", "refine_correct", "corrections"):
        return [str(v).upper() for v in value]
    return value


def read_toml(path) -> Params:
    import tomllib
    path = Path(path)
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    p = Params(workdir=path.parent)
    fields = {f.name: f for f in dataclasses.fields(Params)}
    a2k = _attr_to_key()
    fmt = doc.pop("format", 1)
    if not isinstance(fmt, int) or fmt > FORMAT:
        raise ValueError(f"{path}: format {fmt!r}: this XPG reads format {FORMAT} and older (update XPG)")
    unknown = []
    for sec, tab in doc.items():
        if not isinstance(tab, dict) or sec not in SECTIONS:
            unknown.append(sec)
            continue
        m = SECTIONS[sec]
        for k, v in tab.items():
            # [processing] also takes every processing parameter of config.Params by its name
            attr = m[k] if k in m else (k if sec == "processing" and k not in _SKIP and k not in a2k else None)
            if attr not in fields:
                unknown.append(f"{sec}.{k}")
                continue
            setattr(p, attr, _coerce(attr, v, getattr(p, attr)))
    if unknown:
        raise ValueError(f"{path}: unknown settings {', '.join(unknown)} (see docs/FORMAT.md)")
    return p          # relative image paths resolve against the input file's directory (workdir)


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, Path):
        return _toml_value(str(v))
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(type(v))


def params_to_toml(p: Params, all_fields=False) -> str:
    """xpg.toml text for these parameters: experiment description always, processing settings only where they differ
    from the defaults (unless all_fields)."""
    default = Params()
    a2k = _attr_to_key()
    secs = {s: [] for s in SECTIONS}
    for f in dataclasses.fields(Params):
        a = f.name
        if a in _SKIP:
            continue
        v, d = getattr(p, a), getattr(default, a)
        same = (np.array_equal(np.asarray(v, dtype=object), np.asarray(d, dtype=object))
                if isinstance(v, (np.ndarray, list, tuple)) or isinstance(d, (np.ndarray, list, tuple)) else v == d)
        if v is None or (same and a not in _ALWAYS and not all_fields):
            continue
        sec, key = a2k.get(a, ("processing", a))
        try:
            secs[sec].append(f"{key} = {_toml_value(v)}")
        except TypeError:
            continue
    out = ["# XPG input (X-tal data Processing on GPU); run with: xpg xpg.toml", f"format = {FORMAT}"]
    if p.segments:
        out.append("# note: multi-panel detector segments of the source input are not represented here")
    for sec, lines in secs.items():
        if lines:
            out += ["", f"[{sec}]"] + lines
    return "\n".join(out) + "\n"


def is_toml(path) -> bool:
    return str(path).lower().endswith(".toml")


def read_input(path) -> Params:
    """xpg.toml, or another program's keyword input (converted on reading)."""
    from .config import read_xds_inp
    return read_toml(path) if is_toml(path) else read_xds_inp(path)


# ---------------------------------------------------------------------------------------------------------------
# merge input (``xpg merge merge.toml``)
#
#   [merge]
#   output = "merged"                  # output stem: <stem>_unmerged.mtz, <stem>_merged_iso.mtz, ...
#   space_group = 19                   # optional (default: from the inputs); unit_cell = [a, b, c, al, be, ga]
#   friedels_law = true
#   corrections = ["DECAY", "MODULATION", "ABSORP"]
#   reject_nonisomorphous = false      # drop data sets with negative / outlying Delta-CC1/2
#   zero_dose = 0.0                    # significance level of the zero-dose extrapolation (0 = off)
#   [[merge.input]]
#   file = "run1/xpg_unmerged.mtz"     # unmerged MTZ (+ its geometry JSON) or a legacy-format file
#   reference = true                   # optional: reference set for indexing / scale
#   crystal = "xtal1"                  # optional: data sets of the same crystal share scales
#   resolution = [50.0, 1.8]           # optional
#   weight = 1.0                       # optional sigma factor
#   reindex = [...]                    # optional 3x4 matrix
_MERGE_GLOBAL = {"space_group": "SPACE_GROUP_NUMBER", "unit_cell": "UNIT_CELL_CONSTANTS", "friedels_law": "FRIEDEL'S_LAW",
                 "corrections": "CORRECTIONS", "zero_dose": "0-DOSE_SIGNIFICANCE_LEVEL", "nbatch": "NBATCH", "snrc": "SNRC",
                 "wfac1": "WFAC1", "resolution_shells": "RESOLUTION_SHELLS"}
_MERGE_INPUT = {"crystal": "CRYSTAL_NAME", "resolution": "INCLUDE_RESOLUTION_RANGE", "weight": "WEIGHT",
                "reindex": "REIDX_ISET", "starting_dose": "STARTING_DOSE", "dose_rate": "DOSE_RATE"}


def _kw(v):
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (list, tuple)):
        return " ".join(_kw(x) for x in v)
    return str(v)


def read_merge_toml(path):
    """merge.toml -> (settings text for the merge engine, reject_nonisomorphous)."""
    import tomllib
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    fmt = doc.pop("format", 1)
    if not isinstance(fmt, int) or fmt > FORMAT:
        raise ValueError(f"{path}: format {fmt!r}: this XPG reads format {FORMAT} and older (update XPG)")
    if set(doc) - {"merge"}:
        raise ValueError(f"{path}: unknown settings {', '.join(sorted(set(doc) - {'merge'}))} (see docs/FORMAT.md)")
    m = doc.get("merge")
    if not m or not m.get("input"):
        raise ValueError(f"{path}: needs a [merge] table with [[merge.input]] entries")
    lines = [f"OUTPUT_FILE={m.get('output', 'merged')}"]
    for k, kw in _MERGE_GLOBAL.items():
        if k in m:
            lines.append(f"{kw}= {_kw(m[k])}")
    for inp in m["input"]:
        lines.append(f"INPUT_FILE={'*' if inp.get('reference') else ''}{inp['file']}")
        for k, kw in _MERGE_INPUT.items():
            if k in inp:
                lines.append(f"{kw}= {_kw(inp[k])}")
    unknown = [k for k in m if k not in _MERGE_GLOBAL and k not in ("input", "output", "reject_nonisomorphous")]
    unknown += [f"input.{k}" for inp in m["input"] for k in inp if k not in _MERGE_INPUT and k not in ("file", "reference")]
    if unknown:
        raise ValueError(f"{path}: unknown merge settings {', '.join(sorted(set(unknown)))} (see docs/FORMAT.md)")
    return "\n".join(lines) + "\n", bool(m.get("reject_nonisomorphous", False))

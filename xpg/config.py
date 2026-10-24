"""Reading XDS.INP files (converted to XPG settings).

The parser understands the free-format keyword syntax (``KEY= v1 v2 ... KEY2= ...``
with ``!`` comments, several keywords per line, repeated keywords) and converts
it into a typed :class:`Params` object. Unknown keywords are kept in
``Params.extra`` so existing XDS.INP files can be used as they are.
"""
from __future__ import annotations

import os
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_KEY_RE = re.compile(r"([A-Z0-9_'()/.\-]+)=")


def _truthy(x) -> bool:
    """TRUE / T / YES / 1 / ON."""
    return str(x).strip().upper() in ("T", "TRUE", "Y", "YES", "1", "ON") or str(x).strip().upper().startswith("T")


def parse_xds_inp_text(text: str) -> dict[str, list[list[str]]]:
    """Parse XDS.INP text into ``{KEY: [[values of 1st occurrence], ...]}``."""
    out: dict[str, list[list[str]]] = {}
    order = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0]
        if not line.strip():
            continue
        matches = list(_KEY_RE.finditer(line))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
            values = line[m.end():end].split()
            key = m.group(1).upper()
            out.setdefault(key, []).append(values)
            order.append((key, values))
    out["__order__"] = order     # keyword sequence (SEGMENT= blocks are order dependent)
    return out


def _floats(v, n=None):
    a = [float(x) for x in v]
    if n is not None and len(a) < n:
        raise ValueError(f"expected {n} values, got {v}")
    return a


@dataclass
class Params:
    # --- jobs ---
    jobs: list[str] = field(default_factory=lambda: ["DETECTOR", "BACKGROUND", "SPOTS", "INDEX",
                                                     "MASK", "INTEGRATE", "SCALE"])
    # --- images ---
    name_template: str = ""
    data_range: tuple[int, int] = (1, 1)
    spot_range: list[tuple[int, int]] = field(default_factory=list)
    background_range: tuple[int, int] | None = None
    exclude_data_range: list[tuple[int, int]] = field(default_factory=list)
    # --- detector ---
    detector: str = "PILATUS"
    nx: int = 0
    ny: int = 0
    qx: float = 0.0
    qy: float = 0.0
    orgx: float = 0.0
    orgy: float = 0.0
    distance: float = 0.0
    ed1: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    ed2: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    min_valid_pixel: float = 0.0
    overload: float = 1e9
    sensor_thickness: float = 0.0
    sensor_material: str = "Si"            # Si / CdTe / GaAs (SILICON= overrides mu)
    silicon_mu: float | None = None        # SILICON= (linear attenuation coefficient of sensor, 1/mm)
    trusted_region: tuple[float, float] = (0.0, 1.05)
    untrusted_rectangles: list[tuple[int, int, int, int]] = field(default_factory=list)
    untrusted_ellipses: list[tuple[int, int, int, int]] = field(default_factory=list)
    untrusted_quadrilaterals: list[tuple[int, ...]] = field(default_factory=list)
    value_range_trusted: tuple[float, float] = (6000.0, 30000.0)
    gain: float | None = None  # None -> estimated
    offset: float = 0.0                    # OFFSET= (constant dark-current level subtracted from each pixel)
    dark_current_image: str | None = None  # DARK_CURRENT_IMAGE= (per-pixel non-X-ray background, subtracted)
    segments: list = field(default_factory=list)   # SEGMENT= blocks (xpg.detector.Segment)
    x_geo_corr: str | None = None          # X-GEO_CORR= (CBF table, 1/100 pixel)
    y_geo_corr: str | None = None          # Y-GEO_CORR=
    two_theta: float = 0.0                 # informational (detector swing already in DIRECTION_OF_DETECTOR_*)
    # --- beam / goniometer ---
    wavelength: float = 1.0
    beam_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    rotation_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    osc: float = 0.1
    starting_angle: float = 0.0
    starting_frame: int | None = None
    polarization_fraction: float = 0.5
    polarization_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    # --- crystal ---
    space_group_number: int = 0
    unit_cell: tuple[float, ...] | None = None
    friedels_law: bool = True
    reidx: np.ndarray | None = None
    # --- spot finding ---
    strong_pixel: float = 3.0
    min_pixels_per_spot: int = 6           # default
    background_pixel: float = 2.0          # default
    spot_maximum_centroid: float = 3.0     # pixels
    cluster_radius: int = 2                # CLUSTER_RADIUS (strong pixels this close belong to one spot)
    background_template: bool = True       # xpg: background = k x static template (rings etc.) in integration
    export_legacy: bool = False            # also write the legacy-format text files
    prune_image_groups: bool = True        # Delta-CC1/2 image-group filter
    combine_intensities: bool = True       # profile / summation combination (I_mid)
    extra_lattices: bool = True            # xpg: integrate additional lattices separately
    min_lattice_separation: float = 0.5    # xpg: domains closer than this (deg) are integrated as one lattice
    ice_validate: bool = True              # xpg: exclude a detected ice ring only if its data are measurably worse
    prune_chunks: bool = True              # xpg: remove 5%-of-sweep chunks that lower the high-resolution CC1/2
    smooth_orientation: bool = True        # xpg: robust smoothing of the per-block orientation trajectory
    zero_dose: bool = False                # xpg: per-reflection zero-dose extrapolation (empirical Bayes)
    prune_fitness_threshold: float = 0.3   # xpg: remove image ranges with <I/sigma> below this x median (0 = off;
                                           # 0.3 removes only empty stretches, ~0.6 removes more)
    minpk: float = 75.0                    # MINPK (% of profile observed)
    isig_limit_symmetry: float = 6.0       # I/sigma limit of the symmetry analysis
    lattice_tolerance: float = 2.0         # degrees
    reference_data_set: str | None = None  # REFERENCE_DATA_SET= (MTZ or legacy-format file)
    free_r_reference: str | None = None    # xpg: MTZ whose FreeR_flag column is transferred/extended
    # --- indexing ---
    min_fraction_indexed: float = 0.5
    max_cell: float | None = None
    # --- refinement ---
    refine_idxref: list[str] = field(default_factory=lambda: ["BEAM", "AXIS", "ORIENTATION", "CELL"])
    # per phi block: detector POSITION with a Gaussian prior around the global geometry (XPG_POS_PRIOR, 0.5 px / 0.1
    # mm) instead of CELL
    refine_integrate: list[str] = field(default_factory=lambda: ["POSITION", "BEAM", "ORIENTATION"])
    refine_correct: list[str] = field(default_factory=lambda: ["POSITION", "BEAM", "ORIENTATION",
                                                               "CELL", "AXIS"])
    # --- integration ---
    beam_divergence: float | None = None       # e.s.d., degrees
    reflecting_range: float | None = None      # mosaicity e.s.d., degrees
    delphi: float = 5.0
    minimum_zeta: float = 0.05
    include_resolution: tuple[float, float] = (50.0, 0.0)
    exclude_resolution: list[tuple[float, float]] = field(default_factory=list)
    # --- correct / scaling ---
    corrections: list[str] = field(default_factory=lambda: ["DECAY", "MODULATION", "ABSORP"])
    strict_absorption: bool | None = None  # STRICT_ABSORPTION_CORRECTION (None: Friedel mates kept apart
                                           # for scaling whenever FRIEDEL'S_LAW=FALSE)
    minimum_i_sigma: float = 3.0           # MINIMUM_I/SIGMA (strong reflections for post-refinement)
    snrc: float = 0.0                      # SNRC (min I/sigma of reflections used to fit corrections)
                                           # 0.1 is not used: selecting on observed I/sigma biases weak-shell scales
    wfac1: float = 1.0                     # WFAC1 (sigma multiplier for misfit classification)
    nbatch: int | None = None              # NBATCH (number of scale batches; None -> 5 degree knots)
    fixed_scale_range: tuple | None = None # DATA_RANGE_FIXED_SCALE_FACTOR= first last factor
    # --- runtime ---
    lib: str | None = None
    extra: dict = field(default_factory=dict)
    workdir: Path = Path(".")

    @property
    def last_frame(self) -> int:
        return self.data_range[1]

    @property
    def phi_start_frame(self) -> int:
        return self.starting_frame if self.starting_frame is not None else self.data_range[0]

    def is_excluded(self, frame: int) -> bool:
        return any(a <= frame <= b for a, b in self.exclude_data_range)

    def frames(self, a=None, b=None):
        """Frame numbers in [a, b] (default DATA_RANGE) minus EXCLUDE_DATA_RANGE."""
        a = self.data_range[0] if a is None else a
        b = self.data_range[1] if b is None else b
        return [f for f in range(a, b + 1) if not self.is_excluded(f)]

    def copy(self, **kw) -> "Params":
        return dataclasses.replace(self, **kw)


def params_from_dict(d: dict[str, list[list[str]]], workdir: Path | str = ".") -> Params:
    p = Params(workdir=Path(workdir))
    used = set()

    def first(key):
        used.add(key)
        if key in d and d[key][0]:
            return d[key][0]
        # XPG extensions may also come from the environment (the XDS.INP keyword wins)
        if key.startswith("XPG_") and os.environ.get(key, "").strip():
            return os.environ[key].split()
        return None

    def all_(key):
        used.add(key)
        return d.get(key, [])

    if (v := first("JOB")) is not None:
        p.jobs = [x.upper() for x in v]
    if (v := first("NAME_TEMPLATE_OF_DATA_FRAMES")) is not None:
        p.name_template = v[0]
    if (v := first("DATA_RANGE")) is not None:
        p.data_range = (int(v[0]), int(v[1]))
    for v in all_("SPOT_RANGE"):
        p.spot_range.append((int(v[0]), int(v[1])))
    if (v := first("BACKGROUND_RANGE")) is not None:
        p.background_range = (int(v[0]), int(v[1]))
    for v in all_("EXCLUDE_DATA_RANGE"):
        p.exclude_data_range.append((int(v[0]), int(v[1])))
    if (v := first("DETECTOR")) is not None:
        p.detector = v[0].upper()
    for k, attr, t in [("NX", "nx", int), ("NY", "ny", int), ("QX", "qx", float), ("QY", "qy", float),
                       ("ORGX", "orgx", float), ("ORGY", "orgy", float),
                       ("DETECTOR_DISTANCE", "distance", float),
                       ("MINIMUM_VALID_PIXEL_VALUE", "min_valid_pixel", float),
                       ("OVERLOAD", "overload", float), ("SENSOR_THICKNESS", "sensor_thickness", float),
                       ("X-RAY_WAVELENGTH", "wavelength", float), ("OSCILLATION_RANGE", "osc", float),
                       ("STARTING_ANGLE", "starting_angle", float), ("STARTING_FRAME", "starting_frame", int),
                       ("FRACTION_OF_POLARIZATION", "polarization_fraction", float),
                       ("SPACE_GROUP_NUMBER", "space_group_number", int),
                       ("STRONG_PIXEL", "strong_pixel", float), ("SIGNAL_PIXEL", "strong_pixel", float),
                       ("MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT", "min_pixels_per_spot", int),
                       ("MINIMUM_FRACTION_OF_INDEXED_SPOTS", "min_fraction_indexed", float),
                       ("BEAM_DIVERGENCE_E.S.D.", "beam_divergence", float),
                       ("REFLECTING_RANGE_E.S.D.", "reflecting_range", float),
                       ("DELPHI", "delphi", float), ("MINIMUM_ZETA", "minimum_zeta", float),
                       ("GAIN", "gain", float), ("BACKGROUND_PIXEL", "background_pixel", float),
                       ("SPOT_MAXIMUM-CENTROID", "spot_maximum_centroid", float),
                       ("MINPK", "minpk", float), ("SILICON", "silicon_mu", float),
                       ("OFFSET", "offset", float), ("MINIMUM_I/SIGMA", "minimum_i_sigma", float),
                       ("SNRC", "snrc", float), ("WFAC1", "wfac1", float), ("NBATCH", "nbatch", int),
                       ("CLUSTER_RADIUS", "cluster_radius", lambda v: int(float(v)))]:
        if (v := first(k)) is not None:
            setattr(p, attr, t(v[0]))
    for k, attr in [("DIRECTION_OF_DETECTOR_X-AXIS", "ed1"), ("DIRECTION_OF_DETECTOR_Y-AXIS", "ed2"),
                    ("INCIDENT_BEAM_DIRECTION", "beam_direction"), ("ROTATION_AXIS", "rotation_axis"),
                    ("POLARIZATION_PLANE_NORMAL", "polarization_normal")]:
        if (v := first(k)) is not None:
            setattr(p, attr, np.array(_floats(v, 3)[:3]))
    if (v := first("UNIT_CELL_CONSTANTS")) is not None:
        p.unit_cell = tuple(_floats(v, 6)[:6])
    if (v := first("FRIEDEL'S_LAW")) is not None:
        p.friedels_law = _truthy(v[0])
    if (v := first("REIDX")) is not None:
        p.reidx = np.array(_floats(v, 12)[:12]).reshape(3, 4)
    if (v := first("TRUSTED_REGION")) is not None:
        p.trusted_region = (float(v[0]), float(v[1]))
    for v in all_("UNTRUSTED_RECTANGLE"):
        p.untrusted_rectangles.append(tuple(int(float(x)) for x in v[:4]))
    for v in all_("UNTRUSTED_ELLIPSE"):
        p.untrusted_ellipses.append(tuple(int(float(x)) for x in v[:4]))
    for v in all_("UNTRUSTED_QUADRILATERAL"):
        p.untrusted_quadrilaterals.append(tuple(int(float(x)) for x in v[:8]))
    if (v := first("VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS")) is not None:
        p.value_range_trusted = (float(v[0]), float(v[1]))
    if (v := first("INCLUDE_RESOLUTION_RANGE")) is not None:
        p.include_resolution = (float(v[0]), float(v[1]))
    for v in all_("EXCLUDE_RESOLUTION_RANGE"):
        p.exclude_resolution.append((float(v[0]), float(v[1])))
    if (v := first("CORRECTIONS")) is not None:
        p.corrections = [x.upper() for x in v]
    if (v := first("STRICT_ABSORPTION_CORRECTION")) is not None:
        p.strict_absorption = _truthy(v[0])
    if (v := first("DATA_RANGE_FIXED_SCALE_FACTOR")) is not None:
        p.fixed_scale_range = (int(v[0]), int(v[1]), float(v[2]) if len(v) > 2 else 1.0)
    if (v := first("DARK_CURRENT_IMAGE")) is not None:
        p.dark_current_image = v[0]
    if (v := first("X-GEO_CORR")) is not None:
        p.x_geo_corr = v[0]
    if (v := first("Y-GEO_CORR")) is not None:
        p.y_geo_corr = v[0]
    from .detector import parse_segments
    p.segments = parse_segments(d)
    for k in ("SEGMENT", "SEGMENT_ORGX", "SEGMENT_ORGY", "SEGMENT_DISTANCE", "DIRECTION_OF_SEGMENT_X-AXIS",
              "DIRECTION_OF_SEGMENT_Y-AXIS", "REFINE_SEGMENT", "__order__"):
        used.add(k)
    if (v := first("LIB")) is not None:
        p.lib = v[0]
    for key, attr in [("REFINE(IDXREF)", "refine_idxref"), ("REFINE(INTEGRATE)", "refine_integrate"),
                      ("REFINE(CORRECT)", "refine_correct")]:
        if (v := first(key)) is not None:
            setattr(p, attr, [x.upper() for x in v])
    # XPG-specific settings (prefixed XPG_)
    if (v := first("REFERENCE_DATA_SET")) is not None:
        p.reference_data_set = v[0]
    if (v := first("FREE_R_REFERENCE")) is not None:
        p.free_r_reference = v[0]
    if (v := first("SENSOR_MATERIAL")) is not None:
        p.sensor_material = v[0]
    if (v := first("XPG_EXTRA_LATTICES")) is not None:
        p.extra_lattices = _truthy(v[0])
    if (v := first("XPG_MIN_LATTICE_SEPARATION")) is not None:
        p.min_lattice_separation = float(v[0])
    if (v := first("XPG_ZERO_DOSE")) is not None:
        p.zero_dose = _truthy(v[0])
    if (v := first("XPG_SMOOTH_ORIENTATION")) is not None:
        p.smooth_orientation = _truthy(v[0])
    if (v := first("XPG_PRUNE_CHUNKS")) is not None:
        p.prune_chunks = _truthy(v[0])
    if (v := first("XPG_ICE_VALIDATE")) is not None:
        p.ice_validate = _truthy(v[0])
    if (v := first("XPG_PRUNE_FITNESS")) is not None:
        p.prune_fitness_threshold = float(v[0])
    if (v := first("XPG_EXPORT_LEGACY")) is not None:
        p.export_legacy = _truthy(v[0])
    if (v := first("XPG_PRUNE_IMAGES")) is not None:
        p.prune_image_groups = _truthy(v[0])
    if (v := first("XPG_BACKGROUND_TEMPLATE")) is not None:
        p.background_template = _truthy(v[0])
    if (v := first("XPG_MAX_CELL")) is not None:
        p.max_cell = float(v[0])
    if p.starting_frame is None:
        p.starting_frame = p.data_range[0]
    if not p.spot_range:
        p.spot_range = [p.data_range]
    if p.background_range is None:
        n = max(1, int(round(5.0 / max(p.osc, 1e-6))))
        p.background_range = (p.data_range[0], min(p.data_range[1], p.data_range[0] + n - 1))
    if p.detector == "EIGER" and p.overload > 1e8:
        p.overload = 2 ** 31 - 2
    p.extra.update({k: v for k, v in d.items() if k not in used and k not in p.extra})
    return p


def read_xds_inp(path: str | Path) -> Params:
    path = Path(path)
    return params_from_dict(parse_xds_inp_text(path.read_text(errors="ignore")), workdir=path.parent)


def format_xds_inp(p: Params) -> str:
    """Write a Params object in XDS.INP syntax."""
    v3 = lambda a: " ".join(f"{x:.6f}" for x in a)
    lines = [
        f"JOB= {' '.join(p.jobs)}",
        f"NAME_TEMPLATE_OF_DATA_FRAMES= {p.name_template}",
        f"DATA_RANGE= {p.data_range[0]} {p.data_range[1]}",
        *[f"SPOT_RANGE= {a} {b}" for a, b in p.spot_range],
        f"BACKGROUND_RANGE= {p.background_range[0]} {p.background_range[1]}",
        f"DETECTOR= {p.detector} MINIMUM_VALID_PIXEL_VALUE= {int(p.min_valid_pixel)} OVERLOAD= {int(p.overload)}",
        f"NX= {p.nx} NY= {p.ny} QX= {p.qx:g} QY= {p.qy:g}",
        f"ORGX= {p.orgx:.3f} ORGY= {p.orgy:.3f}",
        f"DETECTOR_DISTANCE= {p.distance:.4f}",
        f"DIRECTION_OF_DETECTOR_X-AXIS= {v3(p.ed1)}",
        f"DIRECTION_OF_DETECTOR_Y-AXIS= {v3(p.ed2)}",
        f"ROTATION_AXIS= {v3(p.rotation_axis)}",
        f"INCIDENT_BEAM_DIRECTION= {v3(p.beam_direction)}",
        f"X-RAY_WAVELENGTH= {p.wavelength:.6f}",
        f"OSCILLATION_RANGE= {p.osc:.6f}",
        f"STARTING_ANGLE= {p.starting_angle:.4f} STARTING_FRAME= {p.phi_start_frame}",
        f"FRACTION_OF_POLARIZATION= {p.polarization_fraction:g}",
        f"POLARIZATION_PLANE_NORMAL= {v3(p.polarization_normal)}",
        f"SENSOR_THICKNESS= {p.sensor_thickness:g}",
        f"TRUSTED_REGION= {p.trusted_region[0]:g} {p.trusted_region[1]:g}",
        f"VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS= {p.value_range_trusted[0]:g} {p.value_range_trusted[1]:g}",
        f"FRIEDEL'S_LAW= {'TRUE' if p.friedels_law else 'FALSE'}",
        f"SPACE_GROUP_NUMBER= {p.space_group_number}",
        f"SIGNAL_PIXEL= {p.strong_pixel:g}",
        f"INCLUDE_RESOLUTION_RANGE= {p.include_resolution[0]:g} {p.include_resolution[1]:g}",
        *[f"UNTRUSTED_RECTANGLE= {' '.join(map(str, r))}" for r in p.untrusted_rectangles],
        *[f"UNTRUSTED_ELLIPSE= {' '.join(map(str, r))}" for r in p.untrusted_ellipses],
    ]
    if p.unit_cell:
        lines.append("UNIT_CELL_CONSTANTS= " + " ".join(f"{x:.3f}" for x in p.unit_cell))
    if p.lib:
        lines.append(f"LIB= {p.lib}")
    if p.offset:
        lines.append(f"OFFSET= {p.offset:g}")
    for k, v in (("DARK_CURRENT_IMAGE", p.dark_current_image), ("X-GEO_CORR", p.x_geo_corr),
                 ("Y-GEO_CORR", p.y_geo_corr)):
        if v:
            lines.append(f"{k}= {v}")
    if p.segments:
        from .detector import format_segments
        lines += format_segments(p.segments)
    return "\n".join(lines) + "\n"

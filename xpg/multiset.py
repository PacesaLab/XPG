"""Joint scaling and merging of several unmerged data sets (XPG unmerged MTZ + geometry, or legacy-format files).

* input: merge.toml (read by inputs.read_merge_toml into the keyword form used here); XSCALE.INP files are read
  directly (OUTPUT_FILE=, INPUT_FILE= [*=reference], CRYSTAL_NAME=, INCLUDE_RESOLUTION_RANGE=, ...)
* consistent indexing: alternative-indexing operators (lattice symmetry not in the point group) chosen by
  correlation with the reference, then iterated against the merged data (cosym-like)
* joint scaling: per-dataset rotation scale, decay B, absorption (spherical harmonics in the crystal
  frame, reconstructed from the data set's geometry) and detector modulation; shared merged
  intensities; per-dataset error models (ISa); outlier rejection
* zero-dose extrapolation per reflection for datasets grouped by CRYSTAL_NAME (Diederichs 2006)
* per-dataset CC to the rest, inter-dataset correlation matrix and Delta-CC1/2 (sigma-tau method,
  Assmann, Brehm & Diederichs 2016) for spotting non-isomorphous or damaged datasets
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np

from .analysis import ambiguity_operators, best_reindex, merge_asu
from .geometry import Geometry, U_from_Astar, cell_from_Astar, rotation_matrices_np, unit
from .merging import format_stats_table, merging_statistics, unique_ids, cc_half_cutoff
from .scaling import Scaler
from .symops import apply_rot, d_spacing, hkl_key, to_asu


@dataclass
class InputSet:
    path: str
    header: dict
    hkl: np.ndarray
    I: np.ndarray
    sig: np.ndarray
    xd: np.ndarray
    yd: np.ndarray
    zd: np.ndarray
    rlp: np.ndarray
    crystal: str | None = None
    res: tuple = (1e9, 0.0)
    reference: bool = False
    output: int = 0
    geom: Geometry | None = None
    Astar: np.ndarray | None = None
    reindex: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=int))


def _floats(s):
    return [float(x) for x in s.split()]


def read_legacy_unmerged_full(path) -> InputSet:
    hdr = {}
    items = {}
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("!"):
                for m in re.finditer(r"([A-Z0-9_'()/.\-]+)=\s*([^=!]*?)(?=\s+[A-Z0-9_'()/.\-]+=|$)", line[1:].strip()):
                    k, v = m.group(1), m.group(2).strip()
                    hdr[k] = v
                    if k.startswith("ITEM_"):
                        items[k[5:]] = int(v) - 1
                continue
            if line.strip():
                rows.append(line.split())
    a = np.array(rows, float) if rows else np.zeros((0, 12))
    col = lambda name, default=None: a[:, items[name]] if name in items else default
    n = a.shape[0]
    ok = col("SIGMA(IOBS)") > 0                       # negative sigma = rejected misfit
    s = InputSet(path=str(path), header=hdr, hkl=a[:, :3].astype(int)[ok], I=col("IOBS")[ok], sig=col("SIGMA(IOBS)")[ok],
                 xd=col("XD", np.zeros(n))[ok], yd=col("YD", np.zeros(n))[ok], zd=col("ZD", np.zeros(n))[ok],
                 rlp=col("RLP", np.ones(n))[ok])
    s.geom, s.Astar = geometry_from_header(hdr)
    return s


def read_input_set(path) -> InputSet:
    """Unmerged input for scaling / merging: XPG unmerged MTZ (+ its geometry JSON) or a legacy-format text file."""
    if str(path).lower().endswith(".mtz"):
        from .io.xpgfiles import read_unmerged_mtz
        d = read_unmerged_mtz(path)
        ok = np.isfinite(d["I"]) & np.isfinite(d["sig"]) & (d["sig"] > 0)       # negative sigma = rejected misfit (legacy-format input)
        meta = d["meta"]
        hdr = {"SPACE_GROUP_NUMBER": str(d["spacegroup"].number),
               "UNIT_CELL_CONSTANTS": " ".join(f"{v:.4f}" for v in d["cell"]),
               "FRIEDEL'S_LAW": "TRUE" if meta.get("friedel_law", True) else "FALSE",
               "MERGE": "TRUE" if meta.get("merged", False) else "FALSE"}
        if d["geom"] is not None:
            hdr["X-RAY_WAVELENGTH"] = f"{d['geom'].wavelength:.6f}"
        s = InputSet(path=str(path), header=hdr, hkl=d["hkl"][ok], I=d["I"][ok], sig=d["sig"][ok], xd=d["x"][ok],
                     yd=d["y"][ok], zd=d["z"][ok], rlp=d["rlp"][ok])
        s.geom, s.Astar = d["geom"], d["Astar"]
        return s
    return read_legacy_unmerged_full(path)


def geometry_from_header(h):
    try:
        axis = np.array(_floats(h["ROTATION_AXIS"]))
        beam = np.array(_floats(h["INCIDENT_BEAM_DIRECTION"]))
        wl = float(h["X-RAY_WAVELENGTH"].split()[0])
        nxny = re.findall(r"[\d.]+", h.get("NX", "0"))
        g = Geometry(wavelength=wl, beam=unit(beam), axis=unit(axis), orgx=float(h["ORGX"].split()[0]),
                     orgy=float(h["ORGY"].split()[0]), distance=float(h["DETECTOR_DISTANCE"].split()[0]),
                     ed1=unit(_floats(h["DIRECTION_OF_DETECTOR_X-AXIS"])), ed2=unit(_floats(h["DIRECTION_OF_DETECTOR_Y-AXIS"])),
                     qx=float(h["QX"].split()[0]), qy=float(h["QY"].split()[0]), nx=int(float(nxny[0])),
                     ny=int(float(h["NY"].split()[0])), phi0=float(h["STARTING_ANGLE"].split()[0]),
                     osc=float(h["OSCILLATION_RANGE"].split()[0]), starting_frame=int(float(h["STARTING_FRAME"].split()[0])))
        R = np.array([_floats(h["UNIT_CELL_A-AXIS"]), _floats(h["UNIT_CELL_B-AXIS"]), _floats(h["UNIT_CELL_C-AXIS"])])
        return g, np.linalg.inv(R)
    except Exception:
        return None, None


_PER_INPUT = {"CRYSTAL_NAME", "INCLUDE_RESOLUTION_RANGE", "WEIGHT", "REIDX_ISET", "NBATCH", "CORRECTIONS",
              "STARTING_DOSE", "DOSE_RATE"}


def parse_xscale_inp(text):
    """Returns (globals dict, list of outputs) with per-input settings (keywords following an INPUT_FILE=
    apply to it)."""
    glob = {"friedel": True, "merge": False, "sg": None, "cell": None, "corrections": ["DECAY", "MODULATION", "ABSORP"],
            "zero_dose": None, "wfac1": 1.5, "snrc": 0.0, "nbatch": None, "batchsize": None, "shells": None,
            "reidx": None, "reference": None, "strict_absorption": None, "print_correlations": True,
            "save_images": True,
            "friedel_set": False, "merge_set": False}
    outputs = []
    cur = None
    last_inp = None
    ints = lambda v: np.array([int(float(x)) for x in v.split()[:12]]).reshape(3, 4)
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        for k, v in re.findall(r"([A-Z0-9_'()/.\-]+)=\s*([^=]*?)(?=\s+[A-Z0-9_'()/.\-]+=|$)", line):
            k = k.upper()
            if k == "OUTPUT_FILE":
                cur = {"file": v.split()[0], "inputs": [], "friedel": glob["friedel"], "merge": glob["merge"],
                       "friedel_set": False, "merge_set": False}
                outputs.append(cur)
                last_inp = None
            elif k == "INPUT_FILE":
                if cur is None:
                    cur = {"file": "XSCALE.HKL", "inputs": [], "friedel": glob["friedel"], "merge": glob["merge"],
                           "friedel_set": False, "merge_set": False}
                    outputs.append(cur)
                f = v.split()[0]
                last_inp = {"file": f.lstrip("*"), "reference": f.startswith("*"), "crystal": None, "res": (1e9, 0.0),
                            "weight": 1.0, "reidx": None, "nbatch": None, "starting_dose": None, "dose_rate": None,
                            "corrections": None}
                cur["inputs"].append(last_inp)
            elif k in _PER_INPUT and last_inp is not None:
                if k == "CRYSTAL_NAME":
                    last_inp["crystal"] = v.split()[0]
                elif k == "INCLUDE_RESOLUTION_RANGE":
                    last_inp["res"] = tuple(float(x) for x in v.split()[:2])
                elif k == "WEIGHT":
                    last_inp["weight"] = float(v.split()[0])
                elif k == "REIDX_ISET":
                    last_inp["reidx"] = ints(v)
                elif k == "NBATCH":
                    last_inp["nbatch"] = int(v.split()[0])
                elif k == "CORRECTIONS":
                    last_inp["corrections"] = v.upper().split()
                elif k == "STARTING_DOSE":
                    last_inp["starting_dose"] = float(v.split()[0])
                elif k == "DOSE_RATE":
                    last_inp["dose_rate"] = float(v.split()[0])
            elif k == "FRIEDEL'S_LAW":
                val = v.upper().startswith("T")
                tgt = cur if cur is not None else glob
                tgt["friedel"], tgt["friedel_set"] = val, True
            elif k == "MERGE":
                val = v.upper().startswith("T")
                tgt = cur if cur is not None else glob
                tgt["merge"], tgt["merge_set"] = val, True
            elif k == "SPACE_GROUP_NUMBER":
                glob["sg"] = int(v.split()[0])
            elif k == "UNIT_CELL_CONSTANTS":
                glob["cell"] = tuple(float(x) for x in v.split()[:6])
            elif k == "CORRECTIONS":
                glob["corrections"] = v.upper().split()
            elif k == "0-DOSE_SIGNIFICANCE_LEVEL":
                glob["zero_dose"] = float(v.split()[0])
            elif k == "WFAC1":
                glob["wfac1"] = float(v.split()[0])
            elif k == "SNRC":
                glob["snrc"] = float(v.split()[0])
            elif k == "NBATCH":
                glob["nbatch"] = int(v.split()[0])
            elif k == "BATCHSIZE":
                glob["batchsize"] = float(v.split()[0])
            elif k == "RESOLUTION_SHELLS":
                glob["shells"] = [float(x) for x in v.split()]
            elif k == "REIDX":
                glob["reidx"] = ints(v)
            elif k == "REFERENCE_DATA_SET":
                glob["reference"] = v.split()[0]
            elif k == "STRICT_ABSORPTION_CORRECTION":
                glob["strict_absorption"] = v.upper().startswith("T")
            elif k == "PRINT_CORRELATIONS":
                glob["print_correlations"] = v.upper().startswith("T")
            elif k == "SAVE_CORRECTION_IMAGES":
                glob["save_images"] = v.upper().startswith("T")
    return glob, outputs


def _reidx_R(M):
    """REIDX (h' = M[:, :3] h) in the h' = h R convention used here."""
    return np.asarray(M)[:, :3].T.astype(int)


def common_scale_fit(I_ref, I_k, s2, nbins=12, min_n=30):
    """Relative scale between two merged data sets on common reflections: fit ln(sum I_ref / sum I_k)
    per resolution shell as a + b s^2 (scale K = e^a and relative B = -2b). Returns (a, b)."""
    ok = np.isfinite(I_ref) & np.isfinite(I_k)
    if ok.sum() < 3 * min_n:
        return 0.0, 0.0
    q = np.quantile(s2[ok], np.linspace(0, 1, nbins + 1))
    xs, ys, ws = [], [], []
    for i in range(nbins):
        m = ok & (s2 >= q[i]) & (s2 <= q[i + 1])
        sr, sk = I_ref[m].sum(), I_k[m].sum()
        if m.sum() >= min_n and sr > 0 and sk > 0:
            xs.append(s2[m].mean()); ys.append(np.log(sr / sk)); ws.append(sk)
    if len(xs) < 2:
        return (float(np.mean(ys)) if ys else 0.0), 0.0
    b, a = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
    return float(a), float(b)


def cc_half_sigma_tau(uid, I, sig=None):
    """CC1/2 by the sigma-tau method (no random split): var(mu) = var(<I>_h) - <s_h^2/n_h>."""
    n = uid.max() + 1
    cnt = np.bincount(uid, minlength=n)
    m1 = np.bincount(uid, I, n) / np.maximum(cnt, 1)
    m2 = np.bincount(uid, I * I, n) / np.maximum(cnt, 1)
    ok = cnt >= 2
    s2 = (m2 - m1 ** 2) * cnt / np.maximum(cnt - 1, 1)
    vy = np.var(m1[ok])
    ve = np.mean(s2[ok] / cnt[ok])
    return float((vy - ve) / (vy + ve)) if vy + ve > 0 else np.nan


def zero_dose_merge(uid, I, sig, dose, significance=0.1, min_obs=3):
    """Per-reflection weighted linear fit I = I0 + b dose; I0 replaces the mean where the slope is
    significant at the given level (0-DOSE_SIGNIFICANCE_LEVEL)."""
    from scipy.stats import norm
    n = uid.max() + 1
    w = 1 / sig ** 2
    S = np.bincount(uid, w, n)
    Sx = np.bincount(uid, w * dose, n)
    Sy = np.bincount(uid, w * I, n)
    Sxx = np.bincount(uid, w * dose * dose, n)
    Sxy = np.bincount(uid, w * dose * I, n)
    cnt = np.bincount(uid, minlength=n)
    D = S * Sxx - Sx ** 2
    mean = Sy / np.maximum(S, 1e-30)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (S * Sxy - Sx * Sy) / D
        a = (Sy - b * Sx) / S
        sb = np.sqrt(S / D)
        sa = np.sqrt(Sxx / D)
    tval = np.abs(b / sb)
    # the slope error comes from the (error-model calibrated) sigmas, not from the scatter, so the test
    # statistic is normal: a t quantile with n-2 dof made the test far too conservative (2% at a 10% level)
    crit = norm.ppf(1 - significance / 2)
    use = (cnt >= min_obs) & (D > 1e-12 * S ** 2) & np.isfinite(tval) & (tval > crit)
    I0 = np.where(use, a, mean)
    s0 = np.where(use, sa, 1 / np.sqrt(np.maximum(S, 1e-30)))
    return I0, s0, use


DCC_HARMFUL = 0.02     # fallback when the noise cannot be estimated
DCC_K = 6.0            # harmful = Delta-CC1/2 below -DCC_K x its null noise ...
DCC_FLOOR = 0.01       # ... and below -DCC_FLOOR
MIN_SET_CC = 0.1       # sets below this CC (raw I) to the reference share no structure with it: excluded


def _dcc_null_sd(uid, I, act, ds, nsets, n_rep=2, seed=7, max_obs=200000):
    """Sampling noise of per-set Delta-CC1/2: the same statistic for observations randomly re-assigned to pseudo-sets
    with the real sets' sizes (isomorphous by construction); sd over pseudo-sets and repeats."""
    rng = np.random.default_rng(seed)
    a = np.flatnonzero(act)
    if a.size < 1000 or nsets < 2:
        return np.nan
    frac = 1.0
    if a.size > max_obs:          # whole reflections subsampled (the statistic needs all observations of each)
        ur = np.unique(uid[a])
        keep_r = rng.choice(ur, max(int(ur.size * max_obs / a.size), 100), replace=False)
        frac = keep_r.size / ur.size
        a = a[np.isin(uid[a], keep_r)]
    sizes = np.bincount(ds[a], minlength=nsets)
    cc_all = cc_half_sigma_tau(uid[a], I[a])
    vals = []
    for _ in range(n_rep):
        lab = np.repeat(np.arange(nsets), sizes)
        rng.shuffle(lab)
        for i in range(nsets):
            keep = a[lab != i]
            if keep.size > 100:
                vals.append(cc_all - cc_half_sigma_tau(uid[keep], I[keep]))
    vals = np.array(vals)
    # pseudo-sets are isomorphous: their Delta-CC1/2 scatter about a size-dependent mean; use the scatter
    # sampling noise of a CC scales as 1/sqrt(number of reflections): rescale from the subsample to all data
    return float(np.std(vals) * np.sqrt(frac)) if vals.size > 2 else np.nan


def run_multiset(inp_path, outdir=None, log=print, anisotropy=True, reject_nonisomorphous=False, _exclude=(),
               inp_text=None, export_legacy=False, stop_at_reference=False, prior=None, compare_shells=None):
    """stop_at_reference: when the rejections leave only the reference set, return without re-scaling it alone (an
    evaluation that only needs to know whether anything merges; no outputs, cc_half_cutoff None). prior: a fixed
    absorption / modulation prior width instead of the cross-validated one. compare_shells: resolution shells (e.g.
    another data set's) over which the statistics are also returned (shells_compare, overall_compare), for a
    comparison at equal resolution."""
    inp_path = Path(inp_path)
    wd = Path(outdir or inp_path.parent)
    wd.mkdir(parents=True, exist_ok=True)
    # inp_text: settings given directly (pipeline-internal merges write no input file)
    glob, outputs = parse_xscale_inp(inp_text if inp_text is not None else inp_path.read_text())
    sets: list[InputSet] = []
    for oi, out in enumerate(outputs):
        for inp in out["inputs"]:
            p = Path(inp["file"])
            if not p.is_absolute():
                p = inp_path.parent / p
            if str(inp["file"]) in _exclude:
                log(f"  MERGE: excluding {p} (Delta-CC1/2 outlier)")
                continue
            s = read_input_set(p)
            s.crystal, s.res, s.reference, s.output = inp["crystal"], inp["res"], inp["reference"], oi
            s.inp = inp
            dmax, dmin = s.res
            d = 1 / np.linalg.norm(s.hkl @ s.Astar.T, axis=1) if s.Astar is not None else np.full(len(s.I), 10.0)
            keep = (d <= dmax) & (d >= dmin)
            for k in ("hkl", "I", "sig", "xd", "yd", "zd", "rlp"):
                setattr(s, k, getattr(s, k)[keep])
            if inp["weight"] != 1.0:                       # WEIGHT=: multiplies the input sigmas
                s.sig = s.sig * inp["weight"]
            for M in (glob["reidx"], inp["reidx"]):        # REIDX= (all inputs), REIDX_ISET= (this input)
                if M is not None:
                    R = _reidx_R(M)
                    s.hkl = s.hkl @ R
                    s.reindex = s.reindex @ R
            sets.append(s)
            extra = f" (WEIGHT {inp['weight']:g})" if inp["weight"] != 1.0 else ""
            extra += " (REIDX_ISET)" if inp["reidx"] is not None else ""
            log(f"  MERGE: read {len(s.I):8d} reflections from {p}{extra}")
    # --- symmetry and cell ---
    sgnum = glob["sg"] or int(sets[0].header.get("SPACE_GROUP_NUMBER", "1").split()[0])
    sg = gemmi.find_spacegroup_by_number(sgnum)
    cells = np.array([cell_from_Astar(s.Astar) if s.Astar is not None else
                      _floats(s.header["UNIT_CELL_CONSTANTS"]) for s in sets])
    cell = np.array(glob["cell"]) if glob["cell"] else np.median(cells, axis=0)
    log(f"  MERGE: space group {sg.xhm()} (#{sg.number}); cell {np.array2string(cell, precision=3)}")
    for i, (s, c) in enumerate(zip(sets, cells)):
        dev = np.abs(c[:3] - cell[:3]) / cell[:3]
        flag = "  <-- non-isomorphous? (>1% cell difference)" if dev.max() > 0.01 else ""
        log(f"     set {i + 1:3d}: cell {np.array2string(c, precision=2)}{flag}")
    # --- consistent indexing ---
    ops = ambiguity_operators(cell, sg)
    if len(ops) > 1:
        ref_i = next((i for i, s in enumerate(sets) if s.reference), 0)
        log(f"  MERGE: {len(ops) - 1} alternative indexing operator(s); reference = set {ref_i + 1}")
        rk, rI = merge_asu(sets[ref_i].hkl, sets[ref_i].I, sets[ref_i].sig, sg)
        for it in range(2):
            for i, s in enumerate(sets):
                if (it == 0 and i == ref_i) or s.inp["reidx"] is not None:   # user-defined REIDX_ISET wins
                    continue
                R, ccs = best_reindex(s.hkl, s.I, s.sig, rk, rI, cell, sg, ops)
                if not np.array_equal(R, np.eye(3, dtype=int)):
                    s.hkl = apply_rot(s.hkl, R)
                    s.reindex = s.reindex @ R
                    log(f"     set {i + 1}: reindexed by {R.tolist()} (CCs per operator: "
                        f"{', '.join(f'{c:.3f}' for c, _ in ccs)})")
            # iterate against all merged data (cosym-like consensus)
            rk, rI = merge_asu(np.concatenate([s.hkl for s in sets]), np.concatenate([s.I for s in sets]),
                               np.concatenate([s.sig for s in sets]), sg)
    # --- sets without common structure: CC of raw intensities to the reference set ~ 0 (even unrelated structures
    # share the resolution fall-off, so this is far below any isomorphous set): a misindexed lattice or another
    # crystal form. Joint scaling cannot bring them into agreement; they are excluded before scaling.
    if len(sets) > 1:
        ref_i = next((i for i, s in enumerate(sets) if s.reference), 0)
        rk, rI = merge_asu(sets[ref_i].hkl, sets[ref_i].I, sets[ref_i].sig, sg)
        keep = []
        for i, s in enumerate(sets):
            if i != ref_i:
                _, ccs = best_reindex(s.hkl, s.I, s.sig, rk, rI, cell, sg, [np.eye(3, dtype=int)])
                cc, nc = ccs[0]
                if nc >= 100 and not cc >= MIN_SET_CC:
                    log(f"  MERGE WARNING: set {i + 1} ({Path(s.path).name}) correlates with the reference set at CC "
                        f"{cc:.3f} on {nc} common reflections (no common structure: misindexed lattice or another "
                        f"crystal form): excluded")
                    continue
            keep.append(s)
        sets = keep
    # --- assemble observations ---
    hkl = np.concatenate([s.hkl for s in sets])
    I = np.concatenate([s.I for s in sets])
    sig = np.concatenate([s.sig for s in sets])
    ds = np.concatenate([np.full(len(s.I), i, np.int32) for i, s in enumerate(sets)])
    # defaults: Friedel mates separate / data unmerged only if so in all inputs
    hdr_friedel_false = all(str(s.header.get("FRIEDEL'S_LAW", "TRUE")).upper().startswith("F") for s in sets)
    hdr_merged = any(str(s.header.get("MERGE", "FALSE")).upper().startswith("T") for s in sets)
    for o in outputs:
        if not o["friedel_set"] and not glob["friedel_set"]:
            o["friedel"] = not hdr_friedel_false
        if not o["merge_set"] and not glob["merge_set"]:
            o["merge"] = hdr_merged
    friedel_all = all(o["friedel"] for o in outputs)
    hasu, isym, plus = to_asu(hkl, sg)
    strict = (not friedel_all) if glob["strict_absorption"] is None else glob["strict_absorption"]
    uid, _ = unique_ids(hasu, plus, anomalous=strict)
    uid_c = uid
    out_of = np.array([s.output for s in sets])[ds]
    multi_out = len(outputs) > 1
    if multi_out:
        # MAD / several outputs: each output (wavelength) has its own merged intensities, so dispersive
        # differences are not averaged away; the outputs are put on a common scale afterwards
        _, uid = np.unique(uid.astype(np.int64) * len(outputs) + out_of, return_inverse=True)
    d = d_spacing(hasu, cell)
    phi, s1c, xs, ys = [], [], [], []
    for s in sets:
        g = s.geom
        if g is None:
            phi.append(s.zd * 0.1)
            s1c.append(np.tile([0, 0, 1.0], (len(s.I), 1)))
        else:
            ph = g.z_to_phi(s.zd)
            X = g.pixel_to_lab(s.xd, s.yd)
            e = X / np.linalg.norm(X, axis=1, keepdims=True)
            R = rotation_matrices_np(g.axis, -np.radians(ph))
            # undo the reindexing applied to hkl so the crystal frame matches the reindexed cell
            Ast = s.Astar @ np.linalg.inv(s.reindex.T)
            U = U_from_Astar(Ast)
            s.U = U
            s1c.append(np.einsum("nij,nj->ni", R, e) @ U)
            phi.append(ph)
        xs.append(s.xd)
        ys.append(s.yd)
    phi, s1c, xs, ys = map(np.concatenate, (phi, s1c, xs, ys))
    nx = max((s.geom.nx for s in sets if s.geom is not None), default=4000)
    ny = max((s.geom.ny for s in sets if s.geom is not None), default=4000)
    log(f"  MERGE: joint scaling of {len(sets)} data sets, {I.size} observations, {uid.max() + 1} unique reflections")
    nb = [s.inp["nbatch"] or glob["nbatch"] for s in sets]
    sc = Scaler(I, sig, uid, phi, 1 / d ** 2, s1c, xs, ys, nx, ny, corrections=glob["corrections"], ds=ds,
                nsig=6.0 * glob["wfac1"] / 1.5, snrc=glob["snrc"], nbatch=nb if any(nb) else None,
                rot_spacing=glob["batchsize"] or 5.0)
    if prior is not None:
        sc.prior_sd = float(prior)
        log(f"  SCALE: absorption/modulation prior width {sc.prior_sd:g} (fixed)")
    sc.run(log=log, select_prior=prior is None)
    Is, ss, act = sc.scaled()
    s2o = 1 / d ** 2
    if glob["save_images"]:
        from .io.corrimages import write_correction_images
        for i, s in enumerate(sets):
            if s.geom is not None and getattr(s, "U", None) is not None:
                try:
                    write_correction_images(wd, sc, s.geom, s.U, dataset=i, suffix=f"_{i + 1:03d}",
                                            smax2=float(s2o.max()))
                except Exception as e:  # pragma: no cover
                    log(f"  MERGE: correction images for set {i + 1} not written ({e})")
    if multi_out:
        ref_o = sets[next((i for i, s_ in enumerate(sets) if s_.reference), 0)].output
        key = hkl_key(hasu)
        uk, inv = np.unique(key, return_inverse=True)

        def means(o):
            m = act & (out_of == o)
            w = np.where(m, 1 / ss ** 2, 0.0)
            den = np.bincount(inv, w, uk.size)
            return np.where(den > 0, np.bincount(inv, w * Is, uk.size) / np.maximum(den, 1e-30), np.nan)
        Iref = means(ref_o)
        s2u = np.zeros(uk.size)
        s2u[inv] = s2o
        for o in range(len(outputs)):
            if o == ref_o:
                continue
            a_, b_ = common_scale_fit(Iref, means(o), s2u)
            f = np.exp(a_ + b_ * s2o)
            m = out_of == o
            Is[m] *= f[m]
            ss[m] *= f[m]
            log(f"  MERGE: output {outputs[o]['file']} put on the scale of {outputs[ref_o]['file']}: "
                f"K {np.exp(a_):.4f}, relative B {-2 * b_:+.2f} A^2")
    if glob["reference"]:
        from .analysis import read_reference
        rp = Path(glob["reference"])
        rp = rp if rp.is_absolute() else inp_path.parent / rp
        rk, rI = read_reference(str(rp), sg)
        key = hkl_key(hasu)
        uk, inv = np.unique(key, return_inverse=True)
        w = np.where(act, 1 / ss ** 2, 0.0)
        Im = np.bincount(inv, w * Is, uk.size) / np.maximum(np.bincount(inv, w, uk.size), 1e-30)
        refd = dict(zip(rk.tolist(), rI.tolist()))
        Ir = np.array([refd.get(k, np.nan) for k in uk.tolist()])
        s2u = np.zeros(uk.size)
        s2u[inv] = s2o
        a_, b_ = common_scale_fit(Ir, Im, s2u)
        f = np.exp(a_ + b_ * s2o)
        Is, ss = Is * f, ss * f
        log(f"  MERGE: REFERENCE_DATA_SET {rp.name}: overall scale K {np.exp(a_):.4f}, relative B {-2 * b_:+.2f} A^2 "
            f"({np.isfinite(Ir).sum()} common reflections)")
    # --- per-dataset quality (across all outputs: dispersive differences count as non-isomorphism) ---
    dcc_info = {}
    log("\n  MERGE: per data set: N, ISa, CC to rest, Delta-CC1/2 (sigma-tau; negative = makes data worse)")
    cc_all = cc_half_sigma_tau(uid_c[act], Is[act])
    nsets = len(sets)
    w = 1 / ss ** 2
    n_u = uid_c.max() + 1
    tot_w = np.bincount(uid_c[act], w[act], n_u)
    tot_wi = np.bincount(uid_c[act], (w * Is)[act], n_u)
    rows = []
    means = []
    for i in range(nsets):
        m = act & (ds == i)
        wi = np.bincount(uid_c[m], w[m], n_u)
        wii = np.bincount(uid_c[m], (w * Is)[m], n_u)
        mi = wii / np.maximum(wi, 1e-30)
        rest = (tot_wi - wii) / np.maximum(tot_w - wi, 1e-30)
        ok = (wi > 0) & (tot_w - wi > 0)
        cc = np.corrcoef(mi[ok], rest[ok])[0, 1] if ok.sum() > 20 else np.nan
        m_wo = act & (ds != i)
        dcc = cc_all - cc_half_sigma_tau(uid_c[m_wo], Is[m_wo]) if m_wo.any() else np.nan
        means.append((mi, wi > 0))
        rows.append((i, m.sum(), sc.isa(i), cc, dcc))
    dccs = np.array([r[4] for r in rows])
    thr = np.nanmean(dccs) - 3 * np.nanstd(dccs) if nsets >= 4 else -np.inf
    for i, n, isa, cc, dcc in rows:
        flag = "  <-- outlier (Assmann et al. 2016 criterion)" if nsets >= 3 and (dcc < thr or dcc < 0) else ""
        log(f"     set {i + 1:3d} {sets[i].path[-40:]:>40s}  N {n:8d}  ISa {isa:7.1f}  CC(rest) {cc:6.3f}  "
            f"dCC1/2 {dcc:+.4f}{flag}")
    if nsets > 1 and glob["print_correlations"]:
        log("\n  MERGE: correlations between data sets (common reflections)")
        for i in range(nsets):
            line = []
            for j in range(nsets):
                a_, ma = means[i]
                b_, mb = means[j]
                ok = ma & mb
                line.append(f"{np.corrcoef(a_[ok], b_[ok])[0, 1]:6.3f}" if ok.sum() > 20 and i != j else "   -  ")
            log(f"     {i + 1:3d} " + " ".join(line))
    bad = [sets[i] for i, n, isa, cc, dcc in rows if nsets >= 3 and (dcc < thr or dcc < 0)]
    if reject_nonisomorphous == "worst":
        # only clearly harmful sets, one at a time (removing one changes the others' Delta-CC1/2): a set whose
        # inclusion lowers CC1/2 by more than DCC_K x the Delta-CC1/2 noise of THIS data (null: observations randomly
        # re-assigned to pseudo-sets of the same sizes, i.e. isomorphous by construction), at least DCC_FLOOR -- e.g.
        # a spurious lattice; mildly negative lattices (-0.015 at noise ~0.003) still improved the merges
        noise = _dcc_null_sd(uid_c, Is, act, ds, nsets)
        thr_h = max(DCC_K * noise, DCC_FLOOR) if np.isfinite(noise) else DCC_HARMFUL
        dcc_info = dict(dcc_threshold=float(thr_h), dcc_noise=float(noise),
                        delta_cc_half={Path(sets[i].path).name: float(d) for i, *_x, d in rows})
        log(f"  MERGE: Delta-CC1/2 noise (random pseudo-sets) {noise:.4f}; a set is dropped below {-thr_h:+.4f}")
        worst = min(rows, key=lambda r: r[4] if np.isfinite(r[4]) else np.inf)
        bad = [sets[worst[0]]] if nsets >= 2 and np.isfinite(worst[4]) and worst[4] < -thr_h else []
    if reject_nonisomorphous and bad and len(sets) - len(bad) >= (1 if reject_nonisomorphous == "worst" else 2):
        names = {inp["file"] for out in outputs for inp in out["inputs"]
                 if any(Path(b.path).resolve() == (Path(inp["file"]) if Path(inp["file"]).is_absolute()
                                                   else inp_path.parent / inp["file"]).resolve() for b in bad)}
        rest = [s for s in sets if all(s is not b for b in bad)]       # identity: InputSet holds arrays
        if stop_at_reference and len(rest) == 1 and rest[0] is sets[next((i for i, s in enumerate(sets) if s.reference), 0)]:
            log(f"\n  MERGE: rejecting {len(bad)} data set(s) with negative/outlying Delta-CC1/2; only the reference "
                f"set is left (not re-scaled)")
            return {o["file"]: dict(cc_half_cutoff=None, n_obs=None, excluded=list(_exclude) + sorted(names))
                    for o in outputs} | {"per_dataset": rows}
        log(f"\n  MERGE: rejecting {len(bad)} data set(s) with negative/outlying Delta-CC1/2 and re-scaling")
        return run_multiset(inp_path, outdir, log, anisotropy, reject_nonisomorphous, tuple(_exclude) + tuple(names),
                          inp_text=inp_text, export_legacy=export_legacy, stop_at_reference=stop_at_reference,
                          prior=prior, compare_shells=compare_shells)
    # --- outputs ---
    from .io.mtz import write_unmerged_mtz
    from .outputs import write_merged_outputs
    results = {}
    for oi, out in enumerate(outputs):
        sel = np.isin(ds, [i for i, s in enumerate(sets) if s.output == oi])
        a_ = act & sel
        anom = not out["friedel"]
        shells, overall, merged = merging_statistics(hasu[a_], plus[a_], Is[a_], ss[a_], cell, sg, nbins=10,
                                                     anomalous=True, shell_limits=glob["shells"])
        if glob["zero_dose"] is not None and any(sets[i].crystal for i in np.unique(ds[a_])):
            dose = np.zeros(I.size)
            by_xtal = {}
            for i, s in enumerate(sets):
                if s.crystal:
                    m = ds == i
                    if s.inp["starting_dose"] is not None or s.inp["dose_rate"] is not None:
                        # STARTING_DOSE / DOSE_RATE (dose per image) given: accumulated dose per observation
                        dose[m] = (s.inp["starting_dose"] or 0.0) + (s.inp["dose_rate"] or 1.0) * s.zd
                        continue
                    off = by_xtal.get(s.crystal, 0.0)
                    dose[m] = off + (s.zd - s.zd.min())
                    by_xtal[s.crystal] = off + (s.zd.max() - s.zd.min()) + 1
            # Bijvoet mates are measured at different doses: with FRIEDEL'S_LAW=FALSE they are extrapolated
            # separately (merging them first turns the anomalous difference into a spurious dose slope)
            uidm, ukm = unique_ids(hasu[a_], plus[a_], anomalous=anom)
            I0, s0, used = zero_dose_merge(uidm, Is[a_], ss[a_], dose[a_], glob["zero_dose"])
            pk = hkl_key(merged["hkl"])
            if anom:
                pos = np.searchsorted(pk, ukm >> 1)
                minus = (ukm & 1).astype(bool)
                for arr, sarr, sel_ in (("Iplus", "sigIplus", ~minus), ("Iminus", "sigIminus", minus)):
                    merged[arr] = np.full(pk.size, np.nan)
                    merged[sarr] = np.full(pk.size, np.nan)
                    merged[arr][pos[sel_]] = I0[sel_]
                    merged[sarr][pos[sel_]] = s0[sel_]
                wp = np.where(np.isfinite(merged["sigIplus"]), 1 / merged["sigIplus"] ** 2, 0.0)
                wm = np.where(np.isfinite(merged["sigIminus"]), 1 / merged["sigIminus"] ** 2, 0.0)
                merged["I"] = (wp * np.nan_to_num(merged["Iplus"]) + wm * np.nan_to_num(merged["Iminus"])) / (wp + wm)
                merged["sig"] = 1 / np.sqrt(wp + wm)
            else:
                pos = np.searchsorted(pk, ukm)
                merged["I"][pos], merged["sig"][pos] = I0, s0
            log(f"  MERGE: zero-dose extrapolation applied to {used.sum()} of {used.size} reflections "
                f"(significance {glob['zero_dose']})")
        log(f"\n  MERGE: output {out['file']} ({'merged' if out['merge'] else 'unmerged'}, "
            f"FRIEDEL'S_LAW={'TRUE' if out['friedel'] else 'FALSE'})")
        log(format_stats_table(shells, overall, anomalous=anom))
        fine, _, _ = merging_statistics(hasu[a_], plus[a_], Is[a_], ss[a_], cell, sg, nbins=30)
        dcut = cc_half_cutoff(fine, 0.3)
        if dcut:
            log(f"  MERGE: CC1/2 = 0.3 resolution estimate {dcut:.2f} A")
        cmp = {}
        if compare_shells:
            sh_c, ov_c, _ = merging_statistics(hasu[a_], plus[a_], Is[a_], ss[a_], cell, sg, dmin=compare_shells[-1].dmin,
                                               dmax=compare_shells[0].dmax, anomalous=True,
                                               shell_limits=[s.dmin for s in compare_shells])
            cmp = dict(shells_compare=sh_c, overall_compare=ov_c)
        base = wd / out["file"]
        if export_legacy or str(out['file']).upper().endswith(('.HKL', '.AHKL')):
            _write_legacy_merged(base, sets, sg, cell, hkl[sel], Is[sel], np.where(act, ss, -ss)[sel], xs[sel], ys[sel],
                           np.concatenate([s.zd for s in sets])[sel], ds[sel], out, merged)
        stem = base.with_suffix("")
        first = sets[int(ds[np.flatnonzero(sel)[0]])]
        wl = first.geom.wavelength if first.geom is not None else 1.0
        write_unmerged_mtz(f"{stem}_unmerged.mtz", sg, cell, wl, hasu[a_], isym[a_],
                           (ds[a_] * 10000 + np.floor(np.concatenate([s.zd for s in sets])[a_]) + 1).astype(int),
                           Is[a_], ss[a_], xs[a_], ys[a_], phi[a_], np.ones(a_.sum()))
        write_merged_outputs(merged, cell, sg, dcut, wl, Path(f"{stem}"), log=log, anisotropy=anisotropy)
        results[out["file"]] = dict(overall=overall, shells=shells, cc_half_cutoff=dcut, n_obs=int(a_.sum()),
                                    excluded=list(_exclude) if reject_nonisomorphous else None, **dcc_info, **cmp)
    results["per_dataset"] = rows
    return results


def _write_legacy_merged(path, sets, sg, cell, hkl, I, sig, xd, yd, zd, iset, out, merged):
    with open(path, "w") as f:
        f.write(f"!FORMAT=XDS_ASCII    MERGE={'TRUE' if out['merge'] else 'FALSE'}    "
                f"FRIEDEL'S_LAW={'TRUE' if out['friedel'] else 'FALSE'}\n")
        f.write("!Generated by XPG\n")
        f.write(f"!SPACE_GROUP_NUMBER={sg.number:5d}\n")
        f.write("!UNIT_CELL_CONSTANTS=" + "".join(f"{x:10.3f}" for x in cell) + "\n")
        for i, s in enumerate(sets):
            f.write(f"!ISET= {i + 1} INPUT_FILE={s.path}\n")
        if out["merge"]:
            f.write("!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=5\n!ITEM_H=1\n!ITEM_K=2\n!ITEM_L=3\n!ITEM_IOBS=4\n"
                    "!ITEM_SIGMA(IOBS)=5\n!END_OF_HEADER\n")
            for h, v, s_ in zip(merged["hkl"], merged["I"], merged["sig"]):
                f.write(f"{h[0]:6d}{h[1]:6d}{h[2]:6d}{v:11.3E}{s_:11.3E}\n")
        else:
            f.write("!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=9\n!ITEM_H=1\n!ITEM_K=2\n!ITEM_L=3\n!ITEM_IOBS=4\n"
                    "!ITEM_SIGMA(IOBS)=5\n!ITEM_XD=6\n!ITEM_YD=7\n!ITEM_ZD=8\n!ITEM_ISET=9\n!END_OF_HEADER\n")
            for k in range(len(I)):
                f.write(f"{hkl[k, 0]:6d}{hkl[k, 1]:6d}{hkl[k, 2]:6d}{I[k]:11.3E}{sig[k]:11.3E}"
                        f"{xd[k]:8.1f}{yd[k]:8.1f}{zd[k]:9.1f}{iset[k] + 1:4d}\n")
        f.write("!END_OF_DATA\n")


def main(argv=None):
    """``python -m xpg.multiset [merge.toml]``; the command is ``xpg merge``."""
    from .cli import main as cli_main
    return cli_main(["merge"] + list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())

"""PDB deposition file: merged + unmerged reflections as mmCIF (xpg_deposit.cif), with the data-collection
statistics (_reflns / _reflns_shell) computed from the deposited unmerged data over the deposited resolution range."""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

from . import __version__
from .merging import merging_statistics
from .symops import d_spacing


def _num(x, fmt):
    return format(float(x), fmt) if x is not None and np.isfinite(float(x)) else "?"


def _fill_batch_gaps(mtz):
    """Empty batch headers for the frames missing inside one continuous rotation (frames removed by the filters):
    the converter starts a new _diffrn entry at every gap in the batch numbers. A gap where the rotation does not
    continue at the same oscillation (another sweep, or no rotation range in the headers) is kept. The oscillation is
    the mean width of the run of consecutive batches before the gap: the header phi values are float32, and one
    batch's width, multiplied by a long gap on a fine-sliced sweep, misses the rotation by more than the tolerance."""
    bs = sorted((b.clone() for b in mtz.batches), key=lambda b: b.number)    # copies: the vector is cleared below
    out, i0 = [], 0                                     # i0: first batch of the run of consecutive batches up to a
    for i, (a, b) in enumerate(zip(bs, bs[1:] + [None])):
        out.append(a)
        n = b.number - a.number if b is not None else 0
        if n == 1 and b.dataset_id == a.dataset_id:
            continue
        r, m, i0 = bs[i0], i - i0 + 1, i + 1
        osc = (a.floats[37] - r.floats[36]) / m                              # phi start / end
        if n < 2 or b.dataset_id != a.dataset_id or osc == 0:
            continue
        # each phi is within half a float32 spacing: osc within 1/m spacings, the phi predicted at b within 1 + n/m
        ulp = float(np.spacing(np.float32(max(abs(r.floats[36]), abs(a.floats[37]), abs(b.floats[36])))))
        if abs(b.floats[36] - a.floats[36] - n * osc) > 0.01 * abs(osc) + 2 * (1 + n / m) * ulp:
            continue
        for k in range(1, n):
            c = a.clone()
            c.number = a.number + k
            c.floats[36], c.floats[37] = a.floats[36] + k * osc, a.floats[37] + k * osc
            out.append(c)
    mtz.batches.clear()
    for b in out:
        mtz.batches.append(b)


def write_deposition_cif(wd, stem="xpg", log=print):
    wd = Path(wd)
    mfile, ufile = wd / f"{stem}_merged_iso.mtz", wd / f"{stem}_unmerged.mtz"
    if not (mfile.exists() and ufile.exists()):
        return None
    m, u = gemmi.read_mtz_file(str(mfile)), gemmi.read_mtz_file(str(ufile))
    _fill_batch_gaps(u)
    conv = gemmi.MtzToCif()
    conv.free_flag_value = 0
    text = conv.write_cif_to_string(m, u).replace("data_mtz", f"data_{stem}", 1)

    # statistics of the deposited unmerged observations in the deposited resolution range
    a = np.array(u, copy=False)
    lab = [c.label for c in u.columns]
    hkl = a[:, :3].astype(int)
    isym = a[:, lab.index("M/ISYM")].astype(int) % 256
    I, sig = a[:, lab.index("I")], a[:, lab.index("SIGI")]
    ok = np.isfinite(I) & (sig > 0)
    sg = u.spacegroup
    cell = np.array(u.cell.parameters)
    # from the merged reflections with the cell used below (the header value was computed with the unrounded cell
    # and can lie above the outermost reflection's d, which then dropped out of the statistics)
    dmin = float(d_spacing(np.array(m.make_miller_array()), cell).min())
    shells, ov, _ = merging_statistics(hkl[ok], (isym[ok] % 2) == 1, I[ok], sig[ok], cell, sg, nbins=10, dmin=dmin,
                                       anomalous=True)
    soft = ["", "loop_", "_software.pdbx_ordinal", "_software.name", "_software.version", "_software.classification",
            f"1 XPG {__version__} 'data reduction'", f"2 XPG {__version__} 'data scaling'"]
    refl = ["", "_reflns.entry_id xxxx", "_reflns.pdbx_ordinal 1", "_reflns.pdbx_diffrn_id 1",
            f"_reflns.d_resolution_low {_num(ov.dmax, '.3f')}", f"_reflns.d_resolution_high {_num(ov.dmin, '.3f')}",
            f"_reflns.number_obs {int(ov.n_uniq)}", f"_reflns.pdbx_number_measured_all {int(ov.n_obs)}",
            f"_reflns.percent_possible_obs {_num(100 * ov.completeness, '.1f')}",
            f"_reflns.pdbx_redundancy {_num(ov.multiplicity, '.2f')}",
            f"_reflns.pdbx_netI_over_sigmaI {_num(ov.i_over_sigma, '.2f')}",
            f"_reflns.pdbx_Rmerge_I_obs {_num(ov.rmerge, '.4f')}", f"_reflns.pdbx_Rrim_I_all {_num(ov.rmeas, '.4f')}",
            f"_reflns.pdbx_Rpim_I_all {_num(ov.rpim, '.4f')}", f"_reflns.pdbx_CC_half {_num(ov.cc_half, '.4f')}"]
    sh = ["", "loop_"] + [f"_reflns_shell.{k}" for k in (
        "pdbx_ordinal", "pdbx_diffrn_id", "d_res_low", "d_res_high", "number_measured_obs", "number_unique_obs",
        "percent_possible_all", "pdbx_redundancy", "meanI_over_sigI_obs", "Rmerge_I_obs", "pdbx_Rrim_I_all",
        "pdbx_Rpim_I_all", "pdbx_CC_half")]
    for i, s in enumerate(shells):
        sh.append(f"{i + 1} 1 {_num(s.dmax, '.3f')} {_num(s.dmin, '.3f')} {int(s.n_obs)} {int(s.n_uniq)} "
                  f"{_num(100 * s.completeness, '.1f')} {_num(s.multiplicity, '.2f')} {_num(s.i_over_sigma, '.2f')} "
                  f"{_num(s.rmerge, '.4f')} {_num(s.rmeas, '.4f')} {_num(s.rpim, '.4f')} {_num(s.cc_half, '.4f')}")
    # gemmi writes one block (merged _refln + unmerged _diffrn_refln): statistics before the first reflection loop
    k = text.find("\nloop_\n_refln.")
    ins = "\n".join(soft + refl + sh) + "\n"
    text = text[:k] + ins + text[k:] if k >= 0 else text + ins
    gemmi.cif.read_string(text)                       # must parse
    out = wd / f"{stem}_deposit.cif"
    out.write_text(text)
    log(f"  DEPOSITION: {out.name} (merged + unmerged reflections, statistics {ov.dmax:.2f}-{ov.dmin:.2f} A)")
    return out

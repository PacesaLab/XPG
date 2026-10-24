"""PDB deposition file: merged + unmerged reflections as mmCIF (xpg_deposit.cif), with the data-collection
statistics (_reflns / _reflns_shell) computed from the deposited unmerged data over the deposited resolution range."""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

from . import __version__
from .merging import merging_statistics


def _num(x, fmt):
    return format(float(x), fmt) if x is not None and np.isfinite(float(x)) else "?"


def write_deposition_cif(wd, stem="xpg", log=print):
    wd = Path(wd)
    mfile, ufile = wd / f"{stem}_merged_iso.mtz", wd / f"{stem}_unmerged.mtz"
    if not (mfile.exists() and ufile.exists()):
        return None
    m, u = gemmi.read_mtz_file(str(mfile)), gemmi.read_mtz_file(str(ufile))
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
    dmin = m.resolution_high()
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

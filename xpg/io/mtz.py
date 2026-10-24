"""MTZ output (unmerged multi-record and merged) via gemmi."""
from __future__ import annotations

import numpy as np
import gemmi


def write_unmerged_mtz(path, sg, cell, wavelength, hkl_asu, isym, batch, I, sig, xdet, ydet, rot, lp,
                       frac=None, batch_phi=None, dataset="xpg", crystal="xtal", project="proj"):
    m = gemmi.Mtz(with_base=True)
    m.spacegroup = sg if isinstance(sg, gemmi.SpaceGroup) else gemmi.SpaceGroup(sg)
    m.set_cell_for_all(gemmi.UnitCell(*cell))
    ds = m.add_dataset(dataset)
    ds.project_name, ds.crystal_name, ds.wavelength = project, crystal, wavelength
    for lab, t in [("M/ISYM", "Y"), ("BATCH", "B"), ("I", "J"), ("SIGI", "Q"), ("XDET", "R"), ("YDET", "R"),
                   ("ROT", "R"), ("FRACTIONCALC", "R"), ("LP", "R")]:
        m.add_column(lab, t)
    if frac is None:
        frac = np.ones(len(I))
    data = np.column_stack([hkl_asu.astype(float), isym, batch, I, sig, xdet, ydet, rot, frac, lp]).astype(np.float32)
    m.set_data(data)
    # batch headers
    ub = np.unique(batch.astype(int))
    for b in ub:
        bt = gemmi.Mtz.Batch()
        bt.number = int(b)
        bt.dataset_id = 1
        bt.cell = gemmi.UnitCell(*cell)
        bt.wavelength = wavelength
        if batch_phi is not None:
            p0, p1 = batch_phi(int(b))
            bt.floats[36] = p0
            bt.floats[37] = p1
        m.batches.append(bt)
    m.history = ["From XPG unmerged output"]
    m.sort()
    m.write_to_file(str(path))


def write_merged_mtz(path, sg, cell, wavelength, merged: dict, columns: list[tuple[str, str, str]],
                     dataset="xpg", crystal="xtal", project="proj", history=None):
    """``columns``: list of (label, mtz type, key in merged). Missing values are NaN."""
    m = gemmi.Mtz(with_base=True)
    m.spacegroup = sg if isinstance(sg, gemmi.SpaceGroup) else gemmi.SpaceGroup(sg)
    m.set_cell_for_all(gemmi.UnitCell(*cell))
    ds = m.add_dataset(dataset)
    ds.project_name, ds.crystal_name, ds.wavelength = project, crystal, wavelength
    cols = [merged["hkl"].astype(float)]
    for lab, t, key in columns:
        m.add_column(lab, t)
        cols.append(np.asarray(merged[key], float)[:, None])
    data = np.column_stack(cols).astype(np.float32)
    m.set_data(data)
    m.history = history or ["From XPG merged output"]
    m.update_reso()
    m.write_to_file(str(path))


def add_free_flags(hkl, sg, fraction=0.05, seed=1234, cell=None):
    """Free-R flags (0..19 style: flag 0 = test set), deterministic: a hash of the reflection's canonical index under
    the LATTICE symmetry (with ``cell``), so that reflections related by any possible merohedral twin law share a
    flag and the same reflection gets the same flag in every run."""
    import hashlib
    from ..symops import apply_rot, as_spacegroup, hkl_key, to_asu
    hkl = np.asarray(hkl, int)
    nflag = int(round(1.0 / fraction))
    sg = as_spacegroup(sg)
    ops = [np.eye(3, dtype=int)]
    if cell is not None:
        from ..analysis import ambiguity_operators
        ops = ambiguity_operators(cell, sg)
    keys = np.stack([hkl_key(to_asu(apply_rot(hkl, R), sg)[0]) for R in ops], axis=1).min(axis=1)
    h = np.array([int(hashlib.md5(f"{seed}:{int(k)}".encode()).hexdigest()[:8], 16) for k in keys])
    return (h % nflag).astype(float)

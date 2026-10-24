"""Matthews coefficient, solvent content and copy-number probabilities (from ../Matthews_Probability_Calculator.ipynb,
with corrections).

* Vm = V_asu / (n * MW), V_asu = V_cell / (number of symmetry operators incl. centring), from gemmi.
* Solvent fraction Vs = 1 - 1.23 / Vm (Matthews 1968; protein partial specific volume 0.74 cm^3/g).
  (The notebook used 1 - 1/Vm.)
* Probability of Vm given the resolution: Kantardjieff & Rupp (2003) extreme-value fits
  P(Vm) = y0 + A exp(-exp(-z) - z s + 1), z = (Vm - xc)/w, with parameters tabulated by resolution
  (the notebook looked them up with Vm instead of the resolution). Normalised over the copy numbers.
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

# Kantardjieff & Rupp (2003) parameters (columns = upper resolution limit of the bin, A); rows: y0, xc, w, A, s
_KR_RES = np.array([1.199, 1.501, 1.650, 1.801, 1.901, 2.001, 2.201, 2.401, 2.601, 2.801, 3.101, 3.501, 5.001])
_KR = np.array([
    [0.085, 0.312, 0.400, 0.503, 0.597, 0.729, 1.052, 1.781, 2.852, 3.386, 3.841, 4.281, 4.592],
    [2.052, 2.102, 2.122, 2.132, 2.140, 2.155, 2.171, 2.182, 2.191, 2.192, 2.205, 2.211, 2.210],
    [0.213, 0.214, 0.220, 0.225, 0.226, 0.231, 0.236, 0.241, 0.242, 0.244, 0.244, 0.244, 0.245],
    [28.38, 102.7, 187.5, 339.3, 434.1, 540.5, 686.2, 767.9, 835.9, 856.9, 854.0, 849.6, 846.7],
    [0.953, 0.807, 0.775, 0.702, 0.648, 0.640, 0.635, 0.589, 0.584, 0.542, 0.500, 0.485, 0.480],
])

_AA = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388, "E": 129.1155, "Q": 128.1307,
       "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766,
       "P": 97.1167, "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326, "U": 150.0388,
       "O": 237.3018, "X": 110.0}


def mw_from_sequence(seq: str) -> float:
    """Average molecular weight (Da) of a protein sequence (one-letter code; FASTA headers ignored)."""
    s = "".join(l.strip() for l in seq.splitlines() if not l.startswith(">")).upper().replace("*", "")
    return float(sum(_AA.get(c, 110.0) for c in s if c.isalpha()) + 18.015)


TYPES = ("protein", "rna", "dna", "ligand")
_NT = {"rna": 320.5, "dna": 308.9}   # average mass of a nucleotide in a strand (Da)


def read_components(path):
    """Content of the asymmetric unit (one copy) from a FASTA file: [(name, type, sequence)], type protein / rna / dna /
    ligand. The type is a word of the header (">B rna", ">L ligand"); without one, nucleic acids are recognised by
    their alphabet (ACGU with U / ACGT with T, >= 6 residues) and a "smiles:" or "ccd:" line marks a ligand. A ligand
    is a SMILES string or a CCD code ("smiles:CC(=O)O", "ccd:ATP" or just "ATP")."""
    comps, cur = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            words = line[1:].replace("|", " ").split()
            cur = [words[0] if words else f"chain{len(comps) + 1}",
                   next((w.lower() for w in words[1:] if w.lower() in TYPES), None), ""]
            comps.append(cur)
        elif line and cur is not None:
            cur[2] += line if cur[1] == "ligand" or line.lower().startswith(("smiles:", "ccd:")) else line.upper()
    out = []
    for name, t, seq in comps:
        seq = seq.replace("*", "")
        if t is None:
            letters = set(seq.upper())
            t = ("ligand" if seq.lower().startswith(("smiles:", "ccd:")) else
                 "rna" if len(seq) >= 6 and letters <= set("ACGU") and "U" in letters else
                 "dna" if len(seq) >= 6 and letters <= set("ACGT") and "T" in letters else "protein")
        out.append((name, t, seq))
    return out


def mw_of_components(comps):
    """Molecular weight (Da) of the components: proteins from their residues, nucleic acids from the average
    nucleotide mass; ligands are not counted (small next to the polymers)."""
    mw = 0.0
    for _, t, seq in comps:
        if t == "protein":
            mw += mw_from_sequence(seq)
        elif t in _NT:
            mw += _NT[t] * len(seq) + 18.015
    return mw


def model_components(model):
    """The polymer chains of a model as components (protein / rna / dna), for a model given without sequences."""
    st = gemmi.read_structure(str(model))
    st.setup_entities()
    kinds = {gemmi.PolymerType.PeptideL: "protein", gemmi.PolymerType.PeptideD: "protein",
             gemmi.PolymerType.Rna: "rna", gemmi.PolymerType.Dna: "dna"}
    out = []
    for ch in st[0]:
        poly = ch.get_polymer()
        t = kinds.get(poly.check_polymer_type())
        if t and len(poly):
            out.append((ch.name, t, poly.make_one_letter_sequence().replace("-", "")))
    return out


def write_components(comps, path):
    Path(path).write_text("".join(f">{n} {t}\n{s}\n" for n, t, s in comps))
    return Path(path)


def read_fasta(path):
    """{name: sequence} from a FASTA file."""
    out, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0] if len(line) > 1 else f"chain{len(out) + 1}"
            out[name] = ""
        elif line and name is not None:
            out[name] += line
    return out


def kr_probability(vm, resolution):
    """Kantardjieff-Rupp density of Vm at the given resolution."""
    i = min(int(np.searchsorted(_KR_RES, resolution)), _KR.shape[1] - 1)
    y0, xc, w, a, s = _KR[:, i]
    z = (np.asarray(vm, float) - xc) / w
    return np.maximum(y0 + a * np.exp(-np.exp(-z) - z * s + 1.0), 0.0)


def matthews_table(cell, space_group, mw, resolution=2.5, max_copies=24):
    """Rows (n, Vm, solvent %, probability) for plausible copy numbers (10 % < solvent < 90 %)."""
    sg = gemmi.SpaceGroup(space_group) if isinstance(space_group, str) else space_group
    vol = gemmi.UnitCell(*cell).volume
    nops = len(sg.operations())
    vasu = vol / nops
    rows = []
    for n in range(1, max_copies + 1):
        vm = vasu / (n * mw)
        vs = 1.0 - 1.23 / vm
        if 0.1 < vs < 0.9:
            rows.append(dict(n=n, vm=float(vm), solvent=float(100 * vs), p=float(kr_probability(vm, resolution))))
    tot = sum(r["p"] for r in rows) or 1.0
    for r in rows:
        r["p"] /= tot
    return rows



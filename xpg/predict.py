"""Predicted search model for `xpg mr` when no model is given: OpenFold3, run in its own environment (XPG_OPENFOLD3 =
the run_openfold executable, else <source>/.venv-openfold3 from `install.sh --openfold3`, else run_openfold on the
PATH). It runs on CUDA GPUs, the Apple GPU and the CPU. With an MSA the sequence is sent to the ColabFold MSA server.

The whole content of the sequence file is predicted as one complex (protein, RNA and DNA chains and ligands; see
matthews.read_components), which MR then places as one search model, copy after copy (the number of copies comes
from the Matthews table). The best of five samples by OpenFold3's ranking score is written as PDB with pLDDT (0-100)
in the B column, without ligands and waters; `xpg mr` trims it to pLDDT >= 70 and converts pLDDT to B factors.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def openfold3_command():
    side = Path(__file__).resolve().parents[1] / ".venv-openfold3" / "bin" / "run_openfold"
    return os.environ.get("XPG_OPENFOLD3") or (str(side) if side.exists() else shutil.which("run_openfold"))


def _chain_id(i):
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i % 26] * (1 + i // 26)


def query_chains(comps):
    """OpenFold3 query chains for the components (name, type, sequence)."""
    out = []
    for i, (name, t, seq) in enumerate(comps):
        c = {"molecule_type": t, "chain_ids": [_chain_id(i)], "description": name}
        if t != "ligand":
            c["sequence"] = seq.upper()
        elif seq.lower().startswith("smiles:"):
            c["smiles"] = seq[len("smiles:"):].strip()
        else:
            c["ccd_codes"] = [seq[len("ccd:"):].strip().upper() if seq.lower().startswith("ccd:") else seq.strip().upper()]
        out.append(c)
    return out


def predict_model(fasta, out_dir, msa=True, log=print):
    """OpenFold3 model of the content of `fasta` -> out_dir/predicted.pdb (pLDDT in B); returns the path."""
    from .matthews import read_components
    cmd = openfold3_command()
    if not cmd:
        raise RuntimeError("no structure predictor: install OpenFold3 (install.sh --openfold3) or give --model")
    comps = read_components(fasta)
    if not comps:
        raise ValueError(f"{fasta}: no sequence")
    wd = Path(out_dir)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "query.json").write_text(json.dumps({"seeds": [42], "queries": {"model": {
        "chains": query_chains(comps), "use_msas": bool(msa)}}}, indent=1))
    log("MR: predicting the search model with OpenFold3: " + ", ".join(
        f"{n} ({t}{'' if t == 'ligand' else f', {len(q)}'})" for n, t, q in comps) +
        f"; {'MSA from the ColabFold server' if msa else 'single sequence'}")
    with open(wd / "openfold3.log", "w") as f:
        r = subprocess.run([cmd, "predict", "--query_json", str(wd / "query.json"), "--output_dir", str(wd / "out"),
                            "--use_msa_server", "true" if msa else "false"], stdout=f, stderr=subprocess.STDOUT)
    samples = []
    for c in (wd / "out").glob("model/seed_*/*_confidences_aggregated.json"):
        conf = json.loads(c.read_text())
        model = c.with_name(c.name.replace("_confidences_aggregated.json", "_model.cif"))
        if model.exists():
            samples.append((conf.get("sample_ranking_score", 0.0), conf.get("avg_plddt", 0.0), model))
    if r.returncode != 0 or not samples:
        raise RuntimeError(f"OpenFold3 failed, see {wd / 'openfold3.log'}")
    score, plddt, best = max(samples, key=lambda t: (t[0], t[1]))
    log(f"MR: OpenFold3 model {best.name}: ranking score {score:.3f}, mean pLDDT {plddt:.1f}")
    import gemmi
    st = gemmi.read_structure(str(best))
    st.setup_entities()
    st.remove_ligands_and_waters()                     # the search model: the polymer chains
    st.write_pdb(str(wd / "predicted.pdb"))
    return wd / "predicted.pdb"

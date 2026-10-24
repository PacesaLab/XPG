"""Command-line interface: ``xpg [xpg.toml] [options]``, ``xpg merge [input]``, ``xpg convert XDS.INP``, ``xpg serve [run]``,
``xpg mr RUN [--sequence S] [--model M]``, ``xpg report [RUN]``.

Processing reads an xpg.toml (see xpg/inputs.py); an XDS.INP is converted on reading and the equivalent xpg.toml
is kept with the run. ``xpg merge`` scales and merges several unmerged data sets (unmerged MTZ + geometry JSON, or
legacy-format files; a legacy merge input file is recognised by its INPUT_FILE= lines). ``xpg serve [RUN_DIR]``:
live dashboard of a run (events.jsonl written by every run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _is_xscale_inp(path: Path) -> bool:
    if path.name.upper().startswith("XSCALE"):
        return True
    try:
        text = path.read_text(errors="replace").upper()
    except OSError:
        return False
    return "INPUT_FILE=" in text and "NAME_TEMPLATE_OF_DATA_FRAMES=" not in text


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].lower() == "serve":          # live dashboard of a run directory
        sp = argparse.ArgumentParser(prog="xpg serve", description="live monitor of a processing run (events.jsonl)")
        sp.add_argument("run_dir", nargs="?", default=".", help="run / output directory (default: .)")
        sp.add_argument("--port", type=int, default=8765)
        sp.add_argument("--host", default="127.0.0.1", help="bind address (default localhost; use an SSH tunnel remotely)")
        sa = sp.parse_args(argv[1:])
        from .serve import serve
        serve(sa.run_dir, sa.port, sa.host)
        return 0
    if argv and argv[0].lower() == "mr":             # molecular replacement on a finished run
        mp = argparse.ArgumentParser(prog="xpg mr", description="molecular replacement on a finished XPG run")
        mp.add_argument("run_dir", nargs="?", default=".", help="run / output directory (default: .)")
        mp.add_argument("--model", default=None, help="search model (PDB / mmCIF; predicted models are trimmed); "
                        "default: an OpenFold3 prediction of the --sequence content")
        mp.add_argument("--sequence", default=None,
                        help="FASTA with one copy of the asymmetric-unit content: protein, RNA, DNA chains (type in the "
                             "header, e.g. '>B rna', else recognised) and ligands ('>L ligand' with a SMILES string or "
                             "a CCD code); optional with --model (then read from the model)")
        mp.add_argument("--copies", type=int, default=None, help="copies to search (default: Matthews +- 1)")
        mp.add_argument("--alternatives", default=None,
                        help="extra space groups to merge in and search, comma separated (default: the alternatives "
                             "of a close-call point-group decision)")
        mp.add_argument("--no-msa", action="store_true",
                        help="OpenFold3 prediction from the single sequence (default: MSA from the ColabFold server, "
                             "which receives the sequence)")
        ma = mp.parse_args(argv[1:])
        from .mr import run_mr
        from .pipeline import Logger
        run_mr(ma.run_dir, ma.model, ma.sequence, copies=ma.copies, alternatives=ma.alternatives,
               log=Logger(Path(ma.run_dir) / "xpg_mr.log"), msa=not ma.no_msa)
        return 0
    if argv and argv[0].lower() == "report":         # rewrite the PDF summary of a run
        rp = argparse.ArgumentParser(prog="xpg report", description="write RUN_DIR/xpg_report.pdf (processing and MR "
                                     "summary on two pages)")
        rp.add_argument("run_dir", nargs="?", default=".")
        from .pdf_report import write_pdf_report
        return 0 if write_pdf_report(rp.parse_args(argv[1:]).run_dir) else 1
    if argv and argv[0].lower() == "convert":        # another program's input -> xpg.toml
        cp = argparse.ArgumentParser(prog="xpg convert", description="convert an XDS.INP into an xpg.toml")
        cp.add_argument("inp")
        cp.add_argument("-o", "--out", default=None, help="output file (default: xpg.toml next to the input)")
        cp.add_argument("--all", action="store_true", help="write every setting, not only those differing from defaults")
        ca = cp.parse_args(argv[1:])
        from .inputs import params_to_toml, read_input
        out = Path(ca.out or Path(ca.inp).with_name("xpg.toml"))
        out.write_text(params_to_toml(read_input(ca.inp), all_fields=ca.all))
        print(f"xpg: wrote {out}")
        return 0
    forced = bool(argv) and argv[0].lower() == "merge"
    if forced:
        argv = argv[1:]
    ap = argparse.ArgumentParser(prog="xpg", description="XPG: X-tal data Processing on GPU (rotation data: spots, "
                                 "indexing, integration, scaling, merging)")
    ap.add_argument("inp", nargs="?", default=None,
                    help="xpg.toml (default; an XDS.INP is converted on reading), or after 'merge' a merge input")
    ap.add_argument("-o", "--outdir", default=None, help="output directory (default: directory of the input file)")
    ap.add_argument("--jobs", nargs="*", default=None,
                    help="stages to run: DETECTOR BACKGROUND SPOTS INDEX MASK INTEGRATE SCALE (or ALL); resume after an "
                         "interruption with the remaining stages (e.g. INTEGRATE SCALE), or PASS2 to redo only the second "
                         "pass from the pass-1 SCALE checkpoint")
    ap.add_argument("--single-pass", action="store_true", help="skip the second integration pass")
    ap.add_argument("--pass2", choices=("on", "off", "auto"), default=None,
                    help="second integration pass: auto (default: skipped when the pass-1 predictions of strong "
                         "reflections are already accurate and the data have one lattice), on, off (= --single-pass)")
    ap.add_argument("--no-aniso", action="store_true", help="skip anisotropy analysis")
    ap.add_argument("--allow-centrosymmetric", action="store_true", help="small molecules: consider all space groups")
    ap.add_argument("--images", default=None, help="image template or HDF5 master: writes an xpg.toml from the headers "
                                                   "and processes")
    ap.add_argument("--export-legacy", action="store_true",
                    help="also write the legacy-format text files of the reflections and geometry")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is present")
    ap.add_argument("--reject-nonisomorphous", action="store_true",
                    help="merge: drop data sets with negative / outlying Delta-CC1/2 and re-scale")
    a = ap.parse_args(argv)
    if a.cpu:
        import os
        os.environ["XPG_DEVICE"] = "cpu"
    if a.inp:
        inp = Path(a.inp)
    elif forced:
        inp = Path("merge.toml") if Path("merge.toml").exists() else Path("XSCALE.INP")
    else:
        inp = Path("xpg.toml") if Path("xpg.toml").exists() else Path("XDS.INP")
    is_merge_toml = str(inp).lower().endswith(".toml") and inp.exists() and "[merge" in inp.read_text()
    if not a.images and (forced or is_merge_toml or (not str(inp).lower().endswith(".toml") and _is_xscale_inp(inp))):
        from .pipeline import Logger
        from .multiset import run_multiset
        wd = Path(a.outdir or inp.parent)
        wd.mkdir(parents=True, exist_ok=True)
        text, rej = None, a.reject_nonisomorphous
        if str(inp).lower().endswith(".toml"):
            from .inputs import read_merge_toml
            text, rej_t = read_merge_toml(inp)
            rej = rej or rej_t
        run_multiset(inp, wd, log=Logger(wd / "xpg_merge.log"), anisotropy=not a.no_aniso,
                   reject_nonisomorphous=rej, export_legacy=a.export_legacy, inp_text=text)
        return 0
    from .inputs import is_toml, params_to_toml, read_input
    from .pipeline import Pipeline
    if a.images:
        from .setup import params_from_images
        out = Path(a.outdir or ".")
        out.mkdir(parents=True, exist_ok=True)
        p = params_from_images(a.images, out)
        if p.extra.get("XPG_GEOMETRY_NOTE"):
            print(f"xpg: header geometry: {p.extra['XPG_GEOMETRY_NOTE']}")
        (out / "xpg.toml").write_text(params_to_toml(p))
        inp = out / "xpg.toml"
    p = read_input(inp)
    if a.export_legacy:
        p.export_legacy = True
    wd = Path(a.outdir or inp.parent)
    if not is_toml(inp):              # converted input: keep the equivalent xpg.toml with the run
        wd.mkdir(parents=True, exist_ok=True)
        tgt = wd / "xpg.toml"
        if not tgt.exists():
            tgt.write_text(params_to_toml(p))
            print(f"xpg: {inp.name} converted to {tgt}")
    # default auto
    sp = (False if a.single_pass else "auto") if a.pass2 is None else {"on": True, "off": False, "auto": "auto"}[a.pass2]
    pipe = Pipeline(p, workdir=wd, second_pass=sp, anisotropy=not a.no_aniso,
                    chiral=not a.allow_centrosymmetric)
    try:
        pipe.run(a.jobs)
    except RuntimeError as e:          # e.g. no lattice found (salt / no diffraction): report, no traceback
        print(f"xpg: !!! ERROR !!! {e}", file=sys.stderr)
        pipe.log(f"!!! ERROR !!! {e}")
        # resource / library failures (CUDA out of memory, ...) are not data verdicts: keep where they happened
        if not str(e).startswith(("INDEX:", "indexing failed")) or "out of memory" in str(e):
            import traceback
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            pipe.log(tb)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

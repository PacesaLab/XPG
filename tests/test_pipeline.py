"""Fast end-to-end test on a small simulated dataset."""
import json

import numpy as np
import pytest

from xpg.inputs import read_input
from xpg.pipeline import Logger, Pipeline
from xpg.simulate import SimSpec, simulate


@pytest.mark.parametrize("fmt", ["cbf", "eiger"])
def test_end_to_end(tmp_path, fmt):
    spec = SimSpec(cell=(41.3, 52.7, 60.1, 90, 90, 90), space_group="P 21 21 21", nx=512, ny=540, orgx=255.3,
                   orgy=270.6, distance=90.0, osc=0.5, nframes=180, dmin=1.9, gaps=False, mean_i0=8000.0, seed=11)
    simulate(tmp_path, spec, verbose=False, fmt=fmt)
    p = read_input(tmp_path / "xpg.toml")
    st = Pipeline(p, workdir=tmp_path / "run", log=Logger(echo=False), second_pass=False).run()
    # 00l lies in the blind region of this geometry: P 21 21 2 / P 21 21 21 are legitimately tied
    assert "P 21 21 21" in st.results["space_group_alternatives"]
    assert st.space_group.point_group_hm() == "222"
    assert np.allclose(st.exp.crystal.cell[:3], spec.cell[:3], rtol=2e-3)
    ov = st.results["overall"]
    assert ov.cc_half > 0.99 and ov.completeness > 0.6
    summary = json.loads((tmp_path / "run" / "xpg_summary.json").read_text())
    assert "P 21 21 21" in summary["space_group_alternatives"]
    run = tmp_path / "run"
    for f in ("xpg_unmerged.mtz", "xpg_geometry.json", "xpg_merged_iso.mtz", "xpg_spots.txt", "xpg_geometry_indexed.json"):
        assert (run / f).exists(), f
    # legacy-format files only on request
    for f in ("XDS_ASCII.HKL", "INTEGRATE.HKL", "GXPARM.XDS", "SPOT.XDS", "XPARM.XDS"):
        assert not (run / f).exists(), f
    # the unmerged MTZ + geometry is self-contained: the merge step reads it back
    from xpg.multiset import read_input_set
    s = read_input_set(run / "xpg_unmerged.mtz")
    assert s.geom is not None and len(s.I) > 1000 and np.isfinite(s.zd).all()
    assert abs(s.zd.max() - spec.nframes) < 2


def test_export_legacy(tmp_path):
    spec = SimSpec(cell=(41.3, 52.7, 60.1, 90, 90, 90), space_group="P 21 21 21", nx=512, ny=540, orgx=255.3,
                   orgy=270.6, distance=90.0, osc=0.5, nframes=60, dmin=2.2, gaps=False, mean_i0=8000.0, seed=12)
    simulate(tmp_path, spec, verbose=False)
    p = read_input(tmp_path / "xpg.toml")
    p.export_legacy = True
    Pipeline(p, workdir=tmp_path / "run", log=Logger(echo=False), second_pass=False).run()
    for f in ("XDS_ASCII.HKL", "INTEGRATE.HKL", "GXPARM.XDS", "SPOT.XDS", "XPARM.XDS"):
        assert (tmp_path / "run" / f).exists(), f


def test_toml_input_cli(tmp_path):
    """xpg.toml input: convert an XDS.INP, run the command line from the TOML."""
    from xpg.cli import main
    spec = SimSpec(cell=(41.3, 52.7, 60.1, 90, 90, 90), space_group="P 21 21 21", nx=512, ny=540, orgx=255.3,
                   orgy=270.6, distance=90.0, osc=0.5, nframes=60, dmin=2.2, gaps=False, mean_i0=8000.0, seed=13)
    simulate(tmp_path, spec, verbose=False)
    assert main(["convert", str(tmp_path / "XDS.INP")]) == 0
    toml = (tmp_path / "xpg.toml").read_text()
    assert "[data]" in toml and "images" in toml and "NAME_TEMPLATE" not in toml
    assert main([str(tmp_path / "xpg.toml"), "-o", str(tmp_path / "run"), "--single-pass"]) == 0
    assert (tmp_path / "run" / "xpg_unmerged.mtz").exists()

    # merge.toml: two copies of the unmerged MTZ as two data sets
    import shutil
    run = tmp_path / "run"
    for k in (1, 2):
        shutil.copy(run / "xpg_unmerged.mtz", tmp_path / f"set{k}_unmerged.mtz")
        shutil.copy(run / "xpg_geometry.json", tmp_path / f"set{k}_geometry.json")
    (tmp_path / "merge.toml").write_text('[merge]\noutput = "both"\n[[merge.input]]\nfile = "set1_unmerged.mtz"\n'
                                        'reference = true\n[[merge.input]]\nfile = "set2_unmerged.mtz"\n')
    assert main([str(tmp_path / "merge.toml"), "-o", str(tmp_path / "m")]) == 0
    assert (tmp_path / "m" / "both_unmerged.mtz").exists() and (tmp_path / "m" / "both_merged_iso.mtz").exists()
    assert not (tmp_path / "m" / "both").exists()          # no legacy-format merge output by default


def test_gather_process_identical(tmp_path, monkeypatch):
    """XPG_GATHER_PROCESS=1 (frame loop in a forked process, shared-memory shoeboxes) integrates bit-identically."""
    import pickle
    import shutil
    from xpg.shoebox import _gather_process_ok
    monkeypatch.setenv("XPG_GATHER_PROCESS", "1")
    if not _gather_process_ok():
        pytest.skip("no fork / descriptor passing on this platform")
    spec = SimSpec(cell=(41.3, 52.7, 60.1, 90, 90, 90), space_group="P 21 21 21", nx=512, ny=540, orgx=255.3,
                   orgy=270.6, distance=90.0, osc=0.5, nframes=120, dmin=2.2, gaps=False, mean_i0=8000.0, seed=14)
    simulate(tmp_path, spec, verbose=False)
    p = read_input(tmp_path / "xpg.toml")
    refl = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("XPG_GATHER_PROCESS", mode)
        run = tmp_path / f"run{mode}"
        if mode == "1":                       # INTEGRATE alone from the thread run's early checkpoints
            run.mkdir()
            for f in (tmp_path / "run0").glob("state_*.pkl"):
                if f.name != "state_INTEGRATE.pkl":
                    shutil.copy(f, run / f.name)
        jobs = ["INTEGRATE"] if mode == "1" else ["XYCORR", "INIT", "COLSPOT", "IDXREF", "DEFPIX", "INTEGRATE"]
        Pipeline(p, workdir=run, log=Logger(echo=False), second_pass=False).run(jobs)
        with open(run / "state_INTEGRATE.pkl", "rb") as fh:
            refl[mode] = pickle.load(fh)["refl"]
    for k in ("isum", "vsum", "iprf", "vprf", "bg", "xo", "yo", "zo", "n_over"):
        assert np.array_equal(np.asarray(refl["0"][k]), np.asarray(refl["1"][k]), equal_nan=True), k


def test_integration_memory_passes(tmp_path, monkeypatch):
    """A host-memory budget below one pass over the frames splits the blocks over several passes
    (XPG_INTEGRATE_MEM_GB): every reflection is still integrated once, close to the one-pass result (only the
    adaptive sigma_D / sigma_M restart per pass)."""
    import pickle
    import shutil
    spec = SimSpec(cell=(41.3, 52.7, 60.1, 90, 90, 90), space_group="P 21 21 21", nx=512, ny=540, orgx=255.3,
                   orgy=270.6, distance=90.0, osc=0.5, nframes=80, dmin=2.2, gaps=False, mean_i0=8000.0, seed=15)
    simulate(tmp_path, spec, verbose=False)
    p = read_input(tmp_path / "xpg.toml")
    refl = {}
    for mode in ("one", "split"):
        run = tmp_path / mode
        if mode == "split":
            monkeypatch.setenv("XPG_INTEGRATE_MEM_GB", "0.5")         # below the fixed kernel reserve: always split
            run.mkdir()
            for f in (tmp_path / "one").glob("state_*.pkl"):
                if f.name != "state_INTEGRATE.pkl":
                    shutil.copy(f, run / f.name)
        jobs = ["INTEGRATE"] if mode == "split" else ["XYCORR", "INIT", "COLSPOT", "IDXREF", "DEFPIX", "INTEGRATE"]
        run.mkdir(exist_ok=True)
        Pipeline(p, workdir=run, log=Logger(run / "test.log", echo=False), second_pass=False).run(jobs)
        with open(run / "state_INTEGRATE.pkl", "rb") as fh:
            refl[mode] = pickle.load(fh)["refl"]
    log = (tmp_path / "split" / "test.log").read_text()
    assert "INTEGRATE: pass 2/" in log
    a, b = refl["one"], refl["split"]
    assert np.array_equal(np.asarray(a["hkl"]), np.asarray(b["hkl"]))
    ok = np.isfinite(a["iprf"]) & np.isfinite(b["iprf"])
    assert ok.mean() > 0.95 and np.array_equal(np.isfinite(a["iprf"]), np.isfinite(b["iprf"]))
    d = np.abs(np.asarray(a["iprf"])[ok] - np.asarray(b["iprf"])[ok]) / np.sqrt(np.asarray(a["vprf"])[ok])
    assert np.median(d) < 0.1

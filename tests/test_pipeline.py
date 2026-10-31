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


def test_pass2_widths_from_block_fits(tmp_path, monkeypatch):
    """Pass 1 without a sweep-wide spot-width estimate (crowded data: no isolated strong spot) integrates with the
    default widths; the second pass is then always run (--pass2 auto) and starts from the median of pass 1's
    per-block fits, not from the defaults again."""
    import pickle
    from xpg import stage_integrate
    spec = SimSpec(cell=(41.3, 52.7, 60.1, 90, 90, 90), space_group="P 21 21 21", nx=512, ny=540, orgx=255.3,
                   orgy=270.6, distance=90.0, osc=0.5, nframes=80, dmin=2.2, gaps=False, mean_i0=8000.0, seed=16,
                   sigma_d=0.04, sigma_m=0.18)
    simulate(tmp_path, spec, verbose=False)
    p = read_input(tmp_path / "xpg.toml")

    def no_isolated_spots(images, exp, model, *a, **k):      # what the estimator returns when no spot is usable
        model.history = []
        return model
    monkeypatch.setattr(stage_integrate, "estimate_sigmas", no_isolated_spots)
    run = tmp_path / "run"
    run.mkdir()
    st = Pipeline(p, workdir=run, log=Logger(run / "test.log", echo=False), second_pass="auto").run()
    with open(run / "state_SCALE.pkl", "rb") as fh:             # pass 1
        bp1 = pickle.load(fh)["results"]["integration_block_params"]
    fits = np.array([(v["sigma_d_fit"], v["sigma_m_fit"]) for v in bp1.values() if "sigma_m_fit" in v])
    assert len(fits) >= 3
    assert np.allclose([st.model.sigma_d, st.model.sigma_m], np.median(fits, axis=0))
    log = (run / "test.log").read_text()
    assert "pass 1 used the default spot widths" in log and "spot widths from the pass-1 per-block fits" in log
    assert abs(st.model.sigma_m - spec.sigma_m) < 0.5 * abs(0.25 - spec.sigma_m)
    assert abs(st.model.sigma_d - spec.sigma_d) < 0.5 * abs(0.05 - spec.sigma_d)
    # a SCALE-only rerun starts from the pass-2 checkpoint: its fits are not taken for pass 1's, so the second
    # pass is not forced again, and if it runs it keeps the widths of the first run's second pass
    with open(run / "state_INTEGRATE.pkl", "rb") as fh:
        assert pickle.load(fh)["results"]["pass1_default_widths"] is False
    st2 = Pipeline(p, workdir=run, log=Logger(run / "rerun.log", echo=False), second_pass="auto").run(["SCALE"])
    log2 = (run / "rerun.log").read_text()
    assert "pass 1 used the default spot widths" not in log2 and "spot widths from the pass-1 per-block fits" not in log2
    assert (st2.model.sigma_d, st2.model.sigma_m) == (st.model.sigma_d, st.model.sigma_m)


def test_point_group_redecided_end_to_end(tmp_path, monkeypatch):
    """Two-pass run whose pass-1 point-group call is forced to a close mP 2 (posterior 0.3) on P 21 21 21 data: the
    final SCALE decides again on the re-integrated observations as integrated (the kind of input of the pass-1 call,
    not scaled in mP 2), gives oP 222, selects and scales in that group, and the outputs describe it -- unmerged and
    merged MTZ, xpg_summary.json, the cell, one record per decision (the earlier call in the note), the report. The
    pass-2 integration calls the scorer no more. SCALE-only reruns from the pass-2 checkpoint: the real scorer
    re-decides the same way; a second call confirming the current group leaves every output bit-identical to a run
    without it (one log line more)."""
    import dataclasses
    import gemmi
    import shutil
    from xpg.stage_scale import ScaleStage, _rotations_in_basis
    spec = SimSpec(cell=(41.3, 52.7, 60.1, 90, 90, 90), space_group="P 21 21 21", nx=512, ny=540, orgx=255.3,
                   orgy=270.6, distance=90.0, osc=0.5, nframes=60, dmin=2.2, gaps=False, mean_i0=8000.0, seed=17)
    simulate(tmp_path, spec, verbose=False)
    p = read_input(tmp_path / "xpg.toml")
    score, calls, seen = ScaleStage._score_point_group, [], []

    def close_mP2_first(self, r, cands, s, log):
        res = score(self, r, cands, s, log)
        calls.append(res)
        seen.append(np.asarray(r["I"]).copy())
        if len(calls) > 1:
            return res
        mp = dataclasses.replace(next(x for x in res[2] if x.cand.bravais == "mP"), prob=0.3)
        return (mp.cand, mp) + res[2:]
    monkeypatch.setattr(ScaleStage, "_score_point_group", close_mP2_first)
    run = tmp_path / "run"
    run.mkdir()
    st = Pipeline(p, workdir=run, log=Logger(run / "test.log", echo=False), second_pass=True).run()
    assert len(calls) == 2 and calls[1][0].point_group == "222"
    assert np.isin(seen[1], st.refl["I"]).all()             # pass-2 intensities as integrated, not scaled ones
    assert st.lattice.bravais == "oP" and st.space_group.point_group_hm() == "222" and st.exp.crystal.system == "o"
    assert np.allclose(np.sort(st.exp.crystal.cell[:3]), spec.cell[:3], rtol=2e-3)
    for f in ("xpg_unmerged.mtz", "xpg_merged_iso.mtz"):
        m = gemmi.read_mtz_file(str(run / f))
        assert m.spacegroup.xhm() == st.space_group.xhm() and np.allclose(m.cell.parameters, st.exp.crystal.cell,
                                                                            atol=1e-3), f
    summary = json.loads((run / "xpg_summary.json").read_text())
    assert summary["space_group"] == st.space_group.xhm() and np.allclose(summary["cell"], st.exp.crystal.cell)
    assert "audit_merges" not in summary                     # the final call is a confident one
    pg = [d for d in st.results["decisions"] if d["name"] == "point group"]
    assert len(pg) == 1 and pg[0]["choice"] == "oP 222" and not pg[0]["low_margin"]
    assert pg[0]["note"] == "re-decided on the re-integrated data (earlier call mP 2, posterior 0.30)"
    assert [d["choice"] for d in st.results["decisions"] if d["name"] == "space group"] == [st.space_group.xhm()]
    log = (run / "test.log").read_text()
    assert log.count("observations selected again in the new point group") == 1
    assert log.count("=== SCALE (final) ===") == 1
    assert "| point group | oP 222 (re-decided on the re-integrated data" in (run / "xpg_summary.md").read_text()
    # SCALE-only reruns from the pass-2 integration (pass-1 group and record): real scorer; second call confirming
    # the current group; no second call
    cur = []

    def confirm(self, r, cands, s, log):
        res = score(self, r, cands, s, log)
        same = next(x for x in res[2] if _rotations_in_basis(x.cand.P, x.cand.ops) ==
                    _rotations_in_basis(self.st.lattice.P, self.st.lattice.ops))
        cur.append(same.cand.bravais)
        return (same.cand, same) + res[2:]
    out = {}
    for mode, patch in (("real", score), ("same", confirm), ("off", None)):
        d = tmp_path / mode
        d.mkdir()
        shutil.copy(run / "state_INTEGRATE.pkl", d)
        monkeypatch.setattr(ScaleStage, "_score_point_group", patch or score)
        if patch is None:
            monkeypatch.setattr(ScaleStage, "_redecide_point_group", lambda self, *a: False)
        out[mode] = Pipeline(p, workdir=d, log=Logger(d / "test.log", echo=False), second_pass=False).run(["SCALE"])
    assert out["real"].space_group.xhm() == st.space_group.xhm() and cur == ["mP"]
    assert out["same"].space_group.point_group_hm() == "2"
    a, b = tmp_path / "same", tmp_path / "off"
    assert (a / "xpg_summary.json").read_text() == (b / "xpg_summary.json").read_text()
    mtz = sorted(f.name for f in b.glob("*.mtz"))
    assert "xpg_unmerged.mtz" in mtz and mtz == sorted(f.name for f in a.glob("*.mtz"))
    for f in mtz:
        assert np.array_equal(np.array(gemmi.read_mtz_file(str(a / f)), copy=False),
                              np.array(gemmi.read_mtz_file(str(b / f)), copy=False), equal_nan=True), f
    la, lb = (a / "test.log").read_text().splitlines(), (b / "test.log").read_text().splitlines()
    assert len(la) == len(lb) + 1 and sum("SCALE: point group mP 2 kept" in ln for ln in la) == 1


def test_point_group_too_high_close_call_redecided(tmp_path, monkeypatch):
    """A close pass-1 call to a group that is too high -- tP 422 forced at posterior 0.6 on P 21 21 21 data of a
    pseudo-tetragonal cell (50.0 x 50.05 x 60.1 A) with an NCS 2-fold k,h,-l -- is corrected on the re-integrated
    data: the final SCALE scores its observations as integrated, on which the NCS element decays with resolution
    (CC well below its expected value), and the run ends in 222. Scored after scaling in 422 instead, the outlier
    rejection and error model fitted to the 422 mates remove that evidence and keep the wrong group."""
    import dataclasses
    from xpg.stage_scale import ScaleStage
    spec = SimSpec(cell=(50.0, 50.05, 60.1, 90, 90, 90), space_group="P 21 21 21", ncs_op="k,h,-l", ncs_c=0.3,
                   ncs_b=20.0, nx=512, ny=540, orgx=255.3, orgy=270.6, distance=90.0, osc=0.5, nframes=90, dmin=2.2,
                   gaps=False, mean_i0=8000.0, seed=17)
    simulate(tmp_path, spec, verbose=False)
    p = read_input(tmp_path / "xpg.toml")
    score, calls = ScaleStage._score_point_group, []

    def tP422_first(self, r, cands, s, log):
        res = score(self, r, cands, s, log)
        calls.append(res)
        if len(calls) > 1:
            return res
        tp = dataclasses.replace(next(x for x in res[2] if (x.cand.bravais, x.cand.point_group) == ("tP", "422")),
                                 prob=0.6)
        return (tp.cand, tp) + res[2:]
    monkeypatch.setattr(ScaleStage, "_score_point_group", tP422_first)
    run = tmp_path / "run"
    run.mkdir()
    st = Pipeline(p, workdir=run, log=Logger(run / "test.log", echo=False), second_pass=True).run()
    assert len(calls) == 2 and (calls[1][0].bravais, calls[1][0].point_group) == ("oP", "222") and calls[1][1].prob > 0.9
    assert st.lattice.bravais == "oP" and st.space_group.point_group_hm() == "222" and st.exp.crystal.system == "o"
    pg = [d for d in st.results["decisions"] if d["name"] == "point group"]
    assert len(pg) == 1 and pg[0]["choice"] == "oP 222" and "earlier call tP 422" in pg[0]["note"]

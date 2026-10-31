import numpy as np
import gemmi

from xpg.multiset import parse_xscale_inp, cc_half_sigma_tau, zero_dose_merge
from xpg.symmetry import _robust_axial_merge
from xpg.spotcluster import _split_merged
from xpg.analysis import detwin
from xpg.geometry import mu_linear
from xpg.symops import to_asu


def test_parse_xscale_inp():
    g, outs = parse_xscale_inp("""SPACE_GROUP_NUMBER=19
OUTPUT_FILE=a.ahkl  FRIEDEL'S_LAW=FALSE
INPUT_FILE=*x1/XDS_ASCII.HKL
CRYSTAL_NAME=c1
INPUT_FILE=x2/XDS_ASCII.HKL
INCLUDE_RESOLUTION_RANGE= 30 2.0
OUTPUT_FILE=b.ahkl
INPUT_FILE=x3/XDS_ASCII.HKL
""")
    assert g["sg"] == 19 and len(outs) == 2
    assert outs[0]["friedel"] is False and outs[0]["inputs"][0]["reference"]
    assert outs[0]["inputs"][0]["crystal"] == "c1" and outs[0]["inputs"][1]["res"] == (30.0, 2.0)


def test_cc_half_sigma_tau_matches_expectation():
    rng = np.random.default_rng(0)
    n, m = 4000, 6
    mu = rng.exponential(10, n)
    uid = np.repeat(np.arange(n), m)
    I = mu[uid] + rng.normal(0, 4, uid.size)
    cc = cc_half_sigma_tau(uid, I)
    # theory: var(mu)=100, error variance of half-mean = 16/3
    expect = 100 / (100 + 16 / 3)
    assert abs(cc - expect) < 0.02


def test_robust_axial_merge_resists_spike():
    hkl = np.array([[0, 0, 3]] * 5)
    I = np.array([0.1, -0.3, 0.2, 50.0, 0.0])          # one multiple-diffraction spike
    s = np.ones(5)
    from xpg.symops import hkl_key
    Im, sm, n = _robust_axial_merge(hkl, I, s, hkl_key(hkl[:1]))
    assert abs(Im[0]) < 0.5 and n[0] == 5


def test_split_merged_two_touching_spots():
    yy, xx = np.mgrid[0:20, 0:20]
    img = 100 * np.exp(-((xx - 7) ** 2 + (yy - 10) ** 2) / 4) + 80 * np.exp(-((xx - 12) ** 2 + (yy - 10) ** 2) / 4)
    sel = img > 2
    y, x = np.nonzero(sel)
    z = np.zeros_like(x)
    v = img[sel]
    lab = np.zeros(x.size, int)
    new = _split_merged(z, y, x, v, np.zeros_like(v) + 1.0, lab, 20, 20)
    assert np.unique(new).size == 2


def test_detwin_recovers_untwinned():
    sg = gemmi.SpaceGroup("P 4")
    rng = np.random.default_rng(1)
    h = rng.integers(-15, 15, (3000, 3))
    hasu, _, _ = to_asu(h, sg)
    hasu = np.unique(hasu, axis=0)
    hasu = hasu[np.any(hasu != 0, axis=1)]
    key = {tuple(x): i for i, x in enumerate(hasu)}
    I0 = rng.exponential(100, len(hasu))
    R = np.array(gemmi.Op("k,h,-l").rot, int) // gemmi.Op.DEN
    h2, _, _ = to_asu(hasu @ R, sg)
    j = np.array([key.get(tuple(x), -1) for x in h2])
    ok = j >= 0
    a = 0.25
    It = I0.copy()
    It[ok] = (1 - a) * I0[ok] + a * I0[j[ok]]
    Id, sd, okd = detwin(hasu, It, np.ones(len(It)), sg, "k,h,-l", a)
    assert np.allclose(Id[okd], I0[okd], rtol=1e-6, atol=1e-6)


def test_attenuation_table():
    assert abs(mu_linear(0.9795, "Si") - 3.707) / 3.707 < 0.05      # tabulated
    assert abs(mu_linear(0.9795, "AIR") - 3.17e-4) / 3.17e-4 < 0.05  # tabulated


def test_zero_dose_detects_linear_decay():
    rng = np.random.default_rng(2)
    n = 500
    uid = np.repeat(np.arange(n), 8)
    dose = np.tile(np.arange(8.0), n)
    I0 = rng.exponential(1000, n) + 200
    slope = np.where(np.arange(n) < 250, -0.05 * I0, 0.0)          # half the reflections decay
    I = I0[uid] + slope[uid] * dose + rng.normal(0, 5, uid.size)
    est, s, used = zero_dose_merge(uid, I, np.full(uid.size, 5.0), dose, 0.01)
    assert used[:250].mean() > 0.9 and used[250:].mean() < 0.1
    assert np.median(np.abs(est[:250] - I0[:250]) / I0[:250]) < 0.02



def test_fm_superpixel_fractions_match_pixel_sums():
    """FM super-pixels: the profile sampled at s x s super-pixel centres (J the super-pixel's solid angle) matches the
    s x s sums of the pixel-level fractions (sigma 5.6 px, s = 2); bin_pixels sums counts and flags masked /
    overloaded super-pixels."""
    import torch
    from xpg import tnp
    from xpg.forward.driver import bin_pixels
    from xpg.forward.model import fraction_terms
    from xpg.geometry import PIX0
    from xpg.integrate_kernels import _detector_consts
    from xpg.simulate import SimSpec, make_geometry
    a = np.arange(30, dtype=np.float32).reshape(5, 6)
    a[0, 0], a[3, 2] = -1, -2
    b = bin_pixels(a, 2)
    assert b.shape == (2, 3) and b[0, 0] == -1 and b[1, 1] == -2 and b[0, 1] == 2 + 3 + 8 + 9
    g = make_geometry(SimSpec(qx=0.075, distance=200.0, nx=1400, ny=1450, orgx=701.3, orgy=722.6))
    s, h2 = 2, 14
    xs, ys = np.array([905.37, 300.81]), np.array([610.81, 1200.22])
    with tnp.on(torch.device("cpu")):
        det = _detector_consts(g)
        Xv = g.pixel_to_lab(xs, ys)
        S = torch.as_tensor(Xv / np.linalg.norm(Xv, axis=1, keepdims=True) / g.wavelength, dtype=torch.float32)
        t = lambda v, dt=torch.float32: torch.as_tensor(np.asarray(v), dtype=dt)
        cb = np.floor(xs - PIX0 + 0.5).astype(int) // s - h2
        rb = np.floor(ys - PIX0 + 0.5).astype(int) // s - h2
        common = (det, t(g.axis), t(g.s0), 0.0, float(np.radians(g.osc)), float(np.radians(0.12)),
                  float(np.radians(0.15)), S, t(np.radians([10.1, 20.2])))
        iz0, Dz = t([20, 40], torch.int64), 1
        Pb = fraction_terms(*common, t(cb, torch.int64), t(rb, torch.int64), iz0, Dz, 2 * h2 + 1, pxs=s)[0][1]
        Pf = fraction_terms(*common, t(cb * s, torch.int64), t(rb * s, torch.int64), iz0, Dz, s * (2 * h2 + 1))[0][1]
    Pf = Pf.reshape(2, 2 * h2 + 1, s, 2 * h2 + 1, s).sum((2, 4))
    assert torch.allclose(Pb.sum((1, 2)), Pf.sum((1, 2)), rtol=5e-3)
    assert float((Pb - Pf).abs().max() / Pf.max()) < 0.02


def test_ice_spot_bands_flags_ring_reflections_only():
    """Reflections on a sharp ice ring (every repeat inflated) are found as a band at the ring; clean Wilson data
    give none (ice.ice_spot_bands, 5UBA)."""
    from xpg.ice import ice_spot_bands
    rng = np.random.default_rng(3)
    cell = gemmi.UnitCell(54.4, 54.5, 105.8, 90, 90, 90)
    g = np.arange(-28, 29)
    h = np.stack(np.meshgrid(g, g, np.arange(0, 56)), -1).reshape(-1, 3)
    h = h[np.any(h != 0, axis=1)]
    d = cell.calculate_d_array(h)
    h, d = h[(d > 1.85) & (d < 15)], d[(d > 1.85) & (d < 15)]
    h = h[rng.random(len(h)) < 0.5]                                 # half of the sphere: ~5e4 unique reflections
    s = 1 / cell.calculate_d_array(h)
    mean_I = 1e4 * np.exp(-2 * 20 * (s / 2) ** 2)
    I_true = rng.exponential(1.0, len(h)) * mean_I
    rep = np.repeat(np.arange(len(h)), 3)                           # three repeats per reflection
    sig = np.sqrt(I_true[rep] + 50)
    I = I_true[rep] + rng.normal(0, 1, rep.size) * sig
    phi = rng.uniform(0, 300, rep.size)
    assert ice_spot_bands(s[rep], I, sig, phi, h[rep]) == []
    # a ring 0.5% above the 2.249 A ice spacing (header geometry slightly off): a fifth of the reflections within
    # 0.002 1/A carry 30x the mean intensity of ring counts in every repeat
    ring = (np.abs(s - 1 / 2.26) < 0.002) & (rng.random(len(h)) < 0.2)
    bands = ice_spot_bands(s[rep], I + np.where(ring[rep], 30 * mean_I[rep], 0.0), sig, phi, h[rep])
    assert len(bands) == 1 and bands[0][0] < 2.26 < bands[0][1] and bands[0][1] - bands[0][0] < 0.04


def test_ice_ring_record_equals_applied_without_space_group():
    """The ice-ring decision is recorded also when SCALE runs before the space group is known (P 1 mates; a
    single-pass run): the report and the extra lattices read the record. The next SCALE run replaces it."""
    from types import SimpleNamespace
    from xpg.stage_scale import ScaleStage
    rng = np.random.default_rng(5)
    a, b, c = 54.4, 54.5, 105.8
    g = np.arange(-30, 31)
    h = np.stack(np.meshgrid(g, g, np.arange(0, 60)), -1).reshape(-1, 3)
    s = np.linalg.norm(h / np.array([a, b, c]), axis=1)
    sel = (s > 1 / 15) & (s < 1 / 1.7) & (rng.random(len(h)) < 0.5)
    h, s = h[sel], s[sel]
    I_true = rng.exponential(1.0, len(h)) * 1e4 * np.exp(-40 * (s / 2) ** 2)
    rep = np.repeat(np.arange(len(h)), 3)
    sig = np.sqrt(I_true[rep] + 50)
    I = I_true[rep] + rng.normal(0, 1, rep.size) * sig
    rings = [(2.240, 2.262), (1.910, 1.925)]
    on = (s[rep] >= 1 / 2.262) & (s[rep] <= 1 / 2.240)
    st = SimpleNamespace(results={"ice_rings": rings}, space_group=None,
                         exp=SimpleNamespace(crystal=SimpleNamespace(Astar=np.diag([1 / a, 1 / b, 1 / c]))))
    pipe = SimpleNamespace(st=st, p=SimpleNamespace(ice_validate=True), log=lambda *_: None)
    refl = dict(hkl=h[rep], I=I + np.where(on, 3 * I_true[rep] + 20 * sig, 0.0), sig=sig)   # ring counts in 2.25 A
    out = ScaleStage._validate_ice_rings(pipe, refl, np.ones(rep.size, bool))
    assert out == [rings[0]] and st.results["ice_rings_excluded"] == out
    st.space_group = gemmi.SpaceGroup("P 1")                         # pass 2 on clean data: decided anew
    out = ScaleStage._validate_ice_rings(pipe, dict(hkl=h[rep], I=I, sig=sig), np.ones(rep.size, bool))
    assert out == [] and st.results["ice_rings_excluded"] == []


def test_smv_reader_and_adsc_header(tmp_path):
    """SMV (ADSC .img): pixels in the header's byte order (the last of repeated keys counts), image series, and the
    ADSC header conventions (beam centre X / Y along slow / fast, pedestal as OFFSET, 0 = dead, OVERLOAD 65000)."""
    from xpg.io.images import open_images, read_smv
    from xpg.setup import params_from_images
    rng = np.random.default_rng(0)
    imgs = []
    for k, bo in ((1, "big_endian"), (2, "little_endian")):
        a = rng.integers(40, 3000, size=(6, 8)).astype(np.uint16)
        a[0, 0], a[5, 7] = 0, 65535
        head = (f"{{\nHEADER_BYTES=  512;\nDIM=2;\nBYTE_ORDER=big_endian;\nTYPE=unsigned_short;\nSIZE1=8;\nSIZE2=6;\n"
                f"PIXEL_SIZE=0.1;\nDISTANCE=250.0;\nWAVELENGTH=0.97;\nBEAM_CENTER_X=0.25;\nBEAM_CENTER_Y=0.35;\n"
                f"OSC_START={k - 1}.0;\nOSC_RANGE=1.0;\nIMAGE_PEDESTAL=40;\nBYTE_ORDER={bo};\nCCD_IMAGE_SATURATION=65535;\n}}\n")
        raw = head.encode().ljust(512, b" ") + a.astype(">u2" if bo == "big_endian" else "<u2").tobytes()
        (tmp_path / f"t_{k:03d}.img").write_bytes(raw)
        imgs.append(a)
    data, h = read_smv(tmp_path / "t_001.img")
    assert np.array_equal(data, imgs[0]) and h["nx"] == 8 and h["ny"] == 6
    iset = open_images(str(tmp_path / "t_???.img"), overload=65000)
    assert (iset.first, iset.last) == (1, 2) and np.array_equal(iset.read_raw(2), imgs[1])
    p = params_from_images(str(tmp_path / "t_001.img"), tmp_path)
    assert p.detector == "ADSC" and p.name_template.endswith("t_???.img") and p.data_range == (1, 2)
    assert np.isclose(p.orgx, 3.5) and np.isclose(p.orgy, 2.5)          # BEAM_CENTER_Y / QX, BEAM_CENTER_X / QY
    assert p.offset == 40.0 and p.min_valid_pixel == 1 and p.overload == 65000.0 and p.sensor_thickness == 0.0


def test_integration_passes_fit_memory_budget():
    """Blocks held open over their widest reflection: one pass over the frames keeps every lattice's blocks of a phi range
    open together; a smaller budget splits the blocks over passes, by lattice first, then along phi."""
    from xpg.shoebox import KERNEL_RESERVE, _plan_passes
    n_lat, n_blk, n_rows, nf = 2, 10, 100, 100
    blocks_refl, lattice_of, need_hi, t_row = [], [], [], []
    for li in range(n_lat):
        for j in range(n_blk):
            r0 = len(t_row)
            t_row += list(10 * j + np.arange(n_rows) % 10)          # rows start within the block's 10 frames
            blocks_refl.append(np.arange(r0, r0 + n_rows))
            lattice_of.append(li)
            need_hi.append(min(10 * j + 50, nf - 1))                # one wide reflection keeps the block open 50 frames
    t_row = np.array(t_row)
    w_row = np.full(t_row.size, 2.0 ** 20)                             # 1 MB per reflection, 100 MB per block
    order = sorted(range(len(blocks_refl)), key=lambda b: (10 * (b % n_blk), b))   # need_lo order, lattices interleaved
    plan = lambda budget, ord_=order: _plan_passes(ord_, blocks_refl, lattice_of, need_hi, t_row, w_row, nf, budget)
    passes, info = plan(None)
    assert passes == [order] and info["peak"] == info["one"]
    one = info["one"] - KERNEL_RESERVE
    assert 0.9e9 < one < 1.4e9                                          # ~5 blocks per lattice open + 2 x one block
    passes, info = plan(KERNEL_RESERVE + 0.7 * one)
    assert len(passes) == 2 and all(len({lattice_of[b] for b in p}) == 1 for p in passes)
    assert sorted(sum(passes, [])) == sorted(order) and all(p == [b for b in order if b in p] for p in passes)
    assert info["peak"] <= KERNEL_RESERVE + 0.7 * one
    lat0 = [b for b in order if lattice_of[b] == 0]
    passes, info = plan(KERNEL_RESERVE + 0.4 * one, lat0)
    assert len(passes) >= 2 and info["peak"] <= KERNEL_RESERVE + 0.4 * one and sorted(sum(passes, [])) == sorted(lat0)


def test_block_parts_fit_memory_budget():
    """One block larger than the budget allows (passes regroup whole blocks: >= 3x the largest block + reserve): the
    blocks in parts of consecutive phi partition the rows with about equal bytes, and some number of parts fits."""
    from xpg.shoebox import KERNEL_RESERVE, _plan_passes, _split_blocks
    n_blk, n_rows, nf = 6, 200, 120
    t_row = np.repeat(20 * np.arange(n_blk), n_rows) + np.tile(np.arange(n_rows) % 20, n_blk)
    t_end = np.minimum(t_row + 10, nf - 1)
    w_row = np.full(t_row.size, 2.0 ** 20)                             # 1 MB per reflection, 200 MB per block
    blocks_refl = [np.arange(b * n_rows, (b + 1) * n_rows) for b in range(n_blk)]
    budget = KERNEL_RESERVE + 1.5 * n_rows * 2.0 ** 20

    def plan(blocks, par):
        hi = [int(t_end[r].max()) for r in blocks]
        order = sorted(range(len(blocks)), key=lambda v: int(t_row[blocks[v]].min()))
        return _plan_passes(order, blocks, [0] * len(par), hi, t_row, w_row, nf, budget)[1]["peak"]
    assert plan(blocks_refl, range(n_blk)) > budget
    fits = []
    for k in range(2, 9):
        parts, par = _split_blocks(blocks_refl, t_row + 0.5, w_row, k)
        assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(t_row.size))
        assert all(np.isin(p_, blocks_refl[b]).all() for p_, b in zip(parts, par))
        assert max(w_row[p_].sum() for p_ in parts) <= n_rows * 2.0 ** 20 / k + 2.0 ** 20
        fits.append(plan(parts, par) <= budget)
    assert any(fits)


def test_block_parallel_clustering_identical():
    """Block-parallel SPOTS clustering gives the sequential spot table (values and order): spots of various lengths in
    frames (some cross block edges, a streak runs through every block), touching pairs that get split, frame gaps."""
    import queue
    from xpg import spotcluster as sc
    rng = np.random.default_rng(5)
    nx = ny = 120
    frames = [f for f in range(1, 121) if f not in (40, 41, 77)]
    pix = {f: [] for f in frames}
    for _ in range(300):
        cx, cy, cz = rng.uniform(5, 115), rng.uniform(5, 115), rng.uniform(1, 120)
        span, amp = rng.integers(1, 12), rng.uniform(20, 200)
        for f in range(int(cz), int(cz) + span):
            if f in pix:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        y, x = int(cy) + dy, int(cx) + dx
                        if 0 <= y < ny and 0 <= x < nx:
                            pix[f].append((y, x, amp * np.exp(-0.5 * (dx * dx + dy * dy)) + 5.0))
    for f in frames:                                            # a streak present in every frame
        pix[f] += [(60, x, 30.0) for x in range(10, 14)]
    batches = []
    for b0 in range(0, len(frames), 8):
        fr = frames[b0:b0 + 8]
        kk, yy, xx, vv = [], [], [], []
        for k, f in enumerate(fr):
            d = {}
            for y, x, v in pix[f]:
                d[(y, x)] = d.get((y, x), 0.0) + v
            for (y, x), v in sorted(d.items()):
                kk.append(k)
                yy.append(y)
                xx.append(x)
                vv.append(v)
        batches.append((fr, np.array(kk), np.array(yy), np.array(xx), np.array(vv, np.float32),
                        np.full(len(vv), 5.0, np.float32)))
    args = (nx, ny, 3, 5000, 3.0, 2)
    acc = sc._Accumulator(*args)
    st = dict(prev=None)
    for b in batches:
        sc._cluster_batch(acc, st, *b)
    acc.flush()
    a = acc.result()
    for nproc, bb in ((2, 1), (3, 2)):
        q = queue.Queue()
        for b in batches:
            q.put(b)
        q.put(None)
        p = sc._cluster_parallel(q, args, nproc, bb)
        assert len(a) > 100 and len(p) == len(a)
        for k in ("x", "y", "z", "intensity", "npix", "zmin", "zmax"):
            assert np.array_equal(getattr(a, k), getattr(p, k)), k


def test_serve_rerun_validation(tmp_path):
    """xpg serve reruns: only from stages with a checkpoint, edits validated before anything runs, no path escapes."""
    import pytest
    import tomllib
    from xpg.serve import _dump_toml, _run_dir, rerun_stages, start_rerun
    (tmp_path / "xpg.toml").write_text('[data]\nimages = "img_????.cbf"\nframes = [1, 10]\n')
    (tmp_path / "state_INDEX.pkl").write_bytes(b"")
    assert rerun_stages(tmp_path) == ["MASK"]
    with pytest.raises(ValueError, match="no checkpoint"):
        start_rerun(tmp_path, "SCALE")
    with pytest.raises(ValueError, match="unknown settings"):
        start_rerun(tmp_path, "MASK", "[processing]\nno_such_key = 1")
    assert not list((tmp_path / "reruns").glob("r*"))           # the refused rerun left nothing behind
    doc = {"data": {"images": "/a/b_????.cbf", "frames": [1, 10]}, "crystal": {"space_group": 4, "unit_cell": [1.0, 2.0, 3.0, 90.0, 90.0, 90.0]}}
    assert tomllib.loads(_dump_toml(doc)) == doc
    assert _run_dir(tmp_path, "") == tmp_path and _run_dir(tmp_path, "../..") is None and _run_dir(tmp_path, "nope") is None


def test_serve_output_files(tmp_path):
    """The dashboard lists (and serves) only the run's output files, nothing outside them."""
    from xpg.serve import output_files
    for n in ("xpg_merged_iso.mtz", "xpg_P1_merged_iso.mtz", "xpg_summary.json", "xpg.log", "state_SCALE.pkl",
              "secret.txt", "events.jsonl"):
        (tmp_path / n).write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "xpg_x.mtz").write_text("x")
    assert [f["name"] for f in output_files(tmp_path)] == ["xpg.log", "xpg_P1_merged_iso.mtz", "xpg_merged_iso.mtz",
                                                           "xpg_summary.json"]


def test_pdf_report(tmp_path):
    """xpg_report.pdf from a run directory's events and summary: two pages, problems and close calls included,
    written even without images or MR."""
    import json
    from xpg.pdf_report import _load, _problems, write_pdf_report
    ev = [dict(t=0, type="run_start", template="x_????.cbf", frames=[1, 90]),
          dict(t=1, type="flag", stage="SCALE", text="SCALE WARNING: 3 detector region(s) give weak intensities"),
          dict(t=2, type="decision", name="point group", choice="mP 2", margin=0.8, unit="posterior", low_margin=True,
               alternatives=["aP 1 (0.2)"]),
          dict(t=3, type="metric", name="shells", title="Resolution shells", xlabel="1/d^2", x=[0.01, 0.1, 0.2],
               series={"CC1/2": [0.99, 0.8, 0.2]}),
          dict(t=4, type="run_end", seconds=60.0, timing={"SPOTS": 10.0, "INTEGRATE": 40.0, "SCALE": 10.0})]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    (tmp_path / "xpg_summary.json").write_text(json.dumps(dict(space_group="P 1 21 1", cc_half_cutoff=2.0, isa=12.0,
                                                               cell=[40, 50, 60, 90, 100, 90])))
    assert [p[0] for p in _problems(_load(tmp_path))] == ["warning", "close"]
    out = write_pdf_report(tmp_path, log=lambda *a: None)
    assert out is not None and out.read_bytes()[:4] == b"%PDF" and out.read_bytes().count(b"/Type /Page\n") + \
        out.read_bytes().count(b"/Type /Page ") >= 2


def test_completeness_limit():
    """Resolution limit from the thin-shell completeness, relative to the completeness inside the complete circle on
    the detector: falling edge towards the corners cut, interior dips (ice rings) and uniformly incomplete data not."""
    from xpg.merging import ShellStats, completeness_limit, shell_edges

    def shells(compl, dmin=1.4):
        e = shell_edges(None, len(compl), dmin, 40.0)
        return [ShellStats(e[i], e[i + 1], 100, 50, 100, c, 2.0, 5.0, 0.1, 0.1, 0.05, 0.9, 0.0, 1.0, 0.5, 1.0)
                for i, c in enumerate(compl)]
    # corners: complete to the inscribed circle (d >= 1.96), then falling to 10%
    fall = shells([0.97] * 20 + [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1])
    d, plateau, thr, d_from = completeness_limit(fall, 1.96, 0.5)
    assert abs(plateau - 0.97) < 1e-9 and abs(thr - 0.485) < 1e-9 and d_from == 1.96
    assert d == fall[24].dmin          # the 50% shell is the last one reaching 48.5%
    # an excluded ice-ring shell inside the limit does not cut; complete data are not cut at all
    ice = shells([0.99] * 22 + [0.3] + [0.99] * 7)
    assert completeness_limit(ice, 1.2, 0.5)[0] is None
    assert completeness_limit(shells([1.0] * 30), 1.2, 0.5)[0] is None
    # low symmetry from a short sweep: ~45% everywhere is no reason to cut; its own falling edge is
    low = shells([0.45, 0.43, 0.46, 0.44] * 6 + [0.42, 0.40, 0.35, 0.25, 0.15, 0.05])
    d, plateau, _, _ = completeness_limit(low, 1.96, 0.5)
    assert abs(plateau - 0.445) < 0.01 and d == low[27].dmin
    # no inscribed-circle resolution, or fewer than 3 shells inside it: plateau from the inner half of the shells
    d2, plateau2, _, d_from2 = completeness_limit(fall, None, 0.5)
    assert abs(plateau2 - 0.97) < 1e-9 and d2 == fall[24].dmin and d_from2 == fall[14].dmin
    assert completeness_limit(fall, 3.1, 0.5)[3] == fall[14].dmin and fall[1].dmin >= 3.1 > fall[2].dmin
    # switched off, or too few shells
    assert completeness_limit(fall, 1.96, 0.0) is None and completeness_limit(fall[:3], 1.96, 0.5) is None


def test_completeness_limit_plateau_range_reported(tmp_path):
    """The SCALE log and the report name the shells the completeness plateau came from: the complete circle on the
    detector, or the inner half of the shells when fewer than 3 thin shells lie inside it (beam near an edge)."""
    from types import SimpleNamespace
    from xpg.merging import ShellStats, shell_edges
    from xpg.report import build_markdown
    from xpg.stage_scale import ScaleStage
    e = shell_edges(None, 30, 1.4, 40.0)
    compl = [0.97] * 20 + [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1]
    fine = [ShellStats(e[i], e[i + 1], 100, 50, 100, c, 2.0, 5.0, 0.1, 0.1, 0.05, 0.9, 0.0, 1.0, 0.5, 1.0)
            for i, c in enumerate(compl)]
    d_u = np.linspace(1.4, 40.0, 5000)

    def run(d_ins):
        lines = []
        g = SimpleNamespace(resolution_at_edge_inscribed=lambda: d_ins, osc=0.1, wavelength=1.0, nx=100, ny=100)
        xtal = SimpleNamespace(cell=np.r_[40.0, 50, 60, 90, 90, 90])
        st = SimpleNamespace(results={}, exp=SimpleNamespace(geom=g, crystal=xtal), space_group=None,
                             model=SimpleNamespace(sigma_d=0.1, sigma_m=0.2))
        p = SimpleNamespace(min_shell_completeness=0.5, include_resolution=(50.0, 0.0), name_template="x_????.cbf",
                            data_range=(1, 10), exclude_data_range=[])
        pipe = SimpleNamespace(st=st, p=p, log=lines.append, wd=tmp_path)
        dcut, _ = ScaleStage._limit_completeness_input(pipe, fine, d_u, 1.45, 1.45, "CC1/2 = 0.3 fit")
        assert abs(dcut - fine[24].dmin) < 0.01
        st.results["cc_half_cutoff"] = dcut
        return st.results["completeness_limit"], lines[-1], build_markdown(pipe)
    cl, log, md = run(1.96)                       # 20 thin shells inside the complete circle: as before
    assert "d_plateau" not in cl and cl["d_inscribed"] == 1.96
    assert log.endswith("reached at d >= 1.96 A, the complete circle on the detector)")
    assert "completeness at low resolution, d >= 1.96 A;" in md
    cl, log, md = run(3.1)                        # 2 thin shells inside it: the inner half of the shells
    assert cl["d_plateau"] == fine[14].dmin and cl["d_inscribed"] == 3.1
    assert f"reached at d >= {fine[14].dmin:.2f} A, the inner half of the shells" in log
    assert "complete circle on the detector)" not in log
    assert f"d >= {fine[14].dmin:.2f} A, the inner half of the shells;" in md and "d >= 3.10 A;" not in md


def test_scale_rerun_drops_previous_limit_statistics():
    """Every SCALE run (pass 2, a SCALE-only rerun) starts without the statistics inside the previous run's
    resolution limit: when it sets no limit of its own, the reports, the summary and the lattice-merge comparison
    must not show the previous pass's shells."""
    import pytest
    from types import SimpleNamespace
    from xpg.stage_scale import ScaleStage

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop
    st = SimpleNamespace(results=dict(shells_cut=["pass 1"], overall_cut="pass 1", ice_rings_excluded=[], isa=10.0),
                         refl={}, exp=SimpleNamespace(geom=None))
    pipe = SimpleNamespace(st=st, p=None, log=lambda *_: None, _select_observations=stop)
    with pytest.raises(Stop):
        ScaleStage.job_scale(pipe, final=True)
    assert st.results == dict(isa=10.0)


def test_scale_rerun_drops_previous_aniso_and_twinning(tmp_path, monkeypatch):
    """Every SCALE run starts without the previous run's anisotropy, twinning and tNCS results: when its own
    analysis fails (or anisotropy is off), the summary must not show the previous pass's limits, anisotropic
    statistics, Wilson B or pathologies, and its outputs must not be detwinned with the previous pass's twin law.
    The tNCS modulation, decided with the symmetry in pass 1, stays."""
    import json
    import pytest
    from types import SimpleNamespace
    from xpg import stage_scale
    from xpg.stage_scale import ScaleStage

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop
    tw = SimpleNamespace(wilson_b=25.0, l_mean=0.45, i2_over_i_sq=1.9, twin_laws=[("-h,-k,l", None, 0.2, 500, 0.2)])
    tm = ([0.5, 0.5, 0.0], 0.3)
    prev = dict(aniso=dict(limits=[2.0, 2.0, 2.4]), shells_aniso=["pass 1"], overall_aniso="pass 1",
                ellipsoidal_completeness=0.9, twinning=tw, tncs=dict(ratio=0.3),
                hidden_twin=[dict(space_group="P 1 2 1")])
    st = SimpleNamespace(results=dict(prev, tncs_modulation=tm, isa=10.0), refl={}, exp=SimpleNamespace(geom=None),
                         space_group=None, model=SimpleNamespace(sigma_d=0.03, sigma_m=0.2))
    pipe = SimpleNamespace(st=st, p=None, log=lambda *_: None, _select_observations=stop, wd=tmp_path, anisotropy=True,
                           _aniso_inside=np.zeros((200, 3), int), _hidden_twin=["pass 1"])
    with pytest.raises(Stop):
        ScaleStage.job_scale(pipe, final=True)
    assert st.results == dict(isa=10.0, tncs_modulation=tm)
    assert pipe._aniso_inside is None and pipe._hidden_twin is None
    # the rest of that run with both analyses failed: no anisotropic statistics, no detwinning, a clean summary
    monkeypatch.setattr(stage_scale, "write_merged_outputs", lambda *a, **k: None)
    pipe.p = SimpleNamespace(free_r_reference=None)
    ScaleStage._write_merged(pipe, dict(hkl=None), None, None, None, 1.0, True)
    ScaleStage._aniso_statistics(pipe, *[None] * 7)
    st.results["overall"] = SimpleNamespace(cc_half=0.99)
    st.exp = SimpleNamespace(geom=SimpleNamespace(wavelength=1.0),
                             crystal=SimpleNamespace(cell=np.r_[[50.0] * 3, [90.0] * 3]))
    ScaleStage._write_report(pipe, True)
    s = json.loads((tmp_path / "xpg_summary.json").read_text())
    assert s["aniso"] is None and not {"statistics_aniso", "wilson_b", "pathologies"} & set(s)


def test_serve_summary_nonfinite(tmp_path):
    """The dashboard's summary and runs list from an xpg_summary.json with NaN (CCano without anomalous pairs, written
    by Python's json): strict JSON with null, which browsers parse; strict files served unchanged."""
    import json
    from xpg.serve import _browser_json, list_runs
    text = json.dumps(dict(space_group="P 1", cc_half_cutoff=2.0, wilson_b=float("nan"),
                           statistics=dict(overall=dict(cc_half=0.99, cc_ano=float("nan"), rmeas=float("inf")))),
                      indent=1)
    assert "NaN" in text
    def reject(c):
        raise ValueError(c)
    strict = json.loads(_browser_json(text), parse_constant=reject)      # as a browser: no NaN / Infinity
    assert strict["wilson_b"] is None and strict["statistics"]["overall"] == dict(cc_half=0.99, cc_ano=None, rmeas=None)
    ok = json.dumps(dict(space_group="P 1", cc_half_cutoff=2.0), indent=1)
    assert _browser_json(ok) == ok and _browser_json("{\"a\": 1") == "{\"a\": 1"
    (tmp_path / "xpg_summary.json").write_text(text)
    s = list_runs(tmp_path)[0]["summary"]
    assert s["cc_half"] == 0.99 and s["rmeas"] is None
    json.dumps(s, allow_nan=False)


def test_fill_batch_gaps_fine_slicing():
    """Deposition: frames removed inside one rotation get empty batch headers (one _diffrn entry), also for a long
    gap at high phi on a fine-sliced sweep (float32 phi in the headers); a gap where the rotation jumps by half a
    frame, or the end of the sweep, is not filled."""
    from xpg.deposit import _fill_batch_gaps

    def batches(numbers, osc, phi0=0.0, jump_after=None, jump=0.0):
        m = gemmi.Mtz(with_base=True)
        for nb in numbers:
            bt = gemmi.Mtz.Batch()
            bt.number, bt.dataset_id = int(nb), 1
            s = phi0 + (nb - 1) * osc + (jump * osc if jump_after is not None and nb > jump_after else 0.0)
            bt.floats[36], bt.floats[37] = s, s + osc
            m.batches.append(bt)
        _fill_batch_gaps(m)
        return [b.number for b in m.batches]
    for osc, n_fr, lo, hi in ((0.1, 3600, 2001, 2180), (0.1, 3600, 1501, 1540), (0.05, 7200, 6001, 6360),
                              (0.01, 36000, 30001, 31800)):
        kept = [f for f in range(1, n_fr + 1) if not lo <= f <= hi]
        assert batches(kept, osc) == list(range(1, n_fr + 1)), (osc, lo, hi)
        assert batches(kept, osc, jump_after=lo, jump=0.5) == kept
    assert batches(list(range(1, 11)) + [20], 0.0) == list(range(1, 11)) + [20]     # no rotation range in the headers


def test_extra_lattice_retry_never_raises_max_cell(monkeypatch):
    """Extra-lattice search: a rejected foreign cell is searched for again with the max cell bounded by the main
    cell, but never with a max cell above the one the rejected search used (the user's max_cell or the capped
    estimate): no retry when the bound would not lower it."""
    from types import SimpleNamespace
    from xpg import indexing
    from xpg.models import Crystal
    from xpg.spotcluster import SpotList
    n = 2000
    rng = np.random.default_rng(0)
    spots = SpotList(*(rng.random(n) for _ in range(4)), np.ones(n), np.zeros(n), np.ones(n))
    final_ok = np.arange(n) < n // 2
    foreign = Crystal(U=np.eye(3), cell=np.array([150.0, 170.0, 190.0, 90, 90, 90]))
    used = []

    def index_spots(geom, sp, dmin, max_cell, **kw):      # every search finds the same foreign cell
        used.append(max_cell)
        r = indexing.IndexingResult(SimpleNamespace(geom=geom, crystal=foreign), np.zeros((len(sp), 3), int),
                                    np.ones(len(sp), bool), 0.9, 0.1, 0.1, [])
        r.max_cell = 400.0 if max_cell is None else max_cell
        return r
    monkeypatch.setattr(indexing, "index_spots", index_spots)

    def searched(cell, max_cell):
        used.clear()
        exp = SimpleNamespace(geom=SimpleNamespace(osc=0.1), crystal=Crystal(U=np.eye(3), cell=np.array(cell, float)))
        assert indexing.search_extra_lattices_fn(exp, spots, final_ok, 3.0, max_cell, verbose=False) == ([], [])
        return used[:]
    assert searched((60, 80, 100, 90, 90, 90), None) == [None, 250.0]
    assert searched((60, 80, 180, 90, 90, 90), None) == [None]          # 450 A: above the capped estimate
    assert searched((60, 80, 100, 90, 90, 90), 150.0) == [150.0]        # 250 A: above the user's max cell
    assert searched((40, 40, 50, 90, 90, 90), 150.0) == [150.0, 125.0]


def test_incomplete_toml_without_images(tmp_path):
    """An xpg.toml without the experiment description whose image template matches no file: a clear error naming
    the file and the template, from the CLI and from dashboard reruns alike."""
    import pytest
    from xpg.inputs import read_toml
    from xpg.setup import params_from_images
    (tmp_path / "xpg.toml").write_text('[data]\nimages = "img_????.cbf"\nframes = [1, 10]\n')
    with pytest.raises(FileNotFoundError, match=r"xpg\.toml: no images match .*img_\?\?\?\?\.cbf"):
        read_toml(tmp_path / "xpg.toml")
    with pytest.raises(FileNotFoundError, match="no images match"):
        params_from_images(str(tmp_path / "x_????.img"))


def test_pass1_block_widths_only_from_pass1():
    """The second pass starts from the per-block fits only while the current integration is pass 1 with the default
    widths; after a second pass (its checkpoint has fits under the same keys) never; checkpoints without the flag:
    the empty width history."""
    from types import SimpleNamespace
    from xpg.stage_integrate import IntegrateStage
    bp = {k: dict(sigma_d=0.05, sigma_m=0.25, rcw=[], n=40, sigma_d_fit=0.04, sigma_m_fit=0.18) for k in range(4)}
    res = dict(integration_block_params=bp, integration_blocks=[None] * 4, width_history=[])
    obj = SimpleNamespace(st=SimpleNamespace(results=res), p=SimpleNamespace(beam_divergence=None, reflecting_range=None))
    assert IntegrateStage._pass1_block_widths(obj) == (0.04, 0.18, 4)          # old checkpoint
    res["pass1_default_widths"] = True
    assert IntegrateStage._pass1_block_widths(obj) == (0.04, 0.18, 4)
    res["pass1_default_widths"] = False                                        # after pass 2
    assert IntegrateStage._pass1_block_widths(obj) is None
    res.pop("pass1_default_widths")
    res["width_history"] = [(0.04, 0.18)]                                      # sweep-wide estimate succeeded
    assert IntegrateStage._pass1_block_widths(obj) is None


def test_audit_reindex_between_centred_settings(tmp_path, monkeypatch):
    """Close-call audit merges: two C cells of one hexagonal lattice are related by a matrix with +-1/2, +-3/2, exact
    on the h+k even reflections; the audit refused such alternatives as 'no integer index transformation' (on a real
    set the two likeliest of three). The reindexing is now accepted, maps every allowed index to integers, and
    merge_in uses it."""
    import xpg.io.xpgfiles as xf
    import xpg.outputs as xo
    from xpg.mr import merge_in
    from xpg.symops import centring_reindex
    a, c = 76.4, 309.0
    a1, a2, a3 = np.array([a, 0, 0]), a * np.array([-0.5, np.sqrt(3) / 2, 0]), np.array([0, 0, c])
    rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    D1 = np.column_stack([a1, a1 + 2 * a2, a3])            # orthohexagonal C cell (direct vectors as columns)
    D2 = rz(np.radians(120)) @ D1                          # the C cell of the next 2-fold pair
    A1, A2 = np.linalg.inv(D1).T, np.linalg.inv(D2).T      # reciprocal bases: q = A h
    T = np.linalg.solve(A2, rz(np.radians(0.1)) @ A1)      # refinement noise
    assert np.abs(T - np.round(T)).max() > 0.4             # what the old integer check refused
    M = centring_reindex(T, "C 2 2 2")
    assert M is not None and np.allclose(np.abs(M[:2, :2]), [[0.5, 0.5], [1.5, 0.5]])
    h = np.array([[h_, k, l] for h_ in range(-4, 5) for k in range(-4, 5) for l in range(-2, 3) if (h_ + k) % 2 == 0])
    hp = h @ M.T
    assert np.allclose(hp, np.rint(hp)) and np.allclose(A2 @ hp.T, A1 @ h.T)
    assert centring_reindex(T, "P 2 2 2") is None          # no centring: the fractions are not an indexing
    assert centring_reindex(np.linalg.solve(A2, rz(np.radians(5)) @ A1), "C 2 2 2") is None
    P = centring_reindex(np.eye(3) + 0.01, "P 1")
    assert P.dtype.kind == "i" and (P == np.eye(3)).all()
    # merge_in with the fractional matrix: integer indices in the new setting, every observation kept
    rng = np.random.default_rng(0)
    obs = h[np.abs(h).sum(1) > 0]
    u = dict(hkl=obs, I=rng.uniform(50, 100, len(obs)), sig=np.full(len(obs), 5.0),
             cell=[a, a * np.sqrt(3), c, 90, 90, 90], meta={})
    got = {}
    monkeypatch.setattr(xf, "read_unmerged_mtz", lambda p: u)
    monkeypatch.setattr(xo, "write_merged_outputs", lambda merged, *a_, **k_: got.setdefault("m", merged))
    merge_in(tmp_path, "C 2 2 2", tmp_path / "xpg_C222", reindex=M, cell=u["cell"], dmin=1.0)
    assert np.issubdtype(np.asarray(got["m"]["hkl"]).dtype, np.integer)
    assert int(np.sum(got["m"]["mult"])) == len(obs)

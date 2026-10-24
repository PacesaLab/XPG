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

"""Multi-segment detectors, distortion tables, goniometer chains and input keywords."""
import numpy as np

from xpg.config import params_from_dict, parse_xds_inp_text, format_xds_inp
from xpg.detector import Segment
from xpg.geometry import Geometry, cell_to_B, predict_reflections
from xpg.models import Crystal, Experiment


def _geom(**kw):
    return Geometry(wavelength=1.0, beam=np.array([0, 0, 1.]), axis=np.array([1, 0, 0.]), orgx=1030, orgy=1080,
                    distance=150, ed1=np.array([1, 0, 0.]), ed2=np.array([0, 1, 0.]), qx=0.075, qy=0.075, nx=2068,
                    ny=2162, phi0=0, osc=0.2, starting_frame=1, sensor_thickness=0.45, mu=3.7, **kw)


def test_segment_keywords_roundtrip():
    txt = """NX=2068 NY=2162 QX=0.075 QY=0.075 ORGX=1030 ORGY=1080 DETECTOR_DISTANCE=150
 SEGMENT= 1 1028 1 512
  SEGMENT_ORGX= 0.5 SEGMENT_ORGY= -0.25 SEGMENT_DISTANCE= 0.1
  DIRECTION_OF_SEGMENT_X-AXIS= 1 0.001 0
  DIRECTION_OF_SEGMENT_Y-AXIS= -0.001 1 0
  REFINE_SEGMENT= POSITION
 SEGMENT= 1041 2068 1 512
OFFSET= 40  WFAC1= 1.3  SNRC= 0.5  MINIMUM_I/SIGMA= 4  NBATCH= 7
DATA_RANGE_FIXED_SCALE_FACTOR= 1 10 1.0
"""
    p = params_from_dict(parse_xds_inp_text(txt))
    assert len(p.segments) == 2 and p.segments[0].orgx == 0.5 and p.segments[0].refine == ("POSITION",)
    assert p.segments[1].rect == (1041, 2068, 1, 512) and p.segments[1].orgx == 0.0
    assert p.offset == 40 and p.wfac1 == 1.3 and p.snrc == 0.5 and p.minimum_i_sigma == 4 and p.nbatch == 7
    assert p.fixed_scale_range == (1, 10, 1.0)
    p2 = params_from_dict(parse_xds_inp_text(format_xds_inp(p)))
    assert len(p2.segments) == 2 and abs(p2.segments[0].orgy + 0.25) < 1e-6
    assert "SEGMENT" not in p.extra and "__order__" not in p.extra


def test_single_segment_equals_plain_detector():
    g = _geom()
    rng = np.random.default_rng(0)
    x, y = rng.uniform(1, 2068, 500), rng.uniform(1, 2162, 500)
    g1 = g.copy(segments=[Segment(1, 2068, 1, 2162)])
    assert np.abs(g1.pixel_to_lab(x, y) - g.pixel_to_lab(x, y)).max() < 1e-9


def test_segment_and_distortion_roundtrip():
    th = np.radians(0.4)
    segs = [Segment(1, 2068, 1, 1000),
            Segment(1, 2068, 1040, 2162, orgx=0.4, orgy=-0.3, distance=0.2,
                    ex=np.array([np.cos(th), np.sin(th), 0.02]), ey=np.array([-np.sin(th), np.cos(th), 0]))]
    yy, xx = np.mgrid[0:2162, 0:2068]
    geo = ((0.5 * np.sin(xx / 150.)).astype(np.float32), (0.4 * np.cos(yy / 200.)).astype(np.float32))
    g = _geom(segments=segs, geo_corr=geo)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(1, 2068, 2000), rng.uniform(1, 2162, 2000)
    k = g.segment_of(x, y)
    m = k >= 0
    assert (~m).sum() > 0                                  # some points in the gap
    xb, yb, kb = g.s_to_pixel(g.pixel_to_s(x[m], y[m]), return_segment=True)
    assert np.abs(xb - x[m]).max() < 1e-6 and np.abs(yb - y[m]).max() < 1e-6 and np.all(kb == k[m])


def test_kernel_detector_matches_numpy():
    from xpg import tnp as jnp
    from xpg.integrate_kernels import _detector_consts, _pixel_S
    th = np.radians(0.4)
    segs = [Segment(1, 2068, 1, 1000), Segment(1, 2068, 1040, 2162, orgx=0.4, distance=0.2,
                                               ex=np.array([np.cos(th), np.sin(th), 0.0]))]
    rng = np.random.default_rng(2)
    x, y = rng.uniform(1, 2068, 300), rng.uniform(1, 2162, 300)
    for g in (_geom(), _geom(segments=segs)):
        Sj = np.asarray(_pixel_S(_detector_consts(g), jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32)))
        Sn = g.pixel_to_s(x, y)
        assert np.abs(Sj - Sn).max() / np.abs(Sn).max() < 1e-6


def test_segment_refinement_recovers_module_offset():
    from xpg.refine import refine
    nom = [Segment(x1, x1 + 1027, y1, y1 + 511) for y1 in (1, 551, 1101, 1651) for x1 in (1, 1041)]
    true = [Segment(*s.rect) for s in nom]
    th = np.radians(0.3)
    true[3].orgx, true[3].orgy, true[3].distance = 0.8, -0.5, 0.15
    true[3].ex = np.array([np.cos(th), np.sin(th), 0.0])
    true[3].ey = np.array([-np.sin(th), np.cos(th), 0.0])
    gt = _geom(segments=true)
    A = cell_to_B((50, 60, 70, 90, 95, 90))
    pr = predict_reflections(gt, A, 1.8, z_range=(0, 200))
    rng = np.random.default_rng(1)
    x = pr["x"] + rng.normal(scale=0.1, size=pr["x"].size)
    y = pr["y"] + rng.normal(scale=0.1, size=pr["x"].size)
    z = pr["z"] + rng.normal(scale=0.05, size=pr["x"].size)
    st = Experiment(_geom(segments=[Segment(*s.rect) for s in nom]), Crystal.from_Astar(A, "m"))
    r = refine(st, pr["hkl"], x, y, z, ["POSITION", "BEAM", "ORIENTATION", "CELL", "SEGMENT"], max_err_xy=10)
    assert r.rmsd_xy < 0.16
    yy, xx = np.mgrid[20:2150:40, 20:2060:40]
    xx, yy = xx.ravel() + 0.0, yy.ravel() + 0.0
    m = gt.segment_of(xx, yy) >= 0
    xr, yr = r.exp.geom.s_to_pixel(gt.pixel_to_s(xx[m], yy[m]))
    assert np.nanmax(np.hypot(xr - xx[m], yr - yy[m])) < 0.05


def test_xparm_segments_roundtrip(tmp_path):
    from xpg.io.legacy import read_xparm, write_xparm
    segs = [Segment(1, 1028, 1, 512, orgx=0.3, orgy=-0.2, distance=0.1), Segment(1041, 2068, 1, 512)]
    e = Experiment(_geom(segments=segs), Crystal.from_Astar(cell_to_B((50, 60, 70, 90, 95, 90)), "m"))
    write_xparm(tmp_path / "xparm.txt", e, 4)
    e2, sg = read_xparm(tmp_path / "xparm.txt")
    assert sg == 4 and len(e2.geom.segments) == 2 and abs(e2.geom.segments[0].orgx - 0.3) < 0.01


def test_nexus_kappa_chain(tmp_path):
    import h5py
    from xpg.goniometer import read_nexus_goniometer, rot
    f = h5py.File(tmp_path / "m.h5", "w")
    t = f.create_group("/entry/sample/transformations")
    om = t.create_dataset("omega", data=np.arange(10) * 0.1 + 30.0)
    om.attrs.update(vector=[-1.0, 0, 0], transformation_type="rotation", depends_on=".", units="deg")
    ka = t.create_dataset("kappa", data=[60.0])
    a = np.radians(50)
    kv = [-np.cos(a), 0, np.sin(a)]
    ka.attrs.update(vector=kv, transformation_type="rotation", depends_on="/entry/sample/transformations/omega")
    ph = t.create_dataset("phi", data=[20.0])
    ph.attrs.update(vector=[-1.0, 0, 0], transformation_type="rotation", depends_on="/entry/sample/transformations/kappa")
    f["/entry/sample/depends_on"] = "/entry/sample/transformations/phi"
    f.close()
    with h5py.File(tmp_path / "m.h5") as f:
        ch = read_nexus_goniometer(f)
    assert [a_.name for a_ in ch.axes] == ["omega", "kappa", "phi"]
    assert abs(ch.axes[0].increment - 0.1) < 1e-9
    assert np.allclose(ch.effective_axis(), [-1, 0, 0])          # omega scan: outermost axis
    ch.axes[0].values = np.array([30.0])
    ch.axes[2].values = np.array([20.0, 20.1])                   # phi scan
    expect = rot([-1, 0, 0], 30.0) @ rot(kv, 60.0) @ np.array([-1.0, 0, 0])
    assert np.allclose(ch.effective_axis(), expect)


def test_offset_and_dark_subtraction(tmp_path):
    from xpg.io.cbf import write_cbf
    from xpg.io.images import open_images
    img = np.full((20, 30), 100, np.int32)
    img[0, 0] = -1
    for i in (1, 2):
        write_cbf(tmp_path / f"im_{i:05d}.cbf", img, dict(qx=0.172, qy=0.172, wavelength=1.0, distance=100,
                                                          orgx=15, orgy=10, start_angle=0, osc=0.1))
    dark = np.full((20, 30), 7, np.int32)
    write_cbf(tmp_path / "dark.cbf", dark, dict(qx=0.172, qy=0.172, wavelength=1.0, distance=100, orgx=15, orgy=10,
                                                start_angle=0, osc=0.1))
    iset = open_images(str(tmp_path / "im_?????.cbf"), offset=10, dark_current_image=str(tmp_path / "dark.cbf"))
    a = iset.get(1)
    assert a[5, 5] == 83 and a[0, 0] == -1


def test_merge_input_keywords():
    from xpg.multiset import parse_xscale_inp
    g, outs = parse_xscale_inp("""SPACE_GROUP_NUMBER=92 UNIT_CELL_CONSTANTS= 58 58 150 90 90 90
WFAC1= 2.0 SNRC= 1.0 NBATCH= 4 RESOLUTION_SHELLS= 6 4 3 2.5 2
REIDX= -1 0 0 0  0 1 0 0  0 0 -1 0
STRICT_ABSORPTION_CORRECTION=TRUE PRINT_CORRELATIONS=FALSE SAVE_CORRECTION_IMAGES=FALSE
REFERENCE_DATA_SET= ref.mtz
OUTPUT_FILE=peak.ahkl FRIEDEL'S_LAW=FALSE
INPUT_FILE=a/XDS_ASCII.HKL
  WEIGHT= 1.5 STARTING_DOSE= 10 DOSE_RATE= 0.2 NBATCH= 9
  REIDX_ISET= 0 1 0 0  1 0 0 0  0 0 -1 0
OUTPUT_FILE=remote.ahkl
INPUT_FILE=b/XDS_ASCII.HKL
""")
    assert g["wfac1"] == 2.0 and g["snrc"] == 1.0 and g["nbatch"] == 4 and g["shells"] == [6, 4, 3, 2.5, 2]
    assert g["reidx"][0, 0] == -1 and g["strict_absorption"] is True and g["print_correlations"] is False
    assert g["save_images"] is False and g["reference"] == "ref.mtz"
    a = outs[0]["inputs"][0]
    assert a["weight"] == 1.5 and a["starting_dose"] == 10 and a["dose_rate"] == 0.2 and a["nbatch"] == 9
    assert a["reidx"][0, 1] == 1 and outs[0]["friedel"] is False and outs[1]["inputs"][0]["weight"] == 1.0


def test_common_scale_fit():
    from xpg.multiset import common_scale_fit
    rng = np.random.default_rng(0)
    s2 = rng.uniform(0.001, 0.25, 5000)
    I0 = rng.exponential(1000, 5000) * np.exp(-10 * s2)
    Ik = I0 / (1.3 * np.exp(-0.5 * 4.0 * s2))          # K = 1.3, relative B = 4 A^2
    a, b = common_scale_fit(I0, Ik, s2)
    assert abs(np.exp(a) - 1.3) < 0.02 and abs(-2 * b - 4.0) < 0.3


def test_zero_dose_false_positive_rate_matches_level():
    from xpg.multiset import zero_dose_merge
    rng = np.random.default_rng(0)
    n = 20000
    m = rng.integers(3, 8, n)
    uid = np.repeat(np.arange(n), m)
    dose = rng.uniform(0, 100, uid.size)
    I0 = rng.exponential(100, n)
    s = np.sqrt(I0[uid] + 25)
    I = I0[uid] + rng.normal(0, 1, uid.size) * s
    _, _, used = zero_dose_merge(uid, I, s, dose, 0.1)
    assert 0.08 < used.mean() < 0.12


def test_inconsistent_region_detection():
    from xpg.scaling import inconsistent_regions
    rng = np.random.default_rng(3)
    n = 4000
    uid = np.repeat(np.arange(n), 6)
    I0 = rng.exponential(2000, n) + 500
    x, y = rng.uniform(0, 1000, uid.size), rng.uniform(0, 1000, uid.size)
    I = I0[uid] * (1 + rng.normal(0, 0.03, uid.size))
    sig = np.sqrt(I0[uid])
    bad, rows = inconsistent_regions(I, sig, uid, x, y, 1000, 1000)
    assert not bad.any()
    shadow = (x > 750) & (y > 750)                      # e.g. unmasked beamstop arm
    I2 = np.where(shadow, 0.05 * I, I)
    bad, rows = inconsistent_regions(I2, sig, uid, x, y, 1000, 1000)
    assert bad[shadow].mean() > 0.9 and bad[~shadow].mean() < 0.02


def test_fix_scale_sets_factor():
    from xpg.scaling import Scaler
    rng = np.random.default_rng(4)
    n = 600
    uid = np.repeat(np.arange(n), 5)
    phi = rng.uniform(0, 90, uid.size)
    g = 1 + 0.3 * np.sin(np.radians(phi) * 2)
    I0 = rng.exponential(1000, n) + 100
    I = g * I0[uid] + rng.normal(0, 5, uid.size)
    sc = Scaler(I, np.full(uid.size, 5.0), uid, phi, np.full(uid.size, 0.05), np.tile([0, 0, 1.0], (uid.size, 1)),
                np.zeros(uid.size), np.zeros(uid.size), 100, 100, corrections=["DECAY"])
    sc.run(n_cycles=1, verbose=False)
    sel = phi < 10
    sc.fix_scale(sel, 2.0)
    assert abs(np.exp(-np.mean(np.log(sc.scales()[sel]))) - 2.0) < 0.05


def test_matthews():
    from xpg.matthews import matthews_table, mw_from_sequence
    assert abs(mw_from_sequence(">x\nGG") - (2 * 57.0519 + 18.015)) < 1e-6
    rows = matthews_table((112.71, 38.2486, 77.4729, 90, 112.635, 90), "C 1 2 1", 18907.85, 2.0)
    best = max(rows, key=lambda r: r["p"])
    assert best["n"] == 2 and 35 < best["solvent"] < 45 and abs(sum(r["p"] for r in rows) - 1) < 1e-9


def test_damaged_eiger_file_recovery(tmp_path):
    """A data file HDF5 cannot open (truncated copy) is recovered from its raw chunks: the intact frames come back
    bit-identical, the damaged last frame is excluded, and nothing crashes on the damaged chunk."""
    import h5py
    import hdf5plugin
    from xpg.io.images import EigerImageSet
    rng = np.random.default_rng(0)
    ny, nx, n = 64, 48, 5
    frames = rng.poisson(3.0, (3 * n, ny, nx)).astype(np.uint32)
    with h5py.File(tmp_path / "x_master.h5", "w") as m:
        g = m.create_group("entry/data")
        for b in range(3):
            fn = f"x_data_{b + 1:06d}.h5"
            with h5py.File(tmp_path / fn, "w") as f:
                ds = f.create_dataset("entry/data/data", data=frames[b * n:(b + 1) * n], chunks=(1, ny, nx),
                                      **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
                ds.attrs["image_nr_low"] = np.uint64(b * n + 1)
                ds.attrs["image_nr_high"] = np.uint64(b * n + n)
            g[f"data_{b + 1:06d}"] = h5py.ExternalLink(fn, "entry/data/data")
    with h5py.File(tmp_path / "x_data_000002.h5") as f:          # cut into the last frame's chunk
        info = f["entry/data/data"].id.get_chunk_info(n - 1)
        cut = info.byte_offset + info.size // 2
    with open(tmp_path / "x_data_000002.h5", "r+b") as fh:
        fh.truncate(cut)
    s = EigerImageSet(str(tmp_path / "x_master.h5"))
    assert s.bad_blocks == [] and s.recovered_blocks == [("data_000002", n + 1, n - 1, n)]
    for f in range(1, 3 * n + 1):
        if f == 2 * n:
            assert not s.available(f)
        else:
            assert np.array_equal(s.read_raw(f), frames[f - 1].astype(np.int64))
    # the device-finished decode (SPOTS on CUDA) also reads the recovered chunks
    import torch
    from xpg.io.bslz4 import lz4_stage, unshuffle_frame_torch
    assert s.bitshuffled_ok()
    s.cache_size = 0
    for f in range(1, 3 * n + 1):
        if f == 2 * n:
            continue
        buf = np.zeros(ny * nx * 4, np.uint8)
        block = lz4_stage(s.read_chunk(f), buf, 4)
        img = unshuffle_frame_torch(torch.from_numpy(buf), ny * nx, 4, block, s.overload, s.min_valid)
        assert np.array_equal(img.numpy().reshape(ny, nx), s.get(f))


def test_frame_safe_zd_stays_in_frame():
    """One-decimal ZD must not round into the next frame (batch = int(ZD) + 1; a reflection of the last
    frame printed as N.0 created a near-empty batch past DATA_RANGE)."""
    from xpg.io.legacy import _frame_safe
    z = np.array([3046.96, 3046.94, 3046.5, 0.99, 0.0, 12.04])
    printed = np.array([float(f"{v:9.1f}") for v in _frame_safe(z)])
    assert np.all(np.floor(printed) == np.floor(z))
    assert np.allclose(_frame_safe(z)[[2, 4, 5]], z[[2, 4, 5]])


def test_free_flags_twin_safe_and_deterministic():
    """Reflections related by a possible merohedral twin law share a free flag (BBF-14-b3: P32 twinned by
    -h,-k,l), and flags do not depend on which other reflections are present."""
    import gemmi
    from xpg.io.mtz import add_free_flags
    from xpg.symops import to_asu
    sg = gemmi.SpaceGroup("P 32")
    cell = (127.4, 127.4, 52.5, 90, 90, 120)
    rng = np.random.default_rng(0)
    hkl = np.unique(to_asu(rng.integers(-15, 16, (2000, 3)), sg)[0], axis=0)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    f1 = add_free_flags(hkl, sg, cell=cell)
    f2 = add_free_flags(to_asu(hkl * np.array([-1, -1, 1]), sg)[0], sg, cell=cell)
    assert np.all(f1 == f2)
    assert np.all(add_free_flags(hkl[:100], sg, cell=cell) == f1[:100])
    assert 0.03 < np.mean(f1 == 0) < 0.08


def test_lattice_isomorphism_kappa():
    """kappa ~ 1 for two lattices that differ only by noise, < 1 when the structures differ."""
    import gemmi
    from xpg.analysis import lattice_isomorphism
    rng = np.random.default_rng(1)
    sg = gemmi.SpaceGroup("P 1")
    cell = (40, 50, 60, 90, 90, 90)
    hkl = np.array([(h, k, l) for h in range(-12, 13) for k in range(-12, 13) for l in range(0, 14)
                    if (h, k, l) != (0, 0, 0)])
    hkl = hkl[np.array([gemmi.UnitCell(*cell).calculate_d(list(map(int, x))) for x in hkl]) > 3.0]
    I0 = rng.exponential(100.0, len(hkl))
    I_other = 0.5 * I0 + 0.5 * rng.exponential(100.0, len(hkl))       # partly different structure

    def obs(I, n=3):
        h = np.repeat(hkl, n, axis=0)
        sig = np.full(len(h), 20.0)
        return h, np.repeat(I, n) + rng.normal(0, 20.0, len(h)), sig
    r = lattice_isomorphism([obs(I0), obs(I0), obs(I_other)], cell, sg, nbins=2)
    assert all(0.9 < k < 1.1 for k in r["kappa"][(0, 1)])
    assert all(k < 0.85 for k in r["kappa"][(0, 2)])


def test_trigonal_candidates_both_orientations():
    """A primitive hexagonal lattice offers both trigonal orientations, P321 (2-folds along a) and P312 (2-folds
    perpendicular to a); P312 was missing, so P3112 / P3212 (GLF-35) could never be chosen or requested."""
    from xpg.lattice import lattice_candidates
    from xpg.symmetry import candidate_space_groups
    for cell in [(147.3, 147.3, 67.4, 90, 90, 120), (67.4, 147.3, 147.3, 60, 90, 90)]:
        got = {c.point_group: sorted(sg.xhm() for sg in candidate_space_groups(c))
               for c in lattice_candidates(cell) if c.bravais == "hP"}
        assert got["32"] == ["P 3 2 1", "P 31 2 1", "P 32 2 1"]
        assert got["312"] == ["P 3 1 2", "P 31 1 2", "P 32 1 2"]
        assert {"3", "6", "622"} <= set(got)


def test_mr_alternative_point_groups():
    """xpg mr: alternatives of a close-call point-group decision that share the cell basis; P321 and P312 are
    distinct although both are point group '32' by name."""
    import gemmi
    from xpg.mr import _same_basis_alternatives
    alts = ["hP 3 (0.005)", "mC 2 (0.1)", "hP 622 (0.02)", "hP 32 (0.01)", "hP 312 (0.4)"]
    assert _same_basis_alternatives(gemmi.SpaceGroup("P 31 1 2"), alts) == ["P 3", "P 6 2 2", "P 3 2 1"]
    assert _same_basis_alternatives(gemmi.SpaceGroup("P 31 2 1"), ["hP 622 (0.3)", "oC 222 (0.1)"]) == ["P 6 2 2"]


def test_fractional_twin_law_matrix():
    """Trigonal 3-fold in a C-centred orthohexagonal cell (fractional matrix): must map allowed reflections onto
    their true symmetry mates, not a floor-divided integer operator."""
    import gemmi
    from xpg.symops import apply_rot, op_matrix
    op = gemmi.Op("x/2-3/2*y,x/2+y/2,z")
    R = op_matrix(op)
    assert not np.issubdtype(R.dtype, np.integer)
    h = np.array([[1, 1, 0], [2, 0, 3], [3, 5, 1]])          # h + k even (C centring)
    h2 = apply_rot(h, R)
    assert np.issubdtype(h2.dtype, np.integer)
    assert np.allclose(h2, h @ R)
    # a lattice rotation: some power of it is the identity, on integer indices throughout
    g, order = h.copy(), 0
    for order in range(1, 7):
        g = apply_rot(g, R)
        if np.array_equal(g, h):
            break
    assert np.array_equal(g, h) and order in (2, 3, 4, 6)
    assert np.array_equal(op_matrix(gemmi.Op("x,-y,-z")), np.diag([1, -1, -1]))


def test_symmetry_decision_ncs_twin_systematic_error():
    """Point-group decision on synthetic data with known truth (tests/symsim.py): strong NCS pseudo-symmetry and a
    near-perfect twin must not pass as symmetry, nor a true element as a twin law; true symmetry with large
    systematic errors must not be rejected."""
    from symsim import CELL, decide, observations
    from xpg.simulate import SimSpec
    cases = [("3", dict(space_group="P 31", ncs_op="k,h,-l", ncs_c=0.45, ncs_b=10.0), {}),
             ("3", dict(space_group="P 31", twin_law="k,h,-l", twin_fraction=0.40), {}),
             # twin law along c (apparent 6): <|L|> saturates near a perfect twin, the 3-fold must not be lost
             ("3", dict(space_group="P 31", twin_law="-h,-k,l", twin_fraction=0.45), {}),
             ("32", dict(space_group="P 31 2 1"), dict(sys_err=0.25))]
    for truth, sk, ok in cases:
        h, I, sig, s, phi = observations(SimSpec(cell=CELL, structure_seed=1000, **sk), seed=0, **ok)
        pg, p = decide(h, I, sig, s, phi)
        assert pg.endswith(" " + truth), (sk, pg, p)

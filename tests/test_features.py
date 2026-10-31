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


def test_zero_dose_merge_skips_systematic_absences():
    """Zero-dose intensities go only to reflections the merged data hold: a measured systematic absence (dropped by
    merging_statistics) took the next reflection's slot (I(+) of 0 1 1, measured as I(-) only, became the absent
    0 1 0's value) or, as the largest key (axial 5 0 0 at the data edge), indexed past the end."""
    import gemmi
    from xpg.merging import merging_statistics, unique_ids
    from xpg.multiset import _zero_dose_into_merged, zero_dose_merge
    from xpg.symops import hkl_key, to_asu
    rng = np.random.default_rng(3)
    for name, cell, H, anom in [
            ("P 1 21 1", [40., 50., 60., 90, 100, 90],
             [(h, k, l) for h in range(-4, 5) for k in range(-4, 5) for l in range(-4, 5)], True),
            ("P 21 21 21", [40., 50., 60., 90, 90, 90],
             [(h, k, l) for h in range(5) for k in range(5) for l in range(5)] + [(5, 0, 0)], False)]:
        sg = gemmi.SpaceGroup(name)
        hasu, _, plus = to_asu(np.array([h for h in H if any(h)] * 4), sg)
        keep = ~(np.all(hasu == (0, 1, 1), axis=1) & plus) if anom else np.ones(len(hasu), bool)
        hasu, plus = hasu[keep], plus[keep]
        I = rng.gamma(1.0, 100.0, len(hasu)) + 5
        sig = np.full(len(hasu), 5.0)
        _, _, merged = merging_statistics(hasu, plus, I, sig, np.array(cell), sg, nbins=5, anomalous=True)
        uid, uk = unique_ids(hasu, plus, anomalous=anom)
        I0, s0, _ = zero_dose_merge(uid, I, sig, rng.uniform(0, 10, len(hasu)), 0.1)
        inm = _zero_dose_into_merged(merged, uk, I0, s0, anom)
        ref = dict(zip(uk.tolist(), I0.tolist()))
        pk = hkl_key(merged["hkl"]).tolist()
        assert (~inm).any()                                   # measured absences, not mapped
        if anom:
            for col, m in (("Iplus", 0), ("Iminus", 1)):
                assert np.array_equal(merged[col], [ref.get(2 * k + m, np.nan) for k in pk], equal_nan=True)
            j = pk.index(int(hkl_key(np.array([0, 1, 1]))))
            assert np.isnan(merged["Iplus"][j]) and np.isclose(merged["I"][j], merged["Iminus"][j])
        else:
            assert np.array_equal(merged["I"], [ref[k] for k in pk])


def test_run_multiset_d_limit(tmp_path):
    """d_limit (the main lattice's applied limit for the all-lattice merge) cuts the merged MTZ when it is more
    conservative than the CC1/2 = 0.3 estimate, which stays the cc_half_cutoff returned (the merge verdicts compare
    fits); the input sets' resolution ranges (dmin) must not overwrite it. Also when the estimate is coarser by at most
    d_limit_tol (the verdict's tolerance: a promoted merge must end at the limit its statistics are reported to), not
    beyond it; d_written only when the limit was the cut, with the outputs' anisotropic statistics (those of the
    anisotropic MTZ), merged reflections and Wilson B."""
    import gemmi
    from xpg.aniso import ellipsoidal_completeness
    from xpg.multiset import run_multiset
    rng = np.random.default_rng(0)
    uc = gemmi.UnitCell(30, 35, 40, 90, 90, 90)
    H = np.array([(h, k, l) for h in range(13) for k in range(15) for l in range(17) if h + k + l])
    d = np.array([uc.calculate_d(list(map(int, x))) for x in H])
    H, d = H[d >= 2.5], d[d >= 2.5]
    Itrue = rng.exponential(1000.0, len(H)) * np.exp(-50.0 / d ** 2)     # CC1/2 = 0.3 near 2.9 A
    inp = "SPACE_GROUP_NUMBER=16\nUNIT_CELL_CONSTANTS= 30 35 40 90 90 90\nOUTPUT_FILE=m\n"
    for i in range(2):
        h = np.tile(H, (3, 1)) * rng.choice([-1, 1], (3 * len(H), 1))
        sig = np.sqrt(np.tile(Itrue, 3) + 50.0)
        I = np.tile(Itrue, 3) + rng.normal(0, 1, len(h)) * sig
        rows = "".join(f"{a:6d}{b:6d}{c:6d}{x:11.3E}{s:11.3E}{z:9.1f}\n" for (a, b, c), x, s, z in
                       zip(h, I, sig, rng.uniform(1, 360, len(h))))
        (tmp_path / f"s{i}.hkl").write_text(
            "!MERGE=FALSE\n!SPACE_GROUP_NUMBER=16\n!UNIT_CELL_CONSTANTS=30 35 40 90 90 90\n"
            + "".join(f"!ITEM_{n}={k + 1}\n" for k, n in enumerate(["H", "K", "L", "IOBS", "SIGMA(IOBS)", "ZD"]))
            + "!END_OF_HEADER\n" + rows + "!END_OF_DATA\n")
        inp += f"INPUT_FILE=s{i}.hkl\n"
    (tmp_path / "merge.inp").write_text(inp)
    run = lambda **kw: run_multiset(tmp_path / "merge.inp", tmp_path, log=lambda *a: None, prior=0.3, **kw)["m"]
    mtz_dmin = lambda: gemmi.read_mtz_file(str(tmp_path / "m_merged_iso.mtz")).resolution_high()
    r = run(anisotropy=False, d_limit=3.5)
    fit = r["cc_half_cutoff"]
    assert r["d_written"] == 3.5 and 2.6 < fit < 3.3
    assert abs(mtz_dmin() - 3.5) < 1e-6                       # 0 10 0; all data: 2.5 A
    d_lim = fit / 1.005
    r = run(anisotropy=True, d_limit=d_lim, d_limit_tol=0.01)
    assert r["d_written"] == d_lim and r["cc_half_cutoff"] == fit and d_lim <= mtz_dmin() < fit
    an = r["aniso"]
    assert "inside_hkl" not in an and r["overall_aniso"].n_uniq == an["n_inside"] == gemmi.read_mtz_file(
        str(tmp_path / "m_anisotropic.mtz")).nreflections
    ec = ellipsoidal_completeness(r["merged"]["hkl"], np.array([30., 35, 40, 90, 90, 90]), gemmi.SpaceGroup("P 2 2 2"),
                                  an["limits"], an["directions_hkl"])
    assert r["ellipsoidal_completeness"] == ec[2]
    assert np.isfinite(r["wilson_b"]) and r["wilson_dmin"] == d_lim
    r = run(anisotropy=False, d_limit=fit / 1.02, d_limit_tol=0.01)
    assert "d_written" not in r and mtz_dmin() >= fit         # the fit is the cut (such a merge costs resolution)


def test_isa_empirical_low_multiplicity():
    """Pair-dominated data (multiplicity ~1.5): too few strong reflections are measured three times, so pairs give the
    systematic-error floor (here 7%, ISa 14.3); with enough triples the multiplicity >= 3 estimate is unchanged."""
    from xpg.merging import isa_empirical
    rng = np.random.default_rng(0)
    n = 50000
    J = rng.exponential(1000.0, n)
    mult = np.where(rng.random(n) < 0.5, 2, 1)
    mult[:5] = 3
    uid = np.repeat(np.arange(n), mult)
    I = J[uid] * (1 + 0.07 * rng.standard_normal(uid.size))
    assert abs(isa_empirical(uid, I) - 1 / 0.07) < 1.4
    uid3 = np.repeat(np.arange(n), 3)
    I3 = J[uid3] * (1 + 0.07 * rng.standard_normal(uid3.size))
    m = np.bincount(uid3, I3) / 3
    sel = m[uid3] >= np.quantile(m[uid3], 0.98)
    ref = 1 / (1.4826 * np.median(np.abs((I3[sel] - m[uid3][sel]) / m[uid3][sel] * np.sqrt(1.5))))
    assert isa_empirical(uid3, I3) == ref


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


def _lattice_pipe(tmp_path, monkeypatch, results, r_all):
    """LatticeStage with one extra lattice (toy reflections) and a stub joint merge returning r_all; the stub
    records run_multiset's keyword arguments."""
    import gemmi
    from types import SimpleNamespace
    import xpg.multiset
    from xpg import stage_lattices
    cell = np.array([40., 50., 60., 90, 90, 90])
    monkeypatch.setattr(stage_lattices, "lattice_candidates", lambda c, max_obliq: [SimpleNamespace(
        bravais="oP", point_group="mmm", cell=cell, P=np.eye(3, dtype=int), system="o")])
    calls = []
    monkeypatch.setattr(xpg.multiset, "run_multiset", lambda *a, **kw: calls.append(kw) or {"xpg_all_lattices": r_all})
    monkeypatch.delenv("XPG_LATTICES_PRIMARY", raising=False)
    hkl = np.array([(h, k, l) for h in range(6) for k in range(6) for l in range(6) if h + k + l])
    n = len(hkl)
    I = np.random.default_rng(0).exponential(100.0, n)
    refl = dict(hkl=hkl, I=I, sig=np.full(n, 5.0), n_over=np.zeros(n, int), peak=np.ones(n), x=np.zeros(n),
                y=np.zeros(n), z=np.zeros(n), rlp=np.ones(n))

    class Pipe(stage_lattices.LatticeStage):
        wd, anisotropy, decisions, logs = tmp_path, False, {}, []
        p = SimpleNamespace(lattice_tolerance=3.0, minpk=0.0, export_legacy=False, friedels_law=True)
        _main_hkl, _main_I, _main_sig = hkl, I, np.full(n, 5.0)
        st = SimpleNamespace(results=dict(results, extra_refl={0: refl}, n_obs_merged=n),
                             lattice=SimpleNamespace(bravais="oP", point_group="mmm"),
                             exp=SimpleNamespace(geom=SimpleNamespace(copy=lambda: None)),
                             indexing=SimpleNamespace(extra_lattices=[SimpleNamespace(exp=SimpleNamespace(
                                 crystal=Crystal.from_Astar(np.diag(1 / cell[:3]), "o")))]))
        log = lambda self, m: self.logs.append(m)
        decide = lambda self, name, choice, **kw: self.decisions.update({name: choice})
        _correct_extra_lattice = lambda self, r, exp_k, sg, label: (r["I"], r["sig"], np.ones(len(r["I"]), bool))
        _write_unmerged = lambda self, *a, **kw: None
        _lattice_report = lambda self, files, cell, sg: 1
        _plausible_merge_limit = lambda self, n_all: None
    return Pipe(), calls, cell, gemmi.SpaceGroup("P 21 21 21")


def test_lattice_merge_main_lattice_excluded(tmp_path, monkeypatch):
    """The Delta-CC1/2 rule may drop the main lattice when an extra lattice is the reference: the merge then holds
    extra lattices only and is not promoted as 'all lattices merged' (final pass), nor kept with a wrong reason
    ('costs resolution', triage)."""
    r_all = dict(cc_half_cutoff=1.8, n_obs=5000, excluded=["xpg_unmerged.mtz"])
    for name in ("merged_iso.mtz", "anisotropic.mtz", "unmerged.mtz"):
        (tmp_path / f"xpg_all_lattices_{name}").write_text("all")
        (tmp_path / f"xpg_{name}").write_text("main")
    pipe, calls, cell, sg = _lattice_pipe(tmp_path, monkeypatch, dict(cc_half_cutoff=2.0, cc_half_fit=2.0,
                                                                       limit_by="cc_half"), r_all)
    assert pipe._process_extra_lattices(None, cell, sg, triage=True) == (
        False, "main lattice excluded from the merge (Delta-CC1/2)")
    pipe._process_extra_lattices(None, cell, sg)
    assert pipe.decisions["lattice merge"] == "main lattice only (main lattice excluded from the merge)"
    assert (tmp_path / "xpg_merged_iso.mtz").read_text() == "main"
    assert not (tmp_path / "xpg_lattice1_merged_iso.mtz").exists()
    assert calls[-1]["d_limit"] is None                       # CC1/2-limited main lattice: merge at its own fit


def test_lattice_merge_promoted_at_main_limit(tmp_path, monkeypatch):
    """Main lattice limited by completeness (fit 1.6 A, limit 1.9 A): the merge, judged on its fit, is written to the
    main lattice's limit (run_multiset d_limit) and, once promoted, the statistics reported are the merge's on the
    main lattice's shells, so the summary, the primary MTZ and the CIF agree: every column of Table 1, the
    anisotropic ones (xpg_anisotropic.mtz is the merge's), the ellipsoidal completeness and Wilson B included. The
    merge is cut at the limit when its fit is within the final-pass verdict's tolerance of it, beyond which it costs
    resolution. No main-lattice limit at all: the 'implausible' verdict is logged, not a TypeError."""
    import inspect
    from types import SimpleNamespace
    from xpg.analysis import TwinningReport
    sh = lambda cc: SimpleNamespace(cc_half=cc)
    tw = lambda b: TwinningReport(b, 2.0, 0.74, 0.49, 0.33, [], 1.9, 1.9)
    r_all = dict(cc_half_cutoff=1.58, n_obs=5000, excluded=[], d_written=1.9, shells_compare=[sh(0.9), sh(0.5)],
                 overall_compare=sh(0.8), shells=[sh(0.9), sh(0.5), sh(0.1)], overall=sh(0.7),
                 aniso=dict(limits=[1.6, 1.7, 1.8]), shells_aniso=[sh(0.9), sh(0.6)], overall_aniso=sh(0.85),
                 ellipsoidal_completeness=0.95, merged=dict(hkl=np.zeros((3, 3), int)), wilson_b=24.0, wilson_dmin=1.9)
    for name in ("merged_iso.mtz", "anisotropic.mtz", "unmerged.mtz"):
        (tmp_path / f"xpg_all_lattices_{name}").write_text("all")
    main = dict(cc_half_cutoff=1.9, cc_half_fit=1.6, limit_by="completeness", shells_cut=[sh(0.9), sh(0.5)],
                aniso=dict(limits=[1.7, 1.8, 1.9]), shells_aniso=[sh(0.8)], overall_aniso=sh(0.8),
                ellipsoidal_completeness=0.9, merged=dict(hkl=np.ones((2, 3), int)), twinning=tw(20.0))
    pipe, calls, cell, sg = _lattice_pipe(tmp_path, monkeypatch, main, r_all)
    pipe._process_extra_lattices(None, cell, sg, triage=True)
    assert calls[-1]["d_limit"] is None                       # triage: verdict only
    pipe._process_extra_lattices(None, cell, sg)
    res = pipe.st.results
    assert calls[-1]["d_limit"] == 1.9 and res["primary_mtz"] == "all lattices"
    # written to the limit when the merge's fit is within the verdict tolerance of the main FIT: none here (1.01 x 1.6
    # A is inside the 1.9 A limit); a fit next to the limit opens the window up to 1.01 x the fit
    from xpg.stage_lattices import _limit_tol
    tol = inspect.signature(pipe._merge_verdict).parameters["tol"].default
    assert calls[-1]["d_limit_tol"] == _limit_tol(1.9, 1.6) == 0.0 and _limit_tol(1.9, None) == tol
    assert abs((1 + _limit_tol(1.9, 1.89)) * 1.9 - (1 + tol) * 1.89) < 1e-12
    assert pipe._merge_verdict(1.0101 * 1.9, 1.6, 5000)[1] == "costs resolution"
    assert res["shells_cut"] is r_all["shells_compare"] and res["overall_cut"] is r_all["overall_compare"]
    assert res["overall"] is r_all["overall"] and res["cc_half_cutoff"] == 1.9
    assert all(res[k] is r_all[k] for k in ("aniso", "shells_aniso", "overall_aniso", "merged"))
    assert res["ellipsoidal_completeness"] == 0.95 and res["twinning"] == tw(24.0)
    res.update(main, primary_mtz=None)               # a merge without a cut-off surface: no anisotropic columns
    no_surf = ("aniso", "shells_aniso", "overall_aniso", "ellipsoidal_completeness")
    pipe._promote_lattice_merge(1.58, 1.6, 5000, r_all={k: v for k, v in r_all.items() if k not in no_surf})
    assert res["primary_mtz"] == "all lattices" and all(res[k] is None for k in no_surf)
    res["cc_half_fit"], r_all["cc_half_cutoff"] = None, 2.2      # no fit: the verdict compares with the 1.9 A limit
    assert pipe._process_extra_lattices(None, cell, sg, triage=True) == (False, "costs resolution")
    pipe._plausible_merge_limit = lambda n_all: 2.0
    pipe._promote_lattice_merge(1.5, None, 3000)
    assert res["primary_mtz"] == "first lattice (all-lattice merge implausible)"


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


def test_mr_alternatives_from_latest_point_group_call(tmp_path):
    """xpg mr without audit merges takes the alternatives from the run's events: the latest point-group decision is
    the operative one. A close pass-1 call decided again with confidence on the re-integrated data: nothing to
    search; the later call a close one itself: its alternatives, not those of pass 1."""
    import gemmi
    from types import SimpleNamespace
    from xpg.events import EventLog
    from xpg.mr import _close_call_alternatives
    from xpg.pipeline import Pipeline
    sg = gemmi.SpaceGroup("P 3 2 1")
    for final_low, want in ((False, []), (True, ["P 6 2 2"])):
        d = tmp_path / str(final_low)
        d.mkdir()
        pipe = SimpleNamespace(st=SimpleNamespace(results={}), events=EventLog(d), log=lambda *a: None)
        Pipeline.decide(pipe, "point group", "hP 3", margin=0.5, unit="posterior", low=True,
                        alternatives=["hP 32 (0.30)", "hP 312 (0.15)"])
        Pipeline.decide(pipe, "point group", "hP 32", margin=0.6 if final_low else 0.97, unit="posterior",
                        low=final_low, alternatives=["hP 622 (0.30)"] if final_low else ["hP 3 (0.02)"],
                        note="re-decided on the re-integrated data (earlier call hP 3, posterior 0.50)")
        pipe.events.f.close()
        assert _close_call_alternatives(d / "events.jsonl", sg) == want, final_low


def test_earlier_scale_outputs_removed(tmp_path, monkeypatch):
    """The optional outputs describe the SCALE run that wrote xpg_unmerged.mtz: hidden-twin subgroup files
    (xpg_unmerged_<symbol>.mtz), detwinned data (xpg_detwinned_*) and anisotropic data (xpg_anisotropic.mtz, analysis
    off or failed) of an earlier run that this one does not write are removed (after the point group was decided
    again they are in another setting); this run's own are written.
    Nothing else goes: xpg_unmerged.mtz, audit merges, extra-lattice outputs, other xpg_unmerged_* names,
    subdirectories."""
    from pathlib import Path
    from types import SimpleNamespace
    import gemmi
    import xpg.analysis
    from xpg import stage_scale

    def merged_out(md, cell, sg, dcut, wl, stem, anisotropy=True, **kw):
        Path(f"{stem}_merged_iso.mtz").write_text("new")
        if anisotropy:
            Path(f"{stem}_anisotropic.mtz").write_text("new")
            return dict(limits=[2.0] * 3, inside_hkl=np.zeros((1, 3), int))
    monkeypatch.setattr(stage_scale, "write_merged_outputs", merged_out)
    monkeypatch.setattr(xpg.analysis, "detwin", lambda hkl, I, sig, sg, op, a: (I, sig, np.ones(len(I), bool)))

    class Pipe(stage_scale.ScaleStage):
        wd, anisotropy, _hidden_twin, lines = tmp_path, False, None, []
        p = SimpleNamespace(free_r_reference=None)
        st = SimpleNamespace(results={})
        log = lambda self, m: self.lines.append(m)
        _write_unmerged = lambda self, path, *a, **kw: path.write_text("new")
    stale = ["xpg_unmerged_P61.mtz", "xpg_unmerged_C2221.mtz", "xpg_unmerged_P31.mtz",
             "xpg_detwinned_merged_iso.mtz", "xpg_detwinned_anisotropic.mtz", "xpg_anisotropic.mtz"]
    keep = ["xpg_unmerged.mtz", "xpg_C2221_merged_iso.mtz", "xpg_lattice2_unmerged.mtz", "xpg_all_lattices_unmerged.mtz",
            "xpg_unmerged_notes.mtz", "xpg_unmerged_P61.json", "sub/xpg_unmerged_P61.mtz",
            "sub/xpg_detwinned_merged_iso.mtz"]
    (tmp_path / "sub").mkdir()
    for n in stale + keep:
        (tmp_path / n).write_text("old")
    pipe = Pipe()
    r, act = {k: np.zeros((2, 3)) for k in ("hkl", "x", "y", "z", "rlp")}, np.ones(2, bool)
    pipe._hidden_twin = [(gemmi.SpaceGroup("P 31"), "-h,-k,l", 0.45, 0.44)]       # this run: P 31 (rewritten)
    pipe._write_subgroup_files(None, r, act, np.ones(2), np.ones(2))
    assert sorted(f.name for f in tmp_path.glob("xpg_unmerged_*.mtz")) == ["xpg_unmerged_P31.mtz",
                                                                            "xpg_unmerged_notes.mtz"]
    assert (tmp_path / "xpg_unmerged_P31.mtz").read_text() == "new"
    assert "removed xpg_unmerged_C2221.mtz, xpg_unmerged_P61.mtz" in pipe.lines[0]
    pipe._hidden_twin = None                                                         # this run: none
    pipe._write_subgroup_files(None, r, act, np.ones(2), np.ones(2))
    assert not (tmp_path / "xpg_unmerged_P31.mtz").exists()
    merged = dict(hkl=np.zeros((2, 3), int), I=np.ones(2), sig=np.ones(2))
    pipe._write_merged(merged, None, gemmi.SpaceGroup("P 3 2 1"), None, 1.0, True)  # no twinning, no anisotropy
    assert not list(tmp_path.glob("xpg_detwinned_*")) and not (tmp_path / "xpg_anisotropic.mtz").exists()
    assert any(m.startswith("  SCALE: removed xpg_anisotropic.mtz (") for m in pipe.lines)
    assert "aniso" not in pipe.st.results
    pipe.st.results["twinning"] = SimpleNamespace(twin_laws=[("-h,-k,l", 0.3, 0.2, 100, 0.21)])
    (tmp_path / "xpg_detwinned_anisotropic.mtz").write_text("old")
    pipe.anisotropy = True                                                           # this run's anisotropic data
    pipe._write_merged(merged, None, gemmi.SpaceGroup("P 3 2 1"), None, 1.0, True)  # alpha 0.2: detwinned
    assert (tmp_path / "xpg_anisotropic.mtz").read_text() == "new" and pipe.st.results["aniso"] == dict(limits=[2.] * 3)
    assert [f.name for f in tmp_path.glob("xpg_detwinned_*")] == ["xpg_detwinned_merged_iso.mtz"]
    assert (tmp_path / "xpg_detwinned_merged_iso.mtz").read_text() == "new"
    assert all((tmp_path / n).read_text() == "old" for n in keep)


def test_mr_hidden_twin_subgroups_from_summary(tmp_path, monkeypatch):
    """xpg mr searches the hidden-twin subgroups the run's summary lists (pathologies.hidden_twin: the SCALE run that
    wrote the final data), each from its xpg_unmerged_<SG>.mtz when that exists; a file of an earlier run left in
    the directory (P 61, before the point group was decided again) is not searched: its group would be merged from
    the final indices as they stand. Every such file only without a summary."""
    import json
    import gemmi
    import pytest
    import xpg.matthews
    import xpg.mr

    def mtz(path, sg):
        m = gemmi.Mtz(with_base=True)
        m.spacegroup = gemmi.SpaceGroup(sg)
        m.set_cell_for_all(gemmi.UnitCell(60, 60, 90, 90, 90, 120))
        m.add_dataset("x")
        m.set_data(np.array([[1, 0, 0]], np.float32))
        m.write_to_file(str(path))

    class Stop(Exception):
        pass
    merged = []
    monkeypatch.setattr(xpg.mr, "prepare_model", lambda model, out: (False, 100, 1))
    monkeypatch.setattr(xpg.mr, "merge_in", lambda run_dir, a, out, log: merged.append(a) or out)
    monkeypatch.setattr(xpg.matthews, "model_components", lambda model: [("A", "protein", "M" * 100)])
    monkeypatch.setattr(xpg.matthews, "write_components", lambda comps, path: path)
    monkeypatch.setattr(xpg.matthews, "mw_of_components", lambda comps: (_ for _ in ()).throw(Stop()))
    mtz(tmp_path / "xpg_merged_iso.mtz", "P 31 2 1")
    mtz(tmp_path / "xpg_unmerged_P31.mtz", "P 31")
    mtz(tmp_path / "xpg_unmerged_P61.mtz", "P 61")               # an earlier run's
    ht = dict(space_group="P 31", law="h,-h-k,-l", alpha_h=0.45, alpha_britton=0.44, alpha_l=0.4)
    for summ, want in ((dict(pathologies=dict(hidden_twin=[ht])), ["P 31"]), ({}, []),
                       (dict(pathologies=dict(hidden_twin=[dict(ht, space_group="P 32")])), []),
                       (None, ["P 31", "P 61"])):
        (tmp_path / "xpg_summary.json").unlink(missing_ok=True)
        if summ is not None:
            (tmp_path / "xpg_summary.json").write_text(json.dumps(summ))
        merged.clear()
        with pytest.raises(Stop):
            xpg.mr.run_mr(tmp_path, "model.pdb", None, log=lambda *a: None)
        assert merged == want, summ


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


def test_symmetry_decision_phi_drift():
    """A systematic error that drifts along the sweep is shared by repeats (measured a few degrees apart) but not by
    symmetry mates (far apart): the expected CC of each element is matched to its pairs' rotation separations, so true
    symmetry under drift (twinned or not) is kept; without drift the single expected CC is used unchanged."""
    from symsim import CELL, decide, observations
    from xpg.lattice import lattice_candidates
    from xpg.simulate import SimSpec
    from xpg.symmetry import score_symmetry
    cases = [("622", dict(space_group="P 61 2 2"), dict(drift=1.5, sweep=90.0), 0),
             ("3", dict(space_group="P 31", twin_law="-h,-k,l", twin_fraction=0.3), dict(drift=1.0, sweep=90.0), 1),
             ("3", dict(space_group="P 31", twin_law="k,h,-l", twin_fraction=0.35), dict(drift=1.0, sweep=90.0), 2)]
    for truth, sk, kw, sd in cases:
        h, I, sig, s, phi = observations(SimSpec(cell=CELL, structure_seed=1000 + sd, **sk), seed=sd, mult=3.0, **kw)
        pg, p = decide(h, I, sig, s, phi)
        assert pg.endswith(" " + truth), (sk, kw, pg, p)
    # no drift (random phi): the decay test stays closed, also where one small cell's jackknife variance came out far
    # too small (seed 5 passed the test at z 5 before the variances of the test were floored)
    cands = lattice_candidates(CELL, max_obliq=3.0)
    for sd in range(8):
        h, I, sig, s, phi = observations(SimSpec(cell=CELL, structure_seed=1000 + sd, space_group="P 31 2 1"),
                                         seed=sd, mult=3.0, sys_err=0.3)
        _, ops, cc_true = score_symmetry(h, I, sig, s, phi, cands, verbose=False)
        assert all(o.cc_exp == cc_true for o in ops.values() if np.isfinite(o.cc)), sd


def test_symmetry_decay_model_multiplicity(monkeypatch):
    """With the separation model forced on for data without drift, each true element's expected CC is the model's
    own: the single-observation CC extrapolated to the multiplicities of the element's pairs, single observations
    included, as the elements' CC is. cc_true (reflections with >= 2 observations) sits above it, the known step where
    the decay test opens; scaling cc_true by the model's decay instead carried that bias into every drifted data set
    (expected CC up 0.02-0.05, true groups' posteriors down, a twinned 3-fold under drift lost to P 1)."""
    import xpg.symmetry as sym
    from symsim import CELL, observations
    from xpg.lattice import lattice_candidates
    from xpg.simulate import SimSpec
    monkeypatch.setattr(sym, "SEP_DECAY_Z", -np.inf)
    monkeypatch.setattr(sym, "SEP_MIN_DECAY", -np.inf)
    h, I, sig, s, phi = observations(SimSpec(cell=CELL, structure_seed=1005, space_group="P 31 2 1"), seed=5,
                                     mult=3.0, sys_err=0.3)
    res, ops, cc_true = sym.score_symmetry(h, I, sig, s, phi, lattice_candidates(CELL, max_obliq=3.0), verbose=False)
    true = [o for o in ops.values() if np.isfinite(o.cc) and o.cc > 0.5]
    assert len(true) == 5 and res[0].cand.point_group == "32"
    assert all(abs(o.cc_exp - o.cc) < 0.02 and o.cc_exp < cc_true - 0.03 for o in true), \
        (cc_true, [(o.cc, o.cc_exp) for o in true])


def _redecide_pipe(pass1, pass1_sg, low=True, truth="P 6 2 2", turn=0.1):
    """SCALE + INDEX mixins on synthetic observations of ``truth`` (symsim: the indexing cell is CELL) after a pass-1
    call ``pass1`` applied as pass 1 does (_apply_lattice: indices and crystal in its conventional setting, then
    refinement turned the crystal by ``turn`` deg), recorded with posterior 0.16 (low) or 0.95, plus results of the
    old setting: reference reindexing done, tNCS modulation vector, a frame filter. Decision events are collected in
    pipe.ev. Returns the pipe, the indexing-cell indices and the gate's inputs (observations, s, reflections in the
    current setting)."""
    import gemmi
    from types import SimpleNamespace
    from symsim import CELL, observations
    from xpg.lattice import lattice_candidates
    from xpg.pipeline import Pipeline
    from xpg.simulate import SimSpec
    from xpg.stage_index import IndexStage
    from xpg.stage_scale import ScaleStage
    h, I, sig, s, phi = observations(SimSpec(cell=CELL, structure_seed=1000, space_group=truth), seed=0)
    g = SimpleNamespace(z_to_phi=lambda z: np.asarray(z, float))
    g.copy = lambda: g

    class Pipe(ScaleStage, IndexStage):
        decide, _write_audit_merges = Pipeline.decide, Pipeline._write_audit_merges
        chiral = True
        log = lambda self, *a: self.lines.append(" ".join(map(str, a)))
    pipe = Pipe()
    pipe.lines, pipe.ev = [], []
    pipe.events = SimpleNamespace(emit=lambda t, **k: pipe.ev.append((t, k)))
    pipe.p = SimpleNamespace(space_group_number=0, unit_cell=None, reidx=None, reference_data_set=None,
                             lattice_tolerance=3.0, isig_limit_symmetry=6.0)
    exp = Experiment(g, Crystal(U=np.eye(3), cell=CELL))
    pipe.st = SimpleNamespace(results={}, exp=exp, indexing=SimpleNamespace(exp=exp), lattice=None, space_group=None)
    cand = next(c for c in lattice_candidates(CELL, max_obliq=3.0) if f"{c.bravais} {c.point_group}" == pass1)
    refl = dict(hkl=h.copy())
    pipe._apply_lattice(cand, gemmi.SpaceGroup(pass1_sg), refl)
    t = np.radians(turn)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    pipe.st.exp = Experiment(g, Crystal.from_Astar(Rx @ pipe.st.exp.crystal.Astar, cand.system, space_group=pass1_sg))
    pipe.decide("point group", pass1, margin=0.16 if low else 0.95, unit="posterior", low=low)
    pipe.decide("frame filter", "3.0% of observations removed", low=False)
    pipe.st.results.update(reference_applied=True, tncs_modulation=([0.5, 0.0, 0.0], 0.3), removed_image_groups=[7])
    pipe.lines.clear()
    return pipe, h, dict(hkl=refl["hkl"].copy(), I=I, sig=sig, z=phi), s, refl


def test_point_group_redecided_on_reintegrated_data(tmp_path, monkeypatch):
    """A close pass-1 call (oC 222 on P 6 2 2 data: on crowded data integrated with the default spot widths, leaked
    neighbour intensity reads as twinning and the posterior spreads over subgroups) is decided again on the final
    SCALE's observations, by the same scorer in the primitive basis of the indexing cell: hP 622 replaces it. The
    observations and the crystal are reindexed to the hexagonal setting (constrained cell), exactly through the
    lattice applied in pass 1 even after a 3 deg orientation change since indexing (beyond what a map solved against
    the indexing crystal tolerates), each decision is recorded once with the final values (the earlier call in its
    note), results of the old setting are dropped, every record is the latest decision event of its name (records of
    the earlier run kept, as from pass 1 to pass 2), and the decision audit merges the final call's alternatives
    from the final indices."""
    import gemmi
    import xpg.mr
    from xpg import stage_scale
    from xpg.models import d_spacing_from_exp
    real, top = stage_scale.score_symmetry, []

    def close(*a, **k):     # the real scorer; posteriors flattened (order kept) to give the final call alternatives
        res, ops, cc = real(*a, **k)
        top.append((f"{res[0].cand.bravais} {res[0].cand.point_group}", res[0].prob))
        for k_, x in enumerate(res):
            x.prob = (0.6, 0.25, 0.15)[k_] if k_ < 3 else 0.0
        return res, ops, cc
    monkeypatch.setattr(stage_scale, "score_symmetry", close)
    from xpg.symops import centring_reindex
    pipe, h, obs, s, refl = _redecide_pipe("oC 222", "C 2 2 2", turn=3.0)
    st = pipe.st
    assert centring_reindex(np.linalg.solve(st.indexing.exp.crystal.Astar, st.exp.crystal.Astar), st.space_group) is None
    red0 = st.exp.crystal.Astar @ np.asarray(st.lattice.P, float).T       # the turned crystal, indexing-cell axes
    assert pipe._redecide_point_group(obs, s, refl)
    assert top[0][0] == "hP 622" and top[0][1] > 0.99
    assert st.lattice.bravais == "hP" and st.space_group.point_group_hm() == "622" and st.exp.crystal.system == "h"
    assert st.exp.crystal.space_group == st.space_group.xhm()
    assert np.array_equal(refl["hkl"], h @ st.lattice.P)              # through the primitive basis, exactly
    # the refined orientation is kept (not the indexing one, 3 deg away): the same reciprocal axes of the indexing
    # cell (exactly hexagonal, so the constraints of either system leave them)
    red1 = st.exp.crystal.Astar @ np.asarray(st.lattice.P, float).T
    cos = np.sum(red0 * red1, 0) / np.linalg.norm(red0, axis=0) / np.linalg.norm(red1, axis=0)
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 0.01 and np.allclose(red1, red0, rtol=1e-6)
    assert np.allclose(1 / d_spacing_from_exp(refl["hkl"], st.exp), s, rtol=1e-6)
    pg = [d for d in st.results["decisions"] if d["name"] == "point group"]
    assert len(pg) == 1 and pg[0]["choice"] == "hP 622" and pg[0]["margin"] == 0.6 and pg[0]["low_margin"]
    assert pg[0]["note"] == "re-decided on the re-integrated data (earlier call oC 222, posterior 0.16)"
    sgr = [d for d in st.results["decisions"] if d["name"] == "space group"]
    assert len(sgr) == 1 and sgr[0]["choice"] == st.space_group.xhm()
    assert st.space_group.xhm() in st.results["space_group_alternatives"]
    assert any(ln.startswith("  DECISION point group: hP 622") and "earlier call oC 222" in ln for ln in pipe.lines)
    assert pipe.lines[-1].endswith("observations selected again in the new point group")
    assert "reference_applied" not in st.results and st.results["removed_image_groups"] == [7]
    assert st.results.get("tncs_modulation") != ([0.5, 0.0, 0.0], 0.3)
    latest = {k["name"]: k for t, k in pipe.ev if t == "decision"}
    assert latest == {d["name"]: d for d in st.results["decisions"]} and "frame filter" in latest
    # audit: the final call's alternatives, each reached from the final indices on the lattice of the indexing cell
    alts = st.results["point_group_alternatives"]
    assert [a["label"] for a in alts] == [x.rsplit(" (", 1)[0] for x in pg[0]["alternatives"]] and len(alts) == 2
    (tmp_path / "xpg_unmerged.mtz").write_text("")
    pipe.wd, merged = tmp_path, []
    monkeypatch.setattr(xpg.mr, "merge_in", lambda wd, hm, out, log, reindex, cell: merged.append((hm, reindex)))
    pipe._write_audit_merges()
    assert [hm for hm, _ in merged] == [gemmi.SpaceGroup(a["space_group"]).hm for a in alts]
    for a, (hm, Ti) in zip(alts, merged):
        hp = refl["hkl"] @ np.asarray(Ti, float).T
        assert np.allclose(hp, np.rint(hp)) and np.allclose(np.linalg.norm(hp @ a["Astar"].T, axis=1), s, rtol=1e-6)
    assert sum("AUDIT: close point-group call (hP 622, posterior 0.60)" in ln for ln in pipe.lines) == 2


def test_point_group_kept_when_reintegrated_data_agree():
    """The final data confirm the close pass-1 call (hP 622 on P 6 2 2 data): one log line, nothing else changes --
    indices, crystal, lattice, space group and every result, the earlier record included (the run stays identical
    to one without the second call). Also with a user space group incompatible with the lattice (P 21 21 21: pass 1
    ignored it and the scorer made the call). The comparison is by rotation group in the primitive basis: every
    candidate's operators give the same set as its space group's, and the six mC 2 settings of a hexagonal lattice,
    one label, are six different groups."""
    import copy
    from symsim import CELL
    from xpg.lattice import lattice_candidates
    from xpg.stage_scale import _rotations_in_basis
    from xpg.symops import rot_matrices
    cands = lattice_candidates(CELL, max_obliq=3.0)
    assert all(_rotations_in_basis(c.P, c.ops) == _rotations_in_basis(c.P, rot_matrices(c.space_group)) for c in cands)
    assert len({frozenset(_rotations_in_basis(c.P, c.ops)) for c in cands if c.bravais == "mC"}) == 6
    for user_sg in (0, 19):
        pipe, h, obs, s, refl = _redecide_pipe("hP 622", "P 6 2 2")
        pipe.p.space_group_number = user_sg
        st = pipe.st
        before, hkl0, objs = copy.deepcopy(st.results), refl["hkl"].copy(), (st.exp, st.lattice, st.space_group)
        assert not pipe._redecide_point_group(obs, s, refl)
        assert np.array_equal(refl["hkl"], hkl0) and all(a is b for a, b in zip((st.exp, st.lattice, st.space_group),
                                                                                objs))
        assert st.results == before
        assert pipe.lines == ["  SCALE: point group hP 622 kept: the re-integrated data give the same group (posterior "
                              "1.00 on them, 0.16 in the earlier call)"], user_sg


def test_point_group_not_redecided(monkeypatch):
    """A confident pass-1 call (posterior >= 0.9), a call already made again on re-integrated data (its note: e.g. a
    SCALE-only run on the pass-2 integration followed by a second pass), a space group given by the user that the
    lattice allows (P 6 2 2), a reindexing given by the user, no recorded call (user cell and space group applied at
    indexing) or no indexing result: the final SCALE runs no second call -- the scorer is not called, nothing is
    logged or changed."""
    from xpg import stage_scale

    def never(*a, **k):
        raise AssertionError("scorer called")
    monkeypatch.setattr(stage_scale, "score_symmetry", never)
    for case in ("confident", "re-decided", "space group", "reindexing", "no record", "no indexing"):
        pipe, h, obs, s, refl = _redecide_pipe("oC 222", "C 2 2 2", low=case != "confident")
        if case == "re-decided":
            pipe.st.results["decisions"][0]["note"] = "re-decided on the re-integrated data (earlier call mC 2, " \
                                                      "posterior 0.40)"
        elif case == "space group":
            pipe.p.space_group_number = 177
        elif case == "reindexing":
            pipe.p.reidx = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]], float)
        elif case == "no record":
            pipe.st.results["decisions"] = []
        elif case == "no indexing":
            pipe.st.indexing = None
        before, hkl0 = dict(pipe.st.results), refl["hkl"].copy()
        assert not pipe._redecide_point_group(obs, s, refl), case
        assert pipe.lines == [] and np.array_equal(refl["hkl"], hkl0) and pipe.st.results == before, case


def test_point_group_redecided_between_mc_settings(monkeypatch):
    """The gate compares rotation groups, not labels: on P 6 2 2 data (hexagonal lattice) pass 1 applied one of the
    six mC 2 settings (A, a close call); the re-integrated data give another (B, a 2-fold along another axis), the
    same label 'mC 2' (and space group C 1 2 1) but another group: decided again, observations and crystal in B's
    setting, exactly through the primitive basis."""
    from symsim import CELL
    from xpg import stage_scale
    from xpg.lattice import lattice_candidates
    from xpg.stage_scale import _rotations_in_basis
    rot = lambda c: _rotations_in_basis(c.P, c.ops)
    mcs = [c for c in lattice_candidates(CELL, max_obliq=3.0) if c.bravais == "mC"]
    A, B = mcs[0], mcs[1]
    assert A.space_group == B.space_group and rot(A) != rot(B)
    real = stage_scale.score_symmetry

    def forced(*a, **k):       # the real scorer, setting B put first
        res, ops, cc = real(*a, **k)
        j = next(i for i, x in enumerate(res) if x.cand.bravais == "mC" and rot(x.cand) == rot(B))
        res = [res[j]] + res[:j] + res[j + 1:]
        res[0].prob = 0.8
        return res, ops, cc
    monkeypatch.setattr(stage_scale, "score_symmetry", forced)
    pipe, h, obs, s, refl = _redecide_pipe("mC 2", A.space_group)
    st = pipe.st
    assert rot(st.lattice) == rot(A)
    assert pipe._redecide_point_group(obs, s, refl)
    assert rot(st.lattice) == rot(B) and st.space_group.xhm() == "C 1 2 1"
    assert np.array_equal(refl["hkl"], h @ st.lattice.P)
    pg = [d for d in st.results["decisions"] if d["name"] == "point group"]
    assert len(pg) == 1 and pg[0]["choice"] == "mC 2" and pg[0]["margin"] == 0.8 and "earlier call mC 2" in pg[0]["note"]

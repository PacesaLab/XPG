import zlib

import numpy as np
import gemmi
import pytest

from xpg.config import parse_xds_inp_text, params_from_dict
from xpg.geometry import Geometry, cell_to_B, cell_from_Astar, predict_reflections
from xpg.io.cbf import decode_byte_offset, encode_byte_offset, read_cbf, write_cbf
from xpg.lattice import lattice_candidates
from xpg.indexing import niggli_reduce
from xpg.symops import to_asu


def test_cbf_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.poisson(3, (37, 53)).astype(np.int32)
    a[3, 4] = 1_000_000
    a[5, 6] = -1
    a[7, 8] = 40_000
    a[9, 10] = -2_000_000
    assert np.array_equal(decode_byte_offset(encode_byte_offset(a), a.size).reshape(a.shape), a)
    write_cbf(tmp_path / "x.cbf", a, dict(wavelength=1.1, distance=150, osc=0.2))
    b, h = read_cbf(tmp_path / "x.cbf")
    assert np.array_equal(a, b)
    assert abs(h["wavelength"] - 1.1) < 1e-6 and abs(h["angle_increment"] - 0.2) < 1e-6


def test_input_conversion():
    txt = """DETECTOR=EIGER NX=4150 NY=4371 QX=0.075  QY=0.075 !EIGER 16M
ORGX= 1907.1 ORGY= 2188.2  ! comment
UNTRUSTED_RECTANGLE=    0 4151    513  553
UNTRUSTED_RECTANGLE=    0 4151   1064 1104
STRONG_PIXEL=6
DATA_RANGE= 1 3600
"""
    p = params_from_dict(parse_xds_inp_text(txt))
    assert p.nx == 4150 and p.qy == 0.075 and p.orgy == 2188.2
    assert len(p.untrusted_rectangles) == 2 and p.strong_pixel == 6
    p2 = params_from_dict(parse_xds_inp_text("SIGNAL_PIXEL= 4.5\nDATA_RANGE=1 10"))
    assert p2.strong_pixel == 4.5


def test_to_asu_matches_gemmi():
    sg = gemmi.SpaceGroup("P 61 2 2")
    rng = np.random.default_rng(1)
    hkl = rng.integers(-15, 15, (3000, 3))
    hasu, isym, _ = to_asu(hkl, sg)
    asu, ops = gemmi.ReciprocalAsu(sg), sg.operations()
    for i in range(0, 3000, 97):
        ref, ref_isym = asu.to_asu(list(map(int, hkl[i])), ops)
        assert list(hasu[i]) == list(ref) and isym[i] == ref_isym


PRIM = {"P": np.eye(3), "C": np.array([[.5, -.5, 0], [.5, .5, 0], [0, 0, 1]]).T,
        "I": np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]]).T,
        "F": np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]).T,
        "R": np.array([[2 / 3, 1 / 3, 1 / 3], [-1 / 3, 1 / 3, 1 / 3], [-1 / 3, -2 / 3, 1 / 3]]).T}


@pytest.mark.parametrize("name,cell,cen", [
    ("aP", (40, 50, 60, 80, 95, 105), "P"), ("mP", (40, 50, 60, 90, 100, 90), "P"),
    ("mC", (118.3, 58.6, 79.4, 90, 104.7, 90), "C"), ("oP", (61.2, 72.5, 84.3, 90, 90, 90), "P"),
    ("oC", (50, 90, 70, 90, 90, 90), "C"), ("oI", (50, 60, 70, 90, 90, 90), "I"),
    ("oF", (60, 80, 100, 90, 90, 90), "F"), ("tP", (79.1, 79.1, 38.0, 90, 90, 90), "P"),
    ("tI", (100, 100, 60, 90, 90, 90), "I"), ("hP", (100, 100, 50, 90, 90, 120), "P"),
    ("hR", (80, 80, 150, 90, 90, 120), "R"), ("cP", (70, 70, 70, 90, 90, 90), "P"),
    ("cI", (100, 100, 100, 90, 90, 90), "I"), ("cF", (150, 150, 150, 90, 90, 90), "F")])
def test_bravais_from_noisy_reduced_cell(name, cell, cen):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    R = np.linalg.inv(cell_to_B(cell))
    Rp = (R.T @ PRIM[cen]).T * (1 + rng.normal(scale=3e-4, size=(3, 3)))   # refined-cell noise
    while True:
        M = rng.integers(-1, 2, (3, 3))
        if round(abs(np.linalg.det(M))) == 1:
            break
    Rn, _ = niggli_reduce(M @ Rp)
    c = [x for x in lattice_candidates(cell_from_Astar(np.linalg.inv(Rn))) if x.penalty < 1.0]
    best = max(c, key=lambda x: len(x.ops))
    assert best.bravais == name
    if name[0] == "m":
        assert best.cell[4] >= 90.0          # ITA convention for monoclinic cells
    assert np.linalg.det(best.P) > 0


def _geom():
    return Geometry(wavelength=1.0, beam=np.array([0, 0, 1.0]), axis=np.array([1.0, 0, 0]), orgx=512.3, orgy=500.8,
                    distance=150.0, ed1=np.array([1.0, 0, 0]), ed2=np.array([0, 1.0, 0]), qx=0.172, qy=0.172,
                    nx=1024, ny=1024, phi0=10.0, osc=0.5, starting_frame=1)


def test_predict_then_invert():
    g = _geom()
    A = cell_to_B((50, 60, 70, 90, 95, 90))
    pr = predict_reflections(g, A, 2.0, z_range=(0, 50))
    p0 = g.spot_to_p0(pr["x"], pr["y"], pr["z"])
    h = p0 @ np.linalg.inv(A).T
    assert np.abs(h - pr["hkl"]).max() < 1e-6


def test_refinement_recovers_geometry():
    from xpg.models import Crystal, Experiment
    from xpg.refine import refine
    g = _geom()
    A = cell_to_B((50, 60, 70, 90, 95, 90))
    pr = predict_reflections(g, A, 2.0, z_range=(0, 80))
    rng = np.random.default_rng(3)
    x = pr["x"] + rng.normal(scale=0.1, size=pr["x"].size)
    y = pr["y"] + rng.normal(scale=0.1, size=pr["x"].size)
    z = pr["z"] + rng.normal(scale=0.05, size=pr["x"].size)
    start = Experiment(g.copy(orgx=513.5, orgy=499.9, distance=151.0), Crystal.from_Astar(A @ np.diag([1.004, 0.998, 1.0]), "m"))
    res = refine(start, pr["hkl"], x, y, z, ["POSITION", "BEAM", "ORIENTATION", "CELL"])
    assert abs(res.exp.geom.orgx - 512.3) < 0.05 and abs(res.exp.geom.distance - 150.0) < 0.1
    assert np.allclose(res.exp.crystal.cell, [50, 60, 70, 90, 95, 90], atol=0.02)
    assert res.rmsd_xy < 0.2


def test_french_wilson_acentric_mean():
    """Posterior <J> for an acentric Wilson prior is the mean of a normal truncated at 0."""
    from scipy.stats import truncnorm
    from xpg.aniso import french_wilson
    I = np.array([-2.0, 0.0, 1.0, 5.0, 30.0])
    s = np.array([2.0, 2.0, 1.5, 2.0, 3.0])
    S = np.full(5, 10.0)
    F, sF, Ip, sIp = french_wilson(I, s, S, np.zeros(5, bool))
    mu = I - s ** 2 / S
    ref = truncnorm.mean((0 - mu) / s, np.inf, loc=mu, scale=s)
    assert np.allclose(Ip, ref, rtol=2e-3, atol=2e-3)
    assert np.all(F > 0)

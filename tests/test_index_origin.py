"""Choice between indexing at the header beam centre and at a shifted one (indexing._choose_origin_candidate)."""
from types import SimpleNamespace

import numpy as np

from xpg.indexing import _cell_volume, _choose_origin_candidate, _primitive_volume


def _res(n, rmsd, cell, n_total=21842):
    indexed = np.zeros(n_total, bool)
    indexed[:n] = True
    return SimpleNamespace(indexed=indexed, rmsd_xy=rmsd, exp=SimpleNamespace(crystal=SimpleNamespace(cell=np.array(cell))))


# a C2 data set (118.8 95.9 87.3, beta 100.5): at the header the reduced cell is right; the shifted beam centre gave
# a cell with c tripled that indexed 0.4% more spots with a similar rmsd
HEADER = ("header", _res(18003, 1.03, [76.5, 76.5, 87.5, 81.9, 81.9, 77.8]))
SHIFTED = ("shifted (+12.82, -7.32) px", _res(18083, 1.07, [76.5, 76.5, 262.7, 98.6, 94.6, 102.2]))
GIVEN = ((118.81, 95.88, 87.31, 90.0, 100.497, 90.0), 5)


def test_primitive_volume_matches_the_reduced_cell():
    assert abs(_primitive_volume(*GIVEN) / _cell_volume(HEADER[1].exp.crystal.cell) - 1) < 0.05
    assert abs(_primitive_volume((50.0, 50.0, 50.0, 90, 90, 90), 1) - 125000.0) < 1e-6
    assert abs(_primitive_volume((60.0, 60.0, 90.0, 90, 90, 120), 146) * 3 - _cell_volume((60, 60, 90, 90, 90, 120))) < 1e-6
    assert abs(_primitive_volume((50.0, 50.0, 50.0, 80, 80, 80), 146) - _cell_volume((50, 50, 50, 80, 80, 80))) < 1e-6


def test_supercell_with_no_more_spots_loses():
    lab, why = _choose_origin_candidate([HEADER, SHIFTED])
    assert lab == "header" and "smaller cell" in why


def test_given_cell_decides():
    lab, why = _choose_origin_candidate([HEADER, SHIFTED], known_cell=GIVEN)
    assert lab == "header" and "given cell" in why
    # the given cell matching the shifted solution picks that one
    big = ((118.81, 95.88, 3 * 87.31, 90.0, 100.497, 90.0), 5)
    assert _choose_origin_candidate([HEADER, SHIFTED], known_cell=big)[0].startswith("shifted")


def test_clearly_more_spots_still_wins():
    shifted = ("shifted", _res(19000, 1.07, [76.5, 76.5, 262.7, 98.6, 94.6, 102.2]))
    assert _choose_origin_candidate([HEADER, shifted]) == ("shifted", None)


def test_same_volume_keeps_the_count_and_rmsd_rules():
    same = [76.5, 76.5, 87.5, 81.9, 81.9, 77.8]
    a, b = ("header", _res(18003, 1.03, same)), ("shifted", _res(18083, 1.07, same))
    assert _choose_origin_candidate([a, b]) == ("shifted", None)            # most spots
    b = ("shifted", _res(18083, 1.40, same))
    assert _choose_origin_candidate([a, b]) == ("header", None)             # clearly tighter fit
    assert _choose_origin_candidate([a]) == ("header", None)


def _scaled(cell, f):
    """The cell with its volume scaled by f (lengths by f^(1/3))."""
    s = f ** (1.0 / 3.0)
    return [cell[0] * s, cell[1] * s, cell[2] * s] + list(cell[3:])


def test_given_cell_does_not_decide_within_one_lattice():
    # the same lattice at both origins, volumes 1% apart, straddling the edge of the given-cell window: the volume
    # difference is noise, so the clearly tighter fit decides as without a given cell
    v0 = _primitive_volume(*GIVEN)
    v_hdr = _cell_volume(HEADER[1].exp.crystal.cell)
    a = ("header", _res(18000, 2.0, _scaled(HEADER[1].exp.crystal.cell, 1.145 * v0 / v_hdr)))
    b = ("shifted", _res(18050, 1.0, _scaled(HEADER[1].exp.crystal.cell, 1.155 * v0 / v_hdr)))
    assert _choose_origin_candidate([a, b], known_cell=GIVEN) == ("shifted", None)
    assert _choose_origin_candidate([a, b]) == ("shifted", None)


def test_given_cell_tolerates_a_few_percent():
    # a given cell 8% off in volume (2.6% per length, e.g. a distance error) still picks the matching lattice
    off = (tuple(_scaled(GIVEN[0], 1.08)[:3]) + GIVEN[0][3:], 5)
    lab, why = _choose_origin_candidate([HEADER, SHIFTED], known_cell=off)
    assert lab == "header" and "given cell" in why
    big = (tuple(_scaled(GIVEN[0], 3 * 0.92)[:3]) + GIVEN[0][3:], 5)
    assert _choose_origin_candidate([HEADER, SHIFTED], known_cell=big)[0].startswith("shifted")


def test_degenerate_given_cell_is_ignored():
    for bad in (((0.0,) * 6, 19), ((50.0, 60.0, 70.0, 0.0, 0.0, 0.0), 19), ((float("nan"),) * 3 + (90.0,) * 3, 1)):
        lab, why = _choose_origin_candidate([HEADER, SHIFTED], known_cell=bad)
        assert lab == "header" and "smaller cell" in why


def test_smaller_cell_with_a_clearly_worse_fit_does_not_win():
    # the bigger cell fits clearly tighter (1.0 against 1.6 px): the rmsd rule keeps it
    big = ("header", _res(18003, 1.00, [76.5, 76.5, 175.0, 81.9, 81.9, 77.8]))
    small = ("shifted", _res(18050, 1.60, [76.5, 76.5, 87.5, 81.9, 81.9, 77.8]))
    assert _choose_origin_candidate([big, small]) == ("header", None)

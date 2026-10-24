"""Synthetic observations with known symmetry for the point-group decision test (test_features)."""
import gemmi
import numpy as np

from xpg.lattice import lattice_candidates
from xpg.simulate import true_intensities
from xpg.symmetry import score_symmetry
from xpg.symops import d_spacing, rot_matrices

CELL = np.array([60.0, 60.0, 90.0, 90.0, 90.0, 120.0])     # Niggli-reduced hexagonal cell (reduced = conventional)


def observations(spec, dmin=2.2, mult=4.0, sys_err=0.0, noise=1.0, seed=0):
    """Unmerged observations: each unique reflection measured ~mult times at random symmetry-equivalent (true
    group) and Friedel-random indices; multiplicative systematic error sys_err (absorption / damage per
    observation), Poisson-like noise. Returns hkl (reduced cell), I, sig, s, phi."""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(CELL[0] / dmin)) + 1
    nl = int(np.ceil(CELL[2] / dmin)) + 1
    g = np.array([(h, k, l) for h in range(-n, n + 1) for k in range(-n, n + 1) for l in range(0, nl + 1)])
    d = d_spacing(g, CELL)
    g = g[(d >= dmin) & (d < 40)]
    from xpg.symops import to_asu
    hasu, _, _ = to_asu(g, spec.space_group)
    key = (hasu[:, 0] + 512) * 1048576 + (hasu[:, 1] + 512) * 1024 + hasu[:, 2] + 512
    _, first = np.unique(key, return_index=True)
    uniq = g[first]
    Itrue, _ = true_intensities(spec, uniq, rng)
    R = rot_matrices(spec.space_group)
    m = rng.poisson(mult, uniq.size // 3 if False else len(uniq))
    idx = np.repeat(np.arange(len(uniq)), m)
    ops = R[rng.integers(0, len(R), idx.size)]
    h = np.einsum("ni,nij->nj", uniq[idx], ops) * rng.choice([-1, 1], idx.size)[:, None]
    It = Itrue[idx]
    phi = rng.uniform(0, 360, idx.size)
    corr = np.exp(sys_err * rng.normal(size=idx.size))                       # per-observation systematic error
    bg = 0.05 * np.mean(Itrue)
    sig = noise * np.sqrt(np.maximum(It * corr, 0) + bg)
    I = It * corr + sig * rng.normal(size=idx.size)
    s = 1.0 / d_spacing(h, CELL)
    return h, I, sig, s, phi


def decide(h, I, sig, s, phi):
    cands = lattice_candidates(CELL, max_obliq=3.0)
    scores, _, cc_true = score_symmetry(h, I, sig, s, phi, cands, verbose=False)
    best = scores[0]
    order = lambda c: len(gemmi.SpaceGroup(c.space_group).operations().sym_ops)
    if cc_true < 0.45 and best.prob < 0.9:                    # pipeline's underpowered rule
        lower = [x for x in scores if order(x.cand) < order(best.cand)]
        if lower:
            best = max(lower, key=lambda x: x.prob)
    return f"{best.cand.bravais} {best.cand.point_group}", best.prob



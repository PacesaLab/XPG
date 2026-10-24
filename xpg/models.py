"""Parameterised experiment models (crystal + geometry) used by refinement and integration."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .geometry import Geometry, cell_to_B, cell_from_Astar, U_from_Astar

# free cell parameters per lattice system (conventional settings)
CELL_PARAMS = {
    "a": ["a", "b", "c", "al", "be", "ga"],   # triclinic
    "m": ["a", "b", "c", "be"],                # monoclinic, unique axis b
    "o": ["a", "b", "c"],                      # orthorhombic
    "t": ["a", "c"],                           # tetragonal
    "h": ["a", "c"],                           # hexagonal / trigonal / rhombohedral (hex. setting)
    "c": ["a"],                                # cubic
}


def cell_from_free(system: str, v):
    """Full cell (..., 6) from free parameters ``v`` (..., n) for a lattice system (torch tensor in, differentiable)."""
    import torch
    e = [v[..., i] for i in range(v.shape[-1])]
    one = torch.ones_like(e[0])
    full = {"a": lambda: e[:6], "m": lambda: [e[0], e[1], e[2], 90.0 * one, e[3], 90.0 * one],
            "o": lambda: [e[0], e[1], e[2], 90.0 * one, 90.0 * one, 90.0 * one],
            "t": lambda: [e[0], e[0], e[1], 90.0 * one, 90.0 * one, 90.0 * one],
            "h": lambda: [e[0], e[0], e[1], 90.0 * one, 90.0 * one, 120.0 * one],
            "c": lambda: [e[0], e[0], e[0], 90.0 * one, 90.0 * one, 90.0 * one]}
    if system not in full:
        raise ValueError(system)
    return torch.stack(full[system](), -1)


def _cell_from_free_np(system: str, v):
    v = np.asarray(v, float)
    full = {"a": lambda: v[:6], "m": lambda: [v[0], v[1], v[2], 90.0, v[3], 90.0],
            "o": lambda: [v[0], v[1], v[2], 90.0, 90.0, 90.0], "t": lambda: [v[0], v[0], v[1], 90.0, 90.0, 90.0],
            "h": lambda: [v[0], v[0], v[1], 90.0, 90.0, 120.0], "c": lambda: [v[0], v[0], v[0], 90.0, 90.0, 90.0]}
    if system not in full:
        raise ValueError(system)
    return np.array(full[system](), float)


def free_from_cell(system: str, cell):
    a, b, c, al, be, ga = cell
    return np.array({"a": [a, b, c, al, be, ga], "m": [a, b, c, be], "o": [a, b, c],
                     "t": [(a + b) / 2, c], "h": [(a + b) / 2, c], "c": [(a + b + c) / 3]}[system], float)


def symmetrize_cell(system: str, cell):
    return np.asarray(cell_from_free_np(system, free_from_cell(system, cell)))


def cell_from_free_np(system, v):
    return _cell_from_free_np(system, v)


def B_t(cell):
    """Busing-Levy B matrix (columns a*, b*, c*) of cell(s) (..., 6), shape (..., 3, 3), differentiable. torch."""
    import torch
    a, b, c = cell[..., 0], cell[..., 1], cell[..., 2]
    al, be, ga = torch.deg2rad(cell[..., 3]), torch.deg2rad(cell[..., 4]), torch.deg2rad(cell[..., 5])
    cx = c * torch.cos(be)
    cy = c * (torch.cos(al) - torch.cos(be) * torch.cos(ga)) / torch.sin(ga)
    cz = torch.sqrt(torch.clamp(c * c - cx * cx - cy * cy, min=1e-12))
    z = torch.zeros_like(a)
    real = torch.stack([torch.stack([a, z, z], -1),
                        torch.stack([b * torch.cos(ga), b * torch.sin(ga), z], -1),
                        torch.stack([cx, cy, cz], -1)], -2)
    return torch.linalg.inv(real)


@dataclass
class Crystal:
    U: np.ndarray            # orientation (3x3)
    cell: np.ndarray         # (6,)
    system: str = "a"        # one of CELL_PARAMS keys
    centring: str = "P"
    space_group: str = "P 1"
    # extra 'missetting' block-wise rotations along phi (scan-varying), optional
    mosaicity: float = 0.1   # reflecting range e.s.d. (deg)
    divergence: float = 0.03  # beam divergence e.s.d. (deg)

    @property
    def Astar(self):
        return self.U @ cell_to_B(self.cell)

    @classmethod
    def from_Astar(cls, Astar, system="a", **kw):
        cell = cell_from_Astar(Astar)
        cell = symmetrize_cell(system, cell)
        return cls(U=U_from_Astar(Astar, cell), cell=cell, system=system, **kw)

    def copy(self, **kw):
        return replace(self, **kw)


@dataclass
class Experiment:
    geom: Geometry
    crystal: Crystal

    def copy(self):
        return Experiment(self.geom.copy(), self.crystal.copy())


def d_spacing_from_exp(hkl, exp):
    p = hkl @ exp.crystal.Astar.T
    return 1.0 / np.linalg.norm(p, axis=1)

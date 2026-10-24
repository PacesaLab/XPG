"""Diagnostic correction images (integer CBF, for adxv / albula).

BACKGROUND:     xpg_background.cbf (mean background x 1000, untrusted = -3), xpg_gain.cbf (gain x 1000, 4x4 binned)
SCALE / merge (SAVE_CORRECTION_IMAGES): reciprocal correction factors x 1000, i.e. the factors g with
I_obs = g I:
  xpg_decay.cbf        rows: 20 resolution bins in 1/d^2, columns: image batches     (scale x B-factor decay)
  xpg_modulation.cbf   detector-surface modulation on a grid of ~20-pixel bins
  xpg_absorption.cbf   rows: 13 detector positions (centre + two rings), columns: image batches (absorption)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry import rotation_matrices_np
from .cbf import write_cbf


def _hdr(g):
    return dict(qx=g.qx, qy=g.qy, wavelength=g.wavelength, distance=g.distance, orgx=g.orgx, orgy=g.orgy,
                start_angle=g.phi0, osc=g.osc)


def write_init_images(wd, geom, background, mask, gain):
    wd = Path(wd)
    bk = np.where(mask & (background >= 0), np.round(background * 1000.0), -3).astype(np.int32)
    write_cbf(wd / "xpg_background.cbf", bk, _hdr(geom))
    ny, nx = mask.shape
    gb = np.full(((ny + 3) // 4, (nx + 3) // 4), int(round(gain * 1000)), np.int32)
    mb = mask[: gb.shape[0] * 4, : gb.shape[1] * 4]
    if mb.shape == (gb.shape[0] * 4, gb.shape[1] * 4):
        gb[~mb.reshape(gb.shape[0], 4, gb.shape[1], 4).any(axis=(1, 3))] = -3
    write_cbf(wd / "xpg_gain.cbf", gb, _hdr(geom))


def _detector_positions(geom, n_ring=6):
    """Centre of the detector plus two rings of 6 positions (13 points)."""
    cx, cy = geom.nx / 2, geom.ny / 2
    r = 0.33 * min(geom.nx, geom.ny)
    pts = [(cx, cy)]
    for rr, off in ((r, 0.0), (2 * r * 0.75, np.pi / 6)):
        for k in range(n_ring):
            a = off + 2 * np.pi * k / n_ring
            pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    return np.array(pts)


def write_correction_images(wd, scaler, geom, U, dataset=0, suffix="", phi_range=None, smax2=None,
                            batch_deg=10.0):
    """DECAY / MODPIX / ABSORP images of a fitted Scaler (dataset index ``dataset``)."""
    wd = Path(wd)
    kn = scaler._knot_def[dataset][0]
    p0, p1 = (kn[0], kn[-1]) if phi_range is None else phi_range
    nbat = max(int(np.ceil(abs(p1 - p0) / batch_deg)), 1)
    phis = p0 + (np.arange(nbat) + 0.5) * (p1 - p0) / nbat
    smax2 = smax2 or float(np.max(scaler.s2))
    s2 = (np.arange(20) + 0.5) * smax2 / 20
    P, S = np.meshgrid(phis, s2)
    t = scaler.eval_terms(dataset, P.ravel(), s2=S.ravel())
    dec = (t["scale"] * t["decay"]).reshape(S.shape)
    write_cbf(wd / f"xpg_decay{suffix}.cbf", np.round(1000 * dec).astype(np.int32), _hdr(geom))
    if scaler.use_mod:
        step = 20
        gy, gx = np.mgrid[step / 2:geom.ny:step, step / 2:geom.nx:step]
        mod = scaler.eval_terms(dataset, np.zeros(gx.size), x=gx.ravel(), y=gy.ravel())["modulation"]
        write_cbf(wd / f"xpg_modulation{suffix}.cbf", np.round(1000 * mod.reshape(gx.shape)).astype(np.int32), _hdr(geom))
    if scaler.nc:
        pts = _detector_positions(geom)
        S1 = geom.pixel_to_s(pts[:, 0], pts[:, 1])
        e = S1 / np.linalg.norm(S1, axis=1, keepdims=True)
        img = np.zeros((len(pts), nbat))
        for j, ph in enumerate(phis):
            R = rotation_matrices_np(geom.axis, -np.radians(ph))
            s1c = (e @ R.T) @ U
            img[:, j] = scaler.eval_terms(dataset, np.full(len(pts), ph), s1c=s1c)["absorption"]
        write_cbf(wd / f"xpg_absorption{suffix}.cbf", np.round(1000 * img).astype(np.int32), _hdr(geom))

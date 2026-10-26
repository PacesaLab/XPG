"""Streaming 3D clustering of strong pixels into spots (SPOTS), without torch: the clustering process and its
workers import only this module (numpy / scipy).

Strong pixels are clustered in (x, y, frame) by sparse connected components, in a streaming fashion so memory does
not grow with the number of frames; touching spots are split at significant local maxima.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PIX0 = 1.0   # = geometry.PIX0 (coordinate of the centre of array element 0), kept here so no torch is imported


@dataclass
class SpotList:
    x: np.ndarray        # pixel coordinates (XPG convention)
    y: np.ndarray
    z: np.ndarray        # continuous frame coordinate
    intensity: np.ndarray
    npix: np.ndarray
    zmin: np.ndarray
    zmax: np.ndarray
    hkl: np.ndarray | None = None

    def __len__(self):
        return self.x.size

    def subset(self, sel):
        return SpotList(**{k: (v[sel] if v is not None else None) for k, v in self.__dict__.items()})

    def write_legacy_spots(self, path):
        with open(path, "w") as f:
            for i in range(len(self)):
                line = f"{self.x[i]:10.2f}{self.y[i]:10.2f}{self.z[i]:10.2f}{self.intensity[i]:10.0f}."
                if self.hkl is not None:
                    line += f"{self.hkl[i, 0]:8d}{self.hkl[i, 1]:8d}{self.hkl[i, 2]:8d}"
                f.write(line + "\n")


_FWD = [(0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1)] + [(1, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def _fwd_offsets(r):
    """Forward half of the neighbourhood |dx|, |dy| <= r (same frame) and adjacent frame."""
    if r <= 1:
        return _FWD
    same = [(0, dy, dx) for dy in range(0, r + 1) for dx in range(-r, r + 1) if (dy, dx) > (0, 0)]
    nxt = [(1, dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    return same + nxt


def _components(z, y, x, nx, ny, r=1):
    """Connected components of sparse voxels; pixels within ``r`` pixels in-plane (and adjacent frames)
    are connected (CLUSTER_RADIUS: weak spots have gaps in their strong-pixel mask). Returns labels."""
    n = z.size
    if n == 0:
        return np.zeros(0, int), 0
    key = (z.astype(np.int64) * ny + y) * nx + x
    order = np.argsort(key)
    ks = key[order]
    xs, ys = x[order], y[order]
    rows, cols = [], []
    # neighbour lookups with sorted queries (ks + offset); the same edges, so the same components
    for dz, dy, dx in _fwd_offsets(r):
        ok = (xs + dx >= 0) & (xs + dx < nx) & (ys + dy >= 0) & (ys + dy < ny)
        tgt = ks + (dz * ny + dy) * nx + dx
        pos = np.minimum(np.searchsorted(ks, tgt), n - 1)
        hit = np.flatnonzero(ok & (ks[pos] == tgt))
        rows.append(order[hit])
        cols.append(order[pos[hit]])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(r.size, np.int8), (r, c)), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    return lab, nc


_ALL26 = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0)]


def _split_merged(z, y, x, v, b, lab, nx, ny, min_sep_xy=2.0, nsig=3.0):
    """Split clusters of touching spots at their significant local maxima (sparse 3D watershed-lite).

    A pixel is a peak if its background-subtracted value is >= all of its 26 neighbours present in the
    strong-pixel set and significant (> nsig * sqrt(bg)); peaks closer than min_sep_xy px (and within
    one frame) are merged keeping the higher. Pixels go to the nearest peak of their own cluster.
    Standard implementations keep such clusters whole (and may then discard them); splitting recovers crowded data
    (long cell axes, high mosaicity, overlapping lattices)."""
    n = z.size
    if n == 0:
        return lab
    net = v - b
    key = (z.astype(np.int64) * ny + y) * nx + x
    order = np.argsort(key)
    ks = key[order]
    xs, ys, nets = x[order], y[order], net[order]          # work in key order (sorted neighbour queries)
    is_max_s = nets > nsig * np.sqrt(np.maximum(b[order], 1.0))
    for dz, dy, dx in _ALL26:
        tgt = ks + (dz * ny + dy) * nx + dx
        pos = np.minimum(np.searchsorted(ks, tgt), n - 1)
        hit = (ks[pos] == tgt) & (xs + dx >= 0) & (xs + dx < nx) & (ys + dy >= 0) & (ys + dy < ny)
        nb_net = np.where(hit, nets[pos], -np.inf)
        # strict on one side of the neighbourhood to break ties
        is_max_s &= (nets > nb_net) | ((nets == nb_net) & ((dz, dy, dx) > (0, 0, 0)))
    is_max = np.empty(n, bool)
    is_max[order] = is_max_s
    pk = np.flatnonzero(is_max)
    if pk.size == 0:
        return lab
    # clusters with >= 2 peaks
    npk = np.bincount(lab[pk], minlength=lab.max() + 1)
    multi = np.flatnonzero(npk >= 2)
    if multi.size == 0:
        return lab
    new = lab.astype(np.int64) * 4096
    from scipy.spatial import cKDTree
    # pixels and peaks grouped by cluster once (a scan of all pixels per cluster was quadratic)
    by_lab = np.argsort(lab, kind="stable")
    lab_s = lab[by_lab]
    p_lo, p_hi = np.searchsorted(lab_s, multi, "left"), np.searchsorted(lab_s, multi, "right")
    pk_s = pk[np.argsort(lab[pk], kind="stable")]
    lab_pk = lab[pk_s]
    k_lo, k_hi = np.searchsorted(lab_pk, multi, "left"), np.searchsorted(lab_pk, multi, "right")
    for ci, c in enumerate(multi):
        pix = by_lab[p_lo[ci]:p_hi[ci]]
        pks = pk_s[k_lo[ci]:k_hi[ci]]
        pks = pks[np.argsort(-net[pks], kind="stable")]
        cx, cy, cz, cnet = x[pix].astype(float), y[pix].astype(float), z[pix].astype(float), net[pix]
        # Is peak q a separate spot from the higher peak r? Requires a dip between them that is significant
        # against the Poisson noise of q: net(q) - saddle > nsig * sqrt(counts(q)); low-count spots otherwise
        # fragment at noise maxima (weak, low-background data). The same detector position in nearby frames is the
        # same reflection (its rocking curve spans several frames on fine-sliced data); otherwise the dip is
        # tested along the 3D path, and a path pixel missing from the strong set means it reaches background.
        # Vectorised over the higher peaks already kept (one Python call per pair was SPOTS's bottleneck), in
        # chunks of bounded size: an all-pairs evaluation ran out of memory on huge clusters (ice rings, streaks).
        thr_v = nsig * np.sqrt(np.maximum(v, np.float32(1.0)))
        chunk = max(1, 4_000_000 // max(pix.size, 1))

        def separated_from_all(q, rs):
            for c0 in range(0, rs.size, chunk):
                r = rs[c0:c0 + chunk]
                far = np.hypot(x[q] - x[r], y[q] - y[r]) >= min_sep_xy
                if not far.all():
                    return False
                a0, a1, a2 = x[r].astype(float), y[r].astype(float), z[r].astype(float)
                d0, d1, d2 = x[q] - a0, y[q] - a1, z[q] - a2
                L2 = d0 * d0 + d1 * d1 + d2 * d2
                t = np.clip(((cx[None] - a0[:, None]) * d0[:, None] + (cy[None] - a1[:, None]) * d1[:, None]
                             + (cz[None] - a2[:, None]) * d2[:, None]) / L2[:, None], 0, 1)
                dist = np.sqrt((cx[None] - a0[:, None] - t * d0[:, None]) ** 2
                               + (cy[None] - a1[:, None] - t * d1[:, None]) ** 2
                               + (cz[None] - a2[:, None] - t * d2[:, None]) ** 2)
                seg = (dist <= 0.75) & (t > 0) & (t < 1)
                nseg = seg.sum(axis=1)
                steps = np.ceil(np.sqrt(L2)).astype(int) - 1
                smin = np.where(seg, cnet[None], np.inf).min(axis=1)
                saddle = np.where((nseg >= steps) & (nseg > 0), smin, 0.0).astype(cnet.dtype)
                if not np.all(net[q] - saddle > thr_v[q]):
                    return False
            return True

        kept = np.empty(pks.size, np.int64)
        kept[0], nk = pks[0], 1
        big = pix.size > 20000 or pks.size > 200
        if big:
            # huge merged clusters (dense / crowded images: a connected strong region over many spots): every peak
            # against every kept peak over every pixel is peaks x kept x pixels (SPOTS stalled for > 20 min on a
            # dense simulation). Peaks farther apart than a few spot separations are distinct spots; the dip test
            # runs only for near pairs, on the pixels around the connecting segment (KD-tree)
            ptree = cKDTree(np.c_[cx, cy, cz])
            far_r = max(4.0 * min_sep_xy, 12.0)
            # the kept peaks by in-plane cell (side >= far_r: the near ones are in the 3 x 3 cells around a peak) and
            # the pixels closer than min_sep_xy in-plane to one (a peak there is not separate), so the work per peak
            # does not grow with the kept peaks: an ice ring on every frame is one cluster with millions of peaks
            cs = int(np.ceil(far_r))
            m = int(np.ceil(min_sep_xy))
            by, bx = (np.array(a, np.int64) for a in zip(*[(i, j) for i in range(-m, m + 1) for j in range(-m, m + 1)
                                                            if np.hypot(i, j) < min_sep_xy]))
            x0, y0 = int(x[pix].min()) - m, int(y[pix].min()) - m
            blocked = np.zeros((int(y[pix].max()) - y0 + m + 1, int(x[pix].max()) - x0 + m + 1), bool)
            cells = {}

            def mark(q):
                blocked[y[q] - y0 + by, x[q] - x0 + bx] = True
                cells.setdefault((int(x[q]) // cs, int(y[q]) // cs), []).append(q)

            def separated_from_all(q, _):      # noqa: F811 -- large-cluster variant of the test above
                if blocked[y[q] - y0, x[q] - x0]:
                    return False
                ix, iy = int(x[q]) // cs, int(y[q]) // cs
                rs = [r for i in (-1, 0, 1) for j in (-1, 0, 1) for r in cells.get((ix + i, iy + j), ())]
                if not rs:
                    return True
                rs = np.sort(np.array(rs, np.int64))
                dxy = np.hypot(x[q] - x[rs], y[q] - y[rs])
                D = np.sqrt(dxy ** 2 + (z[q] - z[rs]).astype(float) ** 2)
                for r in rs[D <= far_r]:
                    a0, a1, a2 = float(x[r]), float(y[r]), float(z[r])
                    d0, d1, d2 = x[q] - a0, y[q] - a1, z[q] - a2
                    L2 = d0 * d0 + d1 * d1 + d2 * d2
                    sel = ptree.query_ball_point([(x[q] + a0) / 2, (y[q] + a1) / 2, (z[q] + a2) / 2],
                                                 np.sqrt(L2) / 2 + 1.0)
                    if not sel:
                        continue
                    sel = np.asarray(sel)
                    t = np.clip(((cx[sel] - a0) * d0 + (cy[sel] - a1) * d1 + (cz[sel] - a2) * d2) / L2, 0, 1)
                    dist = np.sqrt((cx[sel] - a0 - t * d0) ** 2 + (cy[sel] - a1 - t * d1) ** 2 + (cz[sel] - a2 - t * d2) ** 2)
                    seg = (dist <= 0.75) & (t > 0) & (t < 1)
                    steps = int(np.ceil(np.sqrt(L2))) - 1
                    saddle = cnet[sel][seg].min() if (seg.sum() >= steps and seg.any()) else 0.0
                    if not net[q] - saddle > thr_v[q]:
                        return False
                return True

            mark(pks[0])
        for q in pks[1:]:   # keep peaks that are significantly separated from every higher kept one
            if separated_from_all(q, kept[:nk]):
                kept[nk] = q
                nk += 1
                if big:
                    mark(q)
        if nk < 2:
            continue
        kp = kept[:nk]
        tree = cKDTree(np.c_[x[kp], y[kp], z[kp] * 1.5])
        _, ia = tree.query(np.c_[x[pix], y[pix], z[pix] * 1.5])
        new[pix] = c * 4096 + 1 + ia
    return new


class _Accumulator:
    """Streaming 3D clustering of strong pixels."""

    def __init__(self, nx, ny, min_pix, max_pix, max_centroid_dist=3.0, cluster_radius=2, keyed=False, divert=()):
        self.nx, self.ny = nx, ny
        self.cluster_radius = cluster_radius
        self.min_pix, self.max_pix = min_pix, max_pix
        self.max_centroid_dist = max_centroid_dist
        self.buf = []  # list of (z, y, x, val, bg)
        self.out = []
        # block-parallel clustering (_cluster_parallel): keyed adds each spot's position in the sequential output
        # (its cluster's last frame and first pixel, its split label); clusters with a pixel in a divert frame (a
        # block's edge frames) are not closed here but kept as raw pixels for the merge across blocks
        self.keyed = keyed
        self.divert = tuple(divert)
        self.raw = []
        self.n_raw = 0
        # clusters still open after a flush (touching its last frame): their pixels, component labels and that frame.
        # Only their pixels on that frame can connect to later frames, so a flush labels those and the new pixels
        # and joins the result to the open labels: a cluster on every frame (an ice ring, a streak) is not
        # re-labelled again and again, which made the clustering quadratic in the number of frames
        self.open = None
        self.open_lab = None
        self.n_open = 0
        self.open_last = None

    def add(self, frame_idx, y, x, val, bg):
        self.buf.append((np.full(y.size, frame_idx, np.int32), y.astype(np.int32), x.astype(np.int32),
                         val.astype(np.float32), bg.astype(np.float32)))

    def _label(self):
        """Pixels of the open clusters followed by the new ones, with their connected-component labels (numbered by
        first occurrence, as a labelling of all of them at once)."""
        new = [np.concatenate(a) for a in zip(*self.buf)] if self.buf else None
        if self.open is None:
            z, y, x, v, b = new
            lab, nc = _components(z, y, x, self.nx, self.ny, self.cluster_radius)
            return z, y, x, v, b, lab, nc
        n_open = self.n_open
        if new is None:
            return (*self.open, self.open_lab, n_open)
        anchor = np.flatnonzero(self.open[0] == self.open_last)
        za, ya, xa = (np.r_[self.open[i][anchor], new[i]] for i in range(3))
        lab2, nc2 = _components(za, ya, xa, self.nx, self.ny, self.cluster_radius)
        # open clusters and new components joined through the anchor pixels
        na, nn = anchor.size, n_open + nc2
        g = coo_matrix((np.ones(na, np.int8), (self.open_lab[anchor], n_open + lab2[:na])), shape=(nn, nn))
        nc, comp = connected_components(g, directed=False)
        # numbered by first occurrence: the clusters with an open one by their first open one (open labels are in
        # that order), then the new ones (one new component each) in their order
        key = np.full(nc, nn, np.int64)
        np.minimum.at(key, comp, np.arange(nn))
        rank = np.empty(nc, np.int64)
        rank[np.argsort(key)] = np.arange(nc)
        node = rank[comp]
        z, y, x, v, b = (np.r_[self.open[i], new[i]] for i in range(5))
        return z, y, x, v, b, np.r_[node[self.open_lab], node[n_open + lab2[na:]]], nc

    def flush(self, last_frame=None):
        if not self.buf and self.open is None:
            return
        z, y, x, v, b, lab, nc = self._label()
        zmax_c = np.full(nc, -1, np.int64)
        np.maximum.at(zmax_c, lab, z)
        if last_frame is not None:
            keep_open = zmax_c[lab] >= last_frame
        else:
            keep_open = np.zeros(z.size, bool)
        done = ~keep_open
        if self.divert and done.any():
            edge = np.zeros(nc, np.int8)          # bit 1: touches the first edge frame, bit 2: the last
            for zf, bit in self.divert:
                edge[lab[z == zf]] |= bit
            div = done & (edge[lab] > 0)
            if div.any():
                _, cid = np.unique(lab[div], return_inverse=True)      # cluster ids, unique within this accumulator
                self.raw.append((z[div], y[div], x[div], v[div], b[div], self.n_raw + cid))
                self.n_raw += int(cid.max()) + 1
                done &= ~div
        if done.any():
            labd = _split_merged(z[done], y[done], x[done], v[done], b[done], lab[done], self.nx, self.ny)
            self.out.append(self._props(z[done], y[done], x[done], v[done], b[done], labd,
                                        lab[done] if self.keyed else None))
        self.buf = []
        if keep_open.any():
            self.open = (z[keep_open], y[keep_open], x[keep_open], v[keep_open], b[keep_open])
            open_c = zmax_c >= last_frame
            self.open_lab = (np.cumsum(open_c) - 1)[lab[keep_open]]       # still numbered by first occurrence
            self.n_open = int(open_c.sum())
            self.open_last = last_frame
        else:
            self.open = self.open_lab = self.open_last = None
            self.n_open = 0

    def _props(self, z, y, x, v, b, lab, parent=None):
        u, first, lab = np.unique(lab, return_index=True, return_inverse=True)
        nc = u.size
        w = np.maximum(v - b, 0.0).astype(np.float64)
        npix = np.bincount(lab, minlength=nc)
        sw = np.bincount(lab, w, nc)
        sw_safe = np.where(sw > 0, sw, 1.0)
        cx = np.bincount(lab, w * (x + PIX0), nc) / sw_safe
        cy = np.bincount(lab, w * (y + PIX0), nc) / sw_safe
        cz = np.bincount(lab, w * (z + 0.5), nc) / sw_safe   # frame f (0-based index into z) centre
        zmin = np.full(nc, 1 << 30)
        zmax = np.full(nc, -1)
        np.minimum.at(zmin, lab, z)
        np.maximum.at(zmax, lab, z)
        # SPOT_MAXIMUM-CENTROID: reject clusters whose brightest pixel is far from the centroid
        # (typically several merged spots)
        order = np.lexsort((v, lab))
        last = np.r_[np.flatnonzero(np.diff(lab[order])), lab.size - 1]
        imax = order[last]
        dmax = np.hypot(x[imax] + PIX0 - cx, y[imax] + PIX0 - cy)
        ok = (npix >= self.min_pix) & (npix <= self.max_pix) & (sw > 0) & (dmax <= self.max_centroid_dist)
        out = dict(x=cx[ok], y=cy[ok], z=cz[ok], intensity=sw[ok], npix=npix[ok],
                   zmin=zmin[ok].astype(float), zmax=zmax[ok] + 1.0)
        if parent is not None:
            # the unsplit cluster of each spot: its last frame (when the sequential run closes it) and its first
            # pixel in arrival order (frame, row, column: its rank among the clusters closed together)
            pu, pinv = np.unique(parent, return_inverse=True)
            pz = np.full(pu.size, -1, np.int64)
            np.maximum.at(pz, pinv, z)
            pk = np.full(pu.size, np.iinfo(np.int64).max, np.int64)
            np.minimum.at(pk, pinv, (z.astype(np.int64) * self.ny + y) * self.nx + x)
            pi = pinv[first]
            out.update(pz=pz[pi][ok], pk=pk[pi][ok], sub=u.astype(np.int64)[ok])
        return out

    def result(self):
        if not self.out:
            e = np.zeros(0)
            return SpotList(e, e, e, e, e.astype(int), e, e)
        d = {k: np.concatenate([o[k] for o in self.out]) for k in self.out[0]}
        return SpotList(**d)


def _cluster_batch(acc, state, fr, kk, yy, xx, val, bg):
    """Add one batch of frames' strong pixels (kk: index of the frame in the batch) to the accumulator and close the
    clusters that cannot grow any more."""
    cut = np.searchsorted(kk, np.arange(len(fr) + 1))
    for k, f in enumerate(fr):
        if state["prev"] is not None and f != state["prev"] + 1:
            acc.flush()  # gap in frames: close all
        sl = slice(cut[k], cut[k + 1])
        acc.add(f - 1, yy[sl], xx[sl], val[sl], bg[sl])
        state["prev"] = f
    acc.flush(last_frame=state["prev"] - 1)       # finalize clusters that cannot grow any more


def _flush_thresholds(frame_lists):
    """The sequential run's flushes (_cluster_batch over these batches, then a final flush): a cluster is closed by
    the first flush whose threshold exceeds its last frame (0-based); the thresholds do not decrease."""
    c, prev = [], None
    for fr in frame_lists:
        for f in fr:
            if prev is not None and f != prev + 1:
                c.append(prev)                  # gap: every open cluster (last frame <= prev - 1) is closed
            prev = f
        c.append(prev - 1)                      # flush(last_frame=prev - 1): clusters ending before it
    c.append(np.iinfo(np.int64).max)            # final flush
    return np.array(c, np.int64)


def _cluster_block(args, batches, first_edge):
    """One block of consecutive batches: its closed clusters (keyed) and, as raw pixels, the clusters with a pixel in
    an edge frame (the last one, and the first unless the block starts the sweep), which may continue in the
    neighbouring blocks."""
    last = batches[-1][0][-1] - 1
    acc = _Accumulator(*args, keyed=True, divert=(((first_edge, 1),) if first_edge is not None else ()) + ((last, 2),))
    state = dict(prev=None)
    for b in batches:
        _cluster_batch(acc, state, *b)
    acc.flush()
    return acc.out, acc.raw, acc.n_raw


def _merge_group(args, parts):
    """Edge clusters of one merge job (whole groups of connected clusters), merged (keyed)."""
    acc = _Accumulator(*args, keyed=True)
    acc.buf = [tuple(np.concatenate(a) for a in zip(*parts))]
    acc.flush()
    return acc.out


def _edge_jobs(args, raws, n_jobs):
    """Edge clusters of the blocks (raws[k]: block k's (z, y, x, v, b, cluster id)) grouped into clusters that can
    connect: two of them join only through pixels in the two frames at a block boundary, so the components of those
    pixels alone (union-find over the cluster ids) give the groups exactly. Whole groups are packed into n_jobs
    merge jobs of similar size."""
    nx, ny, r = args[0], args[1], args[5]
    parts = [tuple(np.concatenate(a) for a in zip(*rk)) if rk else None for rk in raws]
    off = np.cumsum([0] + [int(p[5].max()) + 1 if p is not None else 0 for p in parts])
    up = np.arange(off[-1])

    def root(i):
        while up[i] != i:
            up[i] = up[up[i]]
            i = up[i]
        return i
    for k in range(len(parts) - 1):
        a, b = parts[k], parts[k + 1]
        if a is None or b is None:
            continue
        last = a[0].max()
        sa, sb = a[0] == last, b[0] == last + 1
        if not (sa.any() and sb.any()):
            continue
        z = np.r_[a[0][sa], b[0][sb]]
        lab, _ = _components(z, np.r_[a[1][sa], b[1][sb]], np.r_[a[2][sa], b[2][sb]], nx, ny, r)
        cid = np.r_[a[5][sa] + off[k], b[5][sb] + off[k + 1]]
        o = np.argsort(lab, kind="stable")
        first = np.r_[True, lab[o][1:] != lab[o][:-1]]
        head = np.repeat(cid[o][first], np.diff(np.r_[np.flatnonzero(first), lab.size]))
        for c, h in zip(cid[o], head):
            rc, rh = root(c), root(h)
            if rc != rh:
                up[rc] = rh
    roots = np.array([root(i) for i in range(off[-1])], np.int64)
    gid = [roots[p[5] + off[k]] if p is not None else None for k, p in enumerate(parts)]
    size = np.bincount(np.concatenate([g for g in gid if g is not None]), minlength=off[-1]) if off[-1] else []
    job_of, load = np.zeros(off[-1], np.int64), np.zeros(n_jobs)
    for g in np.argsort(-np.asarray(size)):               # largest groups first, each to the least loaded job
        if size[g] == 0:
            break
        j = int(np.argmin(load))
        job_of[g] = j
        load[j] += size[g]
    jobs = [[] for _ in range(n_jobs)]
    for p, g in zip(parts, gid):
        if p is None:
            continue
        jg = job_of[g]
        for j in np.unique(jg):
            sel = jg == j
            jobs[j].append(tuple(a[sel] for a in p[:5]))
    return [jb for jb in jobs if jb]


def _cluster_parallel(q, args, nproc, block_batches, progress=None):
    """Block-parallel clustering: blocks of consecutive batches are clustered by nproc worker processes, then the
    clusters at block edges, merged in groups that can connect (_edge_jobs), also in parallel. Every
    cluster has the same pixels as in the sequential run, so the same spots, and they are put in the sequential
    output order (closing flush, first pixel, split label)."""
    import multiprocessing as mp
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    import signal
    outs, raws, frs, futs, cur = [], [], [], [], []
    with ctx.Pool(nproc) as pool:
        # terminated by the frame loop (error): take the workers along
        signal.signal(signal.SIGTERM, lambda *a: (pool.terminate(), os._exit(1)))
        def submit():
            n = sum(len(b[0]) for b in cur)

            def done(_):
                if progress is not None:
                    progress.value += n
            futs.append(pool.apply_async(_cluster_block, (args, list(cur), cur[0][0][0] - 1 if futs else None),
                                         callback=done))
        while True:
            msg = q.get()
            if msg is None:
                break
            frs.append(list(msg[0]))
            cur.append(msg)
            if len(cur) == block_batches:
                submit()
                cur = []
        if cur:
            submit()
        for f in futs:
            o, r, _n = f.get()
            outs += o
            raws.append(r)
        for f in [pool.apply_async(_merge_group, (args, g)) for g in _edge_jobs(args, raws, 2 * nproc)]:
            outs += f.get()
    keys = ("x", "y", "z", "intensity", "npix", "zmin", "zmax")
    if not outs:
        e = np.zeros(0)
        return SpotList(e, e, e, e, e.astype(int), e, e)
    d = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    ev = np.searchsorted(_flush_thresholds(frs), d["pz"], side="right")
    order = np.lexsort((d["sub"], d["pk"], ev))
    return SpotList(**{k: d[k][order] for k in keys})


def _cluster_child(q, conn, args, nproc=1, block_batches=16, progress=None):
    """Clustering process (XPG_SPOTS_PROCESS): the streaming accumulator over the batches arriving on q (None ends),
    the spot table back on conn; progress (a shared integer) counts the frames clustered."""
    try:
        conn.send(("ready", None))
        if nproc > 1:
            conn.send(("ok", _cluster_parallel(q, args, nproc, block_batches, progress).__dict__))
            return
        acc = _Accumulator(*args)
        state = dict(prev=None)
        while True:
            msg = q.get()
            if msg is None:
                break
            _cluster_batch(acc, state, *msg)
            if progress is not None:
                progress.value += len(msg[0])
        acc.flush()
        conn.send(("ok", acc.result().__dict__))
    except BaseException:
        import traceback
        conn.send(("error", traceback.format_exc()))

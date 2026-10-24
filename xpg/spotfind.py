"""Spot finding, GPU-accelerated with PyTorch.

Per-pixel classification uses the "dispersion" criterion (Kabsch 2010) on a local window
(Kabsch 2010; also used by DIALS):

    strong  <=>  var/mean > gain * (1 + n_b * sqrt(2/(n-1)))      (window not Poissonian)
            and  I - mean > n_s * sqrt(gain * mean)                 (pixel significantly high)

with an extended second pass that recomputes the local statistics after excluding (dilated)
strong pixels, which removes the halo bias around strong spots and gives cleaner peaks.
Strong pixels are then clustered in 3D (x, y, frame) (spotcluster.py).
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as tF
import numpy as np

from .backend import compiled, image_device
from .geometry import PIX0


def _box_sum(a, hy, hx):
    """Sum over a (2hy+1)x(2hx+1) window, zero padded; a has shape (B, ny, nx). Pixel values are counts, so the
    float32 sums are exact whatever the summation order."""
    x = a[:, None]
    x = tF.avg_pool2d(x, (1, 2 * hx + 1), stride=1, padding=(0, hx), count_include_pad=True, divisor_override=1)
    x = tF.avg_pool2d(x, (2 * hy + 1, 1), stride=1, padding=(hy, 0), count_include_pad=True, divisor_override=1)
    return x[:, 0]


def _dilate(m, r=1):
    """Binary dilation by a (2r+1)^2 square: separable box count > 0."""
    return _box_sum(m.to(torch.float32), r, r) > 0


def _dispersion(img, valid, gain, hy, hx, n_b, n_s, min_local):
    w = valid.to(torch.float32)
    x = torch.where(valid, img, 0.0)
    n = _box_sum(w, hy, hx)
    s1 = _box_sum(x, hy, hx)
    s2 = _box_sum(x * x, hy, hx)
    nn = torch.clamp(n, min=2.0)
    mean = s1 / nn
    var = torch.clamp((s2 - s1 * mean) / (nn - 1.0), min=0.0)
    disp_thr = gain * (1.0 + n_b * torch.sqrt(2.0 / (nn - 1.0)))
    disp = var > disp_thr * torch.clamp(mean, min=1e-6)
    strong = (x - mean) > n_s * torch.sqrt(gain * torch.clamp(mean, min=1.0))
    ok = valid & (n >= min_local)
    return ok & disp & strong, mean


DISPERSION_SIGMA_B = 6.0   # DIALS dispersion criterion (not BACKGROUND_PIXEL)


def make_finder(nbx=3, nby=3, signal_pixel=3.0, background_pixel=2.0, min_local=2, extended=True):
    """Per-pixel strong-pixel classifier.

    Pass 1: dispersion test (Kabsch 2010 / DIALS) with sigma_b = 6 and sigma_s = SIGNAL_PIXEL.
    Pass 2 (extended): local background re-estimated in a (2*NB+1)-larger window from pixels
    that are (a) not near pass-1 signal and (b) background pixels, i.e. below
    mean + BACKGROUND_PIXEL * sigma; a pixel within 2 px of a pass-1 spot is strong if it exceeds
    that background by SIGNAL_PIXEL * sigma (sigma from Poisson x gain, or the observed scatter if larger).
    """
    hx, hy = int(nbx), int(nby)

    @torch.no_grad()
    def find(img, gain):
        valid = img >= 0
        s1, mean1 = _dispersion(img, valid, gain, hy, hx, DISPERSION_SIGMA_B, signal_pixel, min_local)
        if not extended:
            return s1, mean1
        sig1 = torch.sqrt(gain * torch.clamp(mean1, min=0.5))
        bgpix = valid & ~_dilate(s1, 1) & (img <= mean1 + background_pixel * sig1)
        w = bgpix.to(torch.float32)
        x = torch.where(bgpix, img, 0.0)
        hy2, hx2 = 2 * hy + 1, 2 * hx + 1
        n = _box_sum(w, hy2, hx2)
        m1 = _box_sum(x, hy2, hx2)
        m2 = _box_sum(x * x, hy2, hx2)
        nn = torch.clamp(n, min=2.0)
        mean = m1 / nn
        var = torch.clamp((m2 - m1 * mean) / (nn - 1.0), min=0.0)
        sig = torch.sqrt(torch.maximum(gain * torch.clamp(mean, min=0.5), var))
        strong = valid & (n >= 4 * min_local) & ((img - mean) > signal_pixel * sig)
        # pass 2 only grows spots found by the dispersion test: on low-background data the plain
        # significance test alone flags every 3-photon pixel (~0.5% of all pixels at 0.35 counts)
        return strong & _dilate(s1, 2), mean

    return find


from .spotcluster import SpotList, _Accumulator, _cluster_batch, _cluster_child  # noqa: F401
from .spotcluster import PIX0 as _PIX0_CLUSTER

assert _PIX0_CLUSTER == PIX0


SPOTS_PROCESS = os.environ.get("XPG_SPOTS_PROCESS", "1") != "0"


def _cluster_procs(dev):
    """Clustering worker processes (XPG_SPOTS_CLUSTER_PROCS). On CUDA the frame loop outruns one clustering process
    once the GPU undoes the bit shuffle: block-parallel clustering with up to 3 workers, the same spot table.
    Elsewhere one."""
    v = os.environ.get("XPG_SPOTS_CLUSTER_PROCS")
    if v is not None:
        return max(int(v), 1)
    if dev.type != "cuda":
        return 1
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 2
    return min(3, max(1, (ncpu - 2) // 2))


def find_spots(images, frames, gain=1.0, mask=None, strong_pixel=3.0, background_pixel=2.0,
               min_pixels=6, max_pixels=5000, nbx=3, nby=3, batch=8, extended=True, verbose=True,
               max_centroid_dist=3.0, cluster_radius=2, log=print):
    """Find spots on the given frames (list of frame numbers, 1-based).

    Spot z coordinates: frame i covers [i-1, i). ``mask``: True = trusted pixel.
    """
    dev = image_device()
    finder = compiled(make_finder(nbx, nby, strong_pixel, background_pixel, extended=extended), "spot finder", dev)
    acc = _Accumulator(images.nx, images.ny, min_pixels, max_pixels, max_centroid_dist, cluster_radius)
    mask_j = torch.as_tensor(np.asarray(mask), device=dev) if mask is not None else None
    frames = list(frames)
    t0 = time.time()
    # one device buffer per batch: frames go from the decoder's shared memory straight to the device
    a = torch.full((batch, images.ny, images.nx), -1.0, dtype=torch.float32, device=dev)
    finder(a, float(gain))           # compile (CPU / MPS) before the decode workers occupy the cores
    # clustering of strong pixels (CPU) overlaps decoding and the device: in a separate process for longer runs
    # (XPG_SPOTS_PROCESS, default on; the batches go through a queue, so the frame loop never waits for it),
    # otherwise in a background thread
    from concurrent.futures import ThreadPoolExecutor
    proc = None
    if SPOTS_PROCESS and len(frames) >= 100:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        q = ctx.Queue()
        conn, child_conn = ctx.Pipe(duplex=False)
        ncl = _cluster_procs(dev)
        # a daemonic process cannot start the block-clustering workers; the frame loop terminates it on any error
        proc = ctx.Process(target=_cluster_child, args=(q, child_conn, (images.nx, images.ny, min_pixels, max_pixels,
                                                                      max_centroid_dist, cluster_radius), ncl),
                           daemon=ncl == 1)
        # the child needs nothing of the calling script: with its path / module name hidden, spawn does not run it
        # again in the child (a script without an `if __name__ == "__main__":` guard would start the whole pipeline
        # there, writing into the same run directory)
        main = sys.modules.get("__main__")
        saved = None
        if ctx.get_start_method() != "fork" and main is not None:
            saved = (main.__dict__.pop("__file__", None), getattr(main, "__spec__", None))
            main.__spec__ = None
        try:
            proc.start()
        finally:
            if saved is not None:
                if saved[0] is not None:
                    main.__file__ = saved[0]
                main.__spec__ = saved[1]
        child_conn.close()
    cluster = ThreadPoolExecutor(1) if proc is None else None
    jobs = []
    state = dict(prev=None, nstrong=0)
    ready, pending = False, []

    def cluster_batch(*batch):
        _cluster_batch(acc, state, *batch)

    def child_started(wait=False):
        """True once the clustering process runs. If it dies before (spawn re-imports the calling script, which fails
        without an `if __name__ == "__main__":` guard), clustering falls back to the thread with the batches sent so
        far."""
        nonlocal proc, cluster, ready
        while not ready:
            dead = False
            if conn.poll(1.0 if wait else 0):
                try:
                    conn.recv()
                    ready = True
                    pending.clear()
                    continue
                except EOFError:                      # closed before the start message: the process is gone
                    dead = True
            if dead or not proc.is_alive():
                log("  SPOTS: the clustering process did not start (see its error above); clustering in a thread")
                q.cancel_join_thread()
                proc = None
                cluster = ThreadPoolExecutor(1)
                jobs.extend(cluster.submit(cluster_batch, *b) for b in pending)
                pending.clear()
                return False
            elif not wait:
                return False
        return True

    def process(fr):
        n = len(fr)
        if n < batch:                    # fixed batch shape (one kernel compilation); padding frames are untrusted
            a[n:] = -1.0
        x = torch.where(mask_j[None], a, -1.0) if mask_j is not None else a
        strong, mean = finder(x, float(gain))
        # strong pixels are ~0.03% of the image: extract them on the device and move only those
        kk, yy, xx = torch.nonzero(strong, as_tuple=True)
        val, bg = x[kk, yy, xx], mean[kk, yy, xx]
        kk, yy, xx, val, bg = (t.cpu().numpy() for t in (kk, yy, xx, val, bg))
        state["nstrong"] += kk.size
        if proc is not None:
            q.put((list(fr), kk, yy, xx, val, bg))
            if not ready:
                pending.append((list(fr), kk, yy, xx, val, bg))
                child_started()
            return
        if jobs and jobs[-1].done():
            jobs.pop().result()                      # surface clustering errors early
        jobs.append(cluster.submit(cluster_batch, list(fr), kk, yy, xx, val, bg))

    # Eiger bitshuffle-LZ4 frames on CUDA: the decode workers only undo the LZ4 stage and the device the bit shuffle,
    # most of the decoding cost
    bsh = None
    if dev.type == "cuda" and os.environ.get("XPG_GPU_UNSHUFFLE", "1") != "0":
        bsh = images.iter_frames_bitshuffled(frames)
    if bsh is not None:
        from .io.bslz4 import unshuffle_frame_torch
        esize, nel = images.dtype.itemsize, images.ny * images.nx
        raw_d = torch.empty(nel * esize, dtype=torch.uint8, device=dev)
        sbad = (torch.as_tensor(~np.asarray(images.static_mask), device=dev).reshape(-1)
                if images.static_mask is not None else None)
    batch_frames = []
    try:
        for i, (f, img, *blk) in enumerate(bsh if bsh is not None else images.iter_frames(frames, copy=False)):
            if blk and blk[0]:
                raw_d.copy_(torch.from_numpy(img[:nel * esize]))
                a[len(batch_frames)].view(-1).copy_(unshuffle_frame_torch(
                    raw_d, nel, esize, blk[0], images.overload, images.min_valid, sbad))
            else:
                a[len(batch_frames)].copy_(torch.from_numpy(np.ascontiguousarray(img, np.float32)))
            batch_frames.append(f)
            if len(batch_frames) == batch:
                process(batch_frames)
                batch_frames = []
            if verbose and (i + 1) % 200 == 0:
                log(f"  SPOTS: {i + 1}/{len(frames)} frames, {state['nstrong']} strong pixels, "
                    f"{(i + 1) / (time.time() - t0):.1f} frames/s")
        if batch_frames:
            process(batch_frames)
        if proc is not None:
            child_started(wait=True)
        for j in jobs:
            j.result()
        if proc is not None:
            q.put(None)
            status, payload = conn.recv()
            proc.join()
            if status != "ok":
                raise RuntimeError("spot clustering process failed:\n" + payload)
    finally:
        if cluster is not None:
            cluster.shutdown(wait=True)
        if proc is not None and proc.is_alive():
            proc.terminate()
        if proc is not None and not ready:
            q.cancel_join_thread()                    # nobody reads the queue: do not block the exit on its feeder
    nstrong_total = state["nstrong"]
    if proc is not None:
        spots = SpotList(**payload)
    else:
        acc.flush()
        spots = acc.result()
    if verbose:
        log(f"  SPOTS: {len(spots)} spots from {nstrong_total} strong pixels in {time.time() - t0:.1f}s")
    return spots

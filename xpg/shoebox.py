"""Shoebox buffers of the integration: shared-memory buffers handed to the gather process, the host-memory
budget and the plan of passes over the frames, pinned host-to-GPU upload, and decoding of the stored values."""
from __future__ import annotations

import numpy as np
import torch

from . import tnp as jnp


SB_OFF = np.float32(3.0)   # shoebox buffers store value + 3: unrecorded (-3) = 0 = lazily mapped zero pages


# separate gather process (XPG_GATHER_PROCESS=1): shoebox buffers live in anonymous shared-memory files whose
# descriptors are passed to the computing process over a Unix socket (SCM_RIGHTS)
_FD_BATCH = 200                  # descriptors per message (the kernel limit is 253)


def _shared_zeros(shape, dtype):
    """(array, fd): a zero-filled array backed by an anonymous shared-memory file (pages allocated on first write,
    as np.zeros); the descriptor can be passed to another process and mapped there (_map_fd)."""
    import os
    if hasattr(os, "memfd_create"):
        fd = os.memfd_create("xpg-shoebox", os.MFD_CLOEXEC)
    else:                                       # macOS (tests): an unlinked temporary file
        import tempfile
        fd, path = tempfile.mkstemp(prefix="xpg-shoebox-")
        os.unlink(path)
    os.ftruncate(fd, max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1))
    return _map_fd(fd, shape, dtype), fd


def _map_fd(fd, shape, dtype):
    """Array view of a shared-memory file (mapped while the array lives; the descriptor may be closed)."""
    import mmap
    mm = mmap.mmap(fd, max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1))
    return np.ndarray(shape, dtype, buffer=mm)


def _send_fds(sock, fds):
    import socket
    for i in range(0, len(fds), _FD_BATCH):
        socket.send_fds(sock, [b"F"], fds[i:i + _FD_BATCH])


def _recv_fds(sock, n):
    import socket
    fds = []
    while len(fds) < n:
        _, got, _, _ = socket.recv_fds(sock, 1, min(n - len(fds), _FD_BATCH))
        if not got:
            raise RuntimeError("shoebox gather process: descriptor transfer failed")
        fds += got
    return fds


def _job_memory():
    """(limit, used) bytes of the tightest cgroup memory limit on this process's cgroup path (v2: memory.max /
    memory.current, walking up to the root; v1: the memory controller's limit / usage); (None, None) if unknown."""
    import os
    try:
        lines = open("/proc/self/cgroup").read().splitlines()
    except OSError:
        return None, None
    best = (None, None)

    def take(lim_f, use_f):
        nonlocal best
        try:
            lim = open(lim_f).read().strip()
            use = int(open(use_f).read().strip())
        except (OSError, ValueError):
            return
        try:            # reclaimable page cache (the image files read) is not memory the job is short of
            for ln_ in open(os.path.join(os.path.dirname(lim_f), "memory.stat")):
                k_, v_ = ln_.split()[:2]
                if k_ in ("inactive_file", "total_inactive_file"):
                    use = max(use - int(v_), 0)
                    break
        except (OSError, ValueError):
            pass
        if lim.isdigit() and int(lim) < 2 ** 60 and (best[0] is None or int(lim) - use < best[0] - best[1]):
            best = (int(lim), use)
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) < 3:
            continue
        if parts[0] == "0" and parts[1] == "":                      # v2 unified
            d = parts[2].strip("/")
            while True:
                base = os.path.join("/sys/fs/cgroup", d)
                take(os.path.join(base, "memory.max"), os.path.join(base, "memory.current"))
                if not d:
                    break
                d = os.path.dirname(d)
        elif "memory" in parts[1].split(","):                        # v1
            base = os.path.join("/sys/fs/cgroup/memory", parts[2].strip("/"))
            take(os.path.join(base, "memory.limit_in_bytes"), os.path.join(base, "memory.usage_in_bytes"))
    return best


def _gather_process_ok() -> bool:
    """Separate gather process wanted (XPG_GATHER_PROCESS; default on Linux) and possible (fork, descriptor passing)."""
    import os
    import socket
    # default on where shared-memory files exist; XPG_GATHER_PROCESS=0: gather threads in this process, =1: also
    # elsewhere (macOS: file-backed)
    if os.environ.get("XPG_GATHER_PROCESS", "1" if hasattr(os, "memfd_create") else "0") != "1":
        return False
    import multiprocessing as mp
    return "fork" in mp.get_all_start_methods() and hasattr(socket, "send_fds") and hasattr(os, "fork")


INTEGRATE_MEM_FRAC = 0.7          # share of the available host memory the shoebox buffers may use
KERNEL_RESERVE = 2.0 * 2 ** 30     # kernel temporaries (chunk of ~1.5e7 voxels), frame buffers, outputs


def _integrate_budget():
    """Host bytes the shoebox buffers of one integration may use: XPG_INTEGRATE_MEM_GB, else INTEGRATE_MEM_FRAC of the
    memory available now (macOS free + inactive + purgeable pages; Linux MemAvailable, within the job's cgroup limit
    net of what the job already uses); None if unknown."""
    import os
    gb = os.environ.get("XPG_INTEGRATE_MEM_GB")
    if gb:
        return float(gb) * 2 ** 30
    from .forward.model import host_memory_available
    avail = host_memory_available()
    lim, used = _job_memory()
    if lim is not None:
        avail = min(avail, lim - used) if avail else lim - used
    return INTEGRATE_MEM_FRAC * avail if avail else None


def _plan_passes(order, blocks_refl, lattice_of, need_hi, t_row, w_row, n_frames, budget, max_passes=16):
    """Passes over the frames that keep the shoebox buffers within budget (bytes).

    The buffers are mapped lazily: a reflection's rows cost memory from its first frame (t_row, 0-based within the
    sweep) and are held until the block's last reflection is complete (need_hi); the block is then queued and
    integrated (the CPU kernel path copies each chunk: ~2x its buffers). One loop over a set of blocks peaks at about
    max over frames of the open rows + 2x the largest block + KERNEL_RESERVE. Blocks stay open over their widest
    reflection, so extra lattices (blocks over the same frames) and very wide reflections multiply what is open at
    once. When one loop does not fit, the blocks are split over passes, each reading its frames again: by lattice
    first, then round-robin along phi within each lattice. Returns (passes: block lists in the order of `order`,
    dict(peak / open / block: bytes of the worst pass's peak, open rows and largest block; one: peak of a single pass;
    plan: description))."""
    S = {b: float(w_row[blocks_refl[b]].sum()) for b in order}

    def peak(blist):
        rows = np.concatenate([blocks_refl[b] for b in blist])
        d = np.bincount(t_row[rows], w_row[rows], minlength=n_frames + 1)[:n_frames + 1]
        rel = np.clip([need_hi[b] + 1 for b in blist], 0, n_frames)
        d = d - np.bincount(rel, [S[b] for b in blist], minlength=n_frames + 1)
        op, sm_ = float(np.cumsum(d).max()), max(S[b] for b in blist)
        return op + 2 * sm_ + KERNEL_RESERVE, op, sm_

    lat = [lattice_of[b] for b in order]
    lats = sorted(set(lat))
    phi_rank, seen = [], {}
    for l_ in lat:
        phi_rank.append(seen.get(l_, 0))
        seen[l_] = phi_rank[-1] + 1
    cands = [("one pass", [0] * len(order))]
    cands += [(f"{k} passes by lattice", [lats.index(l_) % k for l_ in lat]) for k in range(2, len(lats) + 1)]
    cands += [(f"{len(lats) * m} passes ({'per lattice, ' if len(lats) > 1 else ''}phi blocks {m} apart)",
               [lats.index(l_) * m + r_ % m for l_, r_ in zip(lat, phi_rank)])
              for m in range(2, max_passes // len(lats) + 1)]
    tried = []
    for name, key in cands:
        passes = [[b for b, k_ in zip(order, key) if k_ == p] for p in range(max(key) + 1)]
        passes = [p for p in passes if p]
        tried.append((passes, max((peak(p) for p in passes), key=lambda v: v[0]), name))
        if budget is None or tried[-1][1][0] <= budget:
            break
    else:
        # nothing fits: the fewest passes that reach 95% of the possible saving (more passes re-read the frames for
        # little gain when the excess is mostly the fixed reserve)
        lo_ = min(t[1][0] for t in tried)
        cut = lo_ + 0.05 * (tried[0][1][0] - lo_)
        p_, w_, n_ = next(t for t in tried if t[1][0] <= cut)
        tried.append((p_, w_, n_ + " (still above the budget)"))
    passes, worst, name = tried[-1]
    return passes, dict(peak=worst[0], open=worst[1], block=worst[2], one=tried[0][1][0], plan=name)


GATHER_CHUNK = 1024   # shoebox rows per gather task (threads: XPG_GATHER_THREADS, default min(8, CPUs))


class _Stager:
    """Host-to-GPU upload of shoebox chunks through two pinned staging buffers (copy into one while the other's
    transfer runs): pageable copies were ~40% of the integration's GPU time."""

    def __init__(self, dev):
        self.dev = dev
        self.on = dev.type == "cuda"
        self.bufs, self.events, self.k = [None, None], [None, None], 0

    def upload(self, a):
        if not self.on:
            return torch.from_numpy(a).to(self.dev)
        k = self.k
        self.k ^= 1
        n = a.size
        if self.bufs[k] is None or self.bufs[k].numel() < n:
            if self.events[k] is not None:
                self.events[k].synchronize()
            self.bufs[k] = torch.empty(n, dtype=torch.float32, pin_memory=True)
            self.events[k] = None
        if self.events[k] is not None:
            self.events[k].synchronize()           # this buffer's previous transfer has finished
        h = self.bufs[k][:n].view(a.shape)
        np.copyto(h.numpy(), a)
        d = torch.empty(a.shape, dtype=torch.float32, device=self.dev)
        d.copy_(h, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.events[k] = ev
        return d


class _HostChunk:
    """Shoebox rows [c0, c0 + m) of a block buffer (stored as value + SB_OFF), padded with copies of the last row to
    m + padn rows: moved to the kernel device, padded and offset there each time a pass needs it (no host copy of
    the chunk is made or held for the block: large blocks exceed the device budget)."""
    __slots__ = ("arr", "c0", "m", "padn", "dev", "stager")

    def __init__(self, arr, c0, m, padn, dev, stager=None):
        self.arr, self.c0, self.m, self.padn, self.dev = arr, c0, m, padn, dev
        self.stager = stager

    @property
    def nbytes(self):
        return self.arr[self.c0:self.c0 + self.m].nbytes

    def device(self):
        a_ = self.arr[self.c0:self.c0 + self.m]
        t_ = self.stager.upload(a_) if self.stager is not None else torch.from_numpy(a_).to(self.dev)
        if self.padn:
            t_ = torch.cat([t_, t_[-1:].expand(self.padn, *t_.shape[1:])])
        return t_.sub_(float(SB_OFF))

    def __getitem__(self, rows):
        return self.device()[jnp.asarray(rows)]


def _sb_arg(sb):
    """Kernel argument for a shoebox chunk (device tensor, host array or _HostChunk)."""
    return sb.device() if isinstance(sb, _HostChunk) else jnp.asarray(sb)


def _decode_bins(arr, cnt):
    """Binned shoebox buffer from raw IEEE-encoded sums (+inf masked, NaN overloaded, 0 = no frame yet) to the stored
    convention, in place. One multi-threaded pass."""
    torch.from_numpy(arr).nan_to_num_(nan=-2.0, posinf=-1.0, neginf=-1.0).add_(float(SB_OFF))
    arr[cnt == 0] = 0.0

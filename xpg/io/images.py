"""Image access: CBF and SMV (ADSC-style .img) frame series and Eiger/NeXus HDF5 master files.

All readers return float32 arrays of shape (ny, nx) where untrusted pixels (detector
mask, gaps, values below MINIMUM_VALID_PIXEL_VALUE) are set to -1 and overloaded pixels
(>= OVERLOAD) to -2.
"""
from __future__ import annotations

import glob
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import bslz4
from .cbf import read_cbf

try:  # registers bitshuffle/LZ4 filters
    import hdf5plugin  # noqa: F401
except Exception:  # pragma: no cover
    pass


def resolve_template(template: str, workdir: Path | str = ".") -> str:
    t = template
    if not os.path.isabs(t):
        t = os.path.normpath(os.path.join(str(workdir), t))
    return t


class ImageSet:
    nx: int
    ny: int
    first: int
    last: int
    static_mask: np.ndarray | None = None  # True = trusted

    def read_raw(self, frame: int) -> np.ndarray:
        raise NotImplementedError

    def _read_native(self, frame: int):
        """(raw, invalid): the frame as stored and a mask of pixels flagged invalid by the format (None: read_raw's
        array, where -1 marks them)."""
        return self.read_raw(frame), None

    def __init__(self, min_valid=0.0, overload=None):
        self.min_valid = min_valid
        self.overload = overload
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.cache_size = 8
        self.offset = 0.0
        self.dark = None      # (ny, nx) float32

    def get(self, frame: int, out=None) -> np.ndarray:
        """The frame as float32 (-1 masked / invalid, -2 overloaded). out: a float32 (ny, nx) array, possibly a view
        (decode workers: the shared frame slot), written instead of a new cached image."""
        with self._lock:
            if frame in self._cache:
                self._cache.move_to_end(frame)
                if out is None:
                    return self._cache[frame]
                np.copyto(out, self._cache[frame])
                return out
        if not self.available(frame):        # missing / corrupt data: fully untrusted frame
            if out is None:
                return np.full((self.ny, self.nx), -1.0, np.float32)
            out[...] = -1.0
            return out
        raw, flagged = self._read_native(frame)
        if out is None:
            img = raw.astype(np.float32)
        else:
            img = out
            np.copyto(img, raw, casting="unsafe")
        # unsigned counts (Eiger): compared in their own dtype, no 8-byte signed copy per frame (decode-bound sweeps)
        if flagged is None:
            bad = raw < self.min_valid
        elif self.min_valid > 0:
            bad = (raw < self.min_valid) | flagged
        else:
            bad = flagged
        # non-X-ray background (CCD/CMOS dark current): OFFSET= and DARK_CURRENT_IMAGE=
        if self.offset:
            img -= np.float32(self.offset)
        if self.dark is not None:
            img -= self.dark
        if self.static_mask is not None:
            bad |= ~self.static_mask
        if self.overload is not None:
            np.copyto(img, np.float32(-2.0), where=raw >= self.overload)   # overloaded (distinguished from masked)
        np.copyto(img, np.float32(-1.0), where=bad)
        if out is None:
            with self._lock:
                self._cache[frame] = img
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return img

    # pickling (decode worker processes): file handles, locks and the cache are rebuilt on the other side
    _transient = ("_lock", "_cache", "_tl", "_h5")

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._transient}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._tl = threading.local()
        self._h5 = None

    def iter_frames(self, frames, nthreads=4, prefetch=8, copy=True):
        """Yield (frame, image) in order. Decompression (HDF5 bitshuffle-LZ4, CBF byte offset) is CPU-bound and
        serialised by h5py's global lock, so large sweeps are decoded by worker processes into shared memory
        (XPG_DECODE_PROCS, default min(16, CPUs - 2); 0 or 1 = threads in this process).
        copy=False: the image may be a view of a shared buffer, valid only until the next frame is requested
        (saves a 4-byte-per-pixel copy per frame for consumers that move it to the GPU at once)."""
        frames = list(frames)
        nproc = _decode_procs()
        if nproc > 1 and len(frames) > 4 * nproc:
            yield from _iter_frames_mp(self, frames, nproc, copy=copy)
            return
        with ThreadPoolExecutor(nthreads) as ex:
            futs = OrderedDict()
            it = iter(frames)
            for f in it:
                futs[f] = ex.submit(self.get, f)
                if len(futs) >= prefetch:
                    break
            while futs:
                f, fut = futs.popitem(last=False)
                img = fut.result()
                nxt = next(it, None)
                if nxt is not None:
                    futs[nxt] = ex.submit(self.get, nxt)
                yield f, img

    def padded_ok(self, n_frames) -> bool:
        """iter_frames_padded serves n_frames (they are decoded by worker processes)."""
        nproc = _decode_procs()
        return nproc > 1 and n_frames > 4 * nproc

    def iter_frames_padded(self, frames, pad, mask=None):
        """(frame, padded frame): as iter_frames(copy=False), but each frame is written by the decode workers inside a
        pad-pixel border of -1, masked pixels (mask False) at min(value, -1) (an overload keeps -2), i.e. the
        integrator's padded frame, gathered from without a per-frame copy. None when padded_ok is False."""
        frames = list(frames)
        if not self.padded_ok(len(frames)):
            return None
        return _iter_frames_mp(self, frames, _decode_procs(), copy=False, layout=(int(pad), mask))

    def bitshuffled_ok(self) -> bool:
        """Frames are bitshuffle-LZ4 chunks the device can finish (iter_frames_bitshuffled)."""
        return False

    def iter_frames_bitshuffled(self, frames):
        """(frame, buffer, block) from decode workers that only undo the LZ4 stage: block > 0: the frame's bit-shuffled
        bytes (uint8 view of a shared slot, valid until the next frame is requested), to be finished on the device by
        bslz4.unshuffle_frame_torch; block 0: the finished float32 frame (a frame the workers decoded fully: missing,
        or not a plain chunk). None when not available (bitshuffled_ok, padded_ok)."""
        frames = list(frames)
        if not (self.bitshuffled_ok() and self.padded_ok(len(frames))):
            return None
        return _iter_frames_mp(self, frames, _decode_procs(), copy=False, bitshuffled=True)

    def available(self, frame) -> bool:
        return True

    def stack(self, frames, nthreads=4):
        return np.stack([im for _, im in self.iter_frames(frames, nthreads)])


_SMV_TYPES = {"unsigned_short": "u2", "unsigned_int": "u4", "unsigned_long": "u4", "signed_short": "i2",
              "signed_int": "i4", "signed_long": "i4", "float": "f4"}


def is_smv(path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"HEADER_BYTES=" in fh.read(64)
    except OSError:
        return False


def read_smv(path, header_only=False):
    """SMV image ('{ HEADER_BYTES=n; KEY=value;... }' then raw pixels: ADSC, also Rigaku / MAR CCD exports): (data,
    header). A repeated key counts with its last value. The header gets nx / ny / count_cutoff like a CBF header."""
    with open(path, "rb") as fh:
        head = fh.read(512)
        m = re.search(rb"HEADER_BYTES=\s*(\d+)", head)
        if not m:
            raise ValueError(f"{path}: not an SMV image (no HEADER_BYTES)")
        hb = int(m.group(1))
        if hb > len(head):
            head += fh.read(hb - len(head))
        h = {k: v.strip() for k, v in re.findall(r"(\w+)\s*=\s*([^;]*);", head[:hb].decode("latin-1"))}
        h["nx"], h["ny"] = int(h["SIZE1"]), int(h["SIZE2"])
        h["count_cutoff"] = float(h.get("CCD_IMAGE_SATURATION") or h.get("SATURATED_VALUE") or 65535)
        if header_only:
            return None, h
        dt = np.dtype(_SMV_TYPES.get(h.get("TYPE", "unsigned_short").lower(), "u2"))
        dt = dt.newbyteorder(">" if h.get("BYTE_ORDER", "little_endian").lower().startswith("big") else "<")
        fh.seek(hb)
        data = np.frombuffer(fh.read(h["nx"] * h["ny"] * dt.itemsize), dt).reshape(h["ny"], h["nx"])
    return data, h


class CBFImageSet(ImageSet):
    _read = staticmethod(read_cbf)

    def __init__(self, template: str, first=None, last=None, **kw):
        super().__init__(**kw)
        self.template = template
        m = re.search(r"\?+", template)
        if not m:
            raise ValueError("template must contain ?-placeholders")
        self._w = len(m.group(0))
        self._pre, self._post = template[:m.start()], template[m.end():]
        nums = sorted(int(Path(f).name[len(Path(self._pre).name):len(Path(self._pre).name) + self._w])
                      for f in glob.glob(self._pre + "[0-9]" * self._w + self._post))
        self.first = first if first is not None else (nums[0] if nums else 1)
        self.last = last if last is not None else (nums[-1] if nums else 1)
        _, self.header = self._read(self.path(self.first), header_only=True)
        self.nx, self.ny = self.header["nx"], self.header["ny"]
        if self.overload is None and self.header.get("count_cutoff"):
            self.overload = self.header["count_cutoff"]

    def path(self, frame):
        return f"{self._pre}{frame:0{self._w}d}{self._post}"

    def read_raw(self, frame):
        data, _ = self._read(self.path(frame))
        return data


class SMVImageSet(CBFImageSet):
    """SMV frame series (template with ?-placeholders, e.g. ADSC ``name_???.img``)."""
    _read = staticmethod(read_smv)


class EigerImageSet(ImageSet):
    """Eiger / NeXus HDF5 master file with external data links (``*_master.h5``)."""

    def __init__(self, master: str, first=None, last=None, **kw):
        super().__init__(**kw)
        import h5py
        self.master = master
        self._h5 = h5py.File(master, "r")
        self._tl = threading.local()
        dg = self._h5["/entry/data"]
        self._blocks = []  # (start_frame, n, key)
        self.bad_blocks = []
        start = 1
        for key in sorted(k for k in dg.keys() if k.startswith("data")):
            try:
                ds = dg[key]
            except (KeyError, OSError):  # dangling external link or unreadable (corrupt) data file
                self.bad_blocks.append(key)
                continue
            lo = ds.attrs.get("image_nr_low", start)
            lo = int(np.asarray(lo).ravel()[0])
            self._blocks.append((lo, ds.shape[0], key))
            start = lo + ds.shape[0]
        if not self._blocks:
            raise ValueError(f"{master}: no data blocks found")
        first_ds = dg[self._blocks[0][2]]
        self.ny, self.nx = first_ds.shape[1:]
        self.dtype = first_ds.dtype
        self._recovered = {}   # key -> list of (offset, length) of the frames' compressed chunks
        self.recovered_blocks = []
        if self.bad_blocks:
            self._recover_bad_blocks(dg, first_ds)
        self.first = first if first is not None else self._blocks[0][0]
        self.last = last if last is not None else self._blocks[-1][0] + self._blocks[-1][1] - 1
        self.header = read_nexus_header(self._h5)
        pm = self.header.get("pixel_mask")
        if pm is not None:
            self.static_mask = pm == 0
        if self.overload is None:
            self.overload = self.header.get("count_cutoff")
            if self.overload is None and np.issubdtype(self.dtype, np.unsignedinteger):
                self.overload = np.iinfo(self.dtype).max

    def _recover_bad_blocks(self, dg, good_ds):
        """Frames of data files HDF5 cannot open (corrupt metadata, truncated copy) recovered by scanning the raw
        file for the per-frame compressed chunks.  Eiger writers store one chunk per frame, back to back in frame
        order, each starting with a fixed header (uint64 uncompressed size, uint32 block size, big endian), which
        is taken from a readable data file.  The k chunks found are frames lo..lo+k-1 of the block (a truncated
        file loses its last frames); a chunk that does not decompress ends the block."""
        if good_ds.chunks is None or good_ds.chunks[0] != 1 or tuple(good_ds.chunks[1:]) != (self.ny, self.nx):
            return
        try:
            off = good_ds.id.get_chunk_info(0).byte_offset
            with open(good_ds.file.filename, "rb") as fh:
                fh.seek(off)
                sig = fh.read(12)
        except Exception:
            return
        if int.from_bytes(sig[:8], "big") != self.ny * self.nx * good_ds.dtype.itemsize:
            return                                  # not a bitshuffle/LZ4-style chunk header
        nominal = max(n for _, n, _ in self._blocks)
        keys = sorted(k for k in dg.keys() if k.startswith("data"))
        lo_of = {k: lo for lo, _, k in self._blocks}
        n_of = {k: n for _, n, k in self._blocks}
        for key in list(self.bad_blocks):
            i = keys.index(key)
            prev = [k for k in keys[:i] if k in lo_of]
            nxt = [k for k in keys[i + 1:] if k in lo_of]
            # frame range of the damaged block from its readable neighbours
            if prev:
                lo = lo_of[prev[-1]] + n_of[prev[-1]] + nominal * (i - keys.index(prev[-1]) - 1)
            else:
                lo = lo_of[nxt[0]] - nominal * (keys.index(nxt[0]) - i)
            n_exp = (lo_of[nxt[0]] - lo - nominal * (keys.index(nxt[0]) - i - 1)) if nxt else nominal
            try:
                link = dg.get(key, getlink=True)
                path = os.path.join(os.path.dirname(self.master), link.filename)
                starts = _scan_signature(path, sig)
            except Exception:
                continue
            if not starts or len(starts) > n_exp or lo < 1:
                continue
            size = os.path.getsize(path)
            chunks = [(s, (starts[j + 1] if j + 1 < len(starts) else size) - s) for j, s in enumerate(starts)]
            good = []
            with open(path, "rb") as fh:            # keep the leading run of chunks that decompress
                for c in chunks:
                    fh.seek(c[0])
                    if not _valid_bslz4(fh.read(c[1]), good_ds.dtype.itemsize):
                        break
                    good.append(c)
            if not good:
                continue
            self._recovered[key] = (path, good, good_ds.dtype)
            self._blocks.append((lo, len(good), key))
            self.bad_blocks.remove(key)
            self.recovered_blocks.append((key, lo, len(good), n_exp))
        self._blocks.sort()

    def _decode_chunk(self, path, chunk, dtype):
        import h5py
        off, length = chunk
        with open(path, "rb") as fh:
            fh.seek(off)
            buf = fh.read(length)
        mem = getattr(self._tl, "mem", None)
        if mem is None or mem[1] != dtype:
            f = h5py.File(f"XPG-chunk-{id(self)}-{threading.get_ident()}", "w", driver="core", backing_store=False)
            ds = f.create_dataset("d", (1, self.ny, self.nx), dtype=dtype, chunks=(1, self.ny, self.nx),
                                  **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
            mem = (f, dtype, ds)
            self._tl.mem = mem
        ds = mem[2]
        ds.id.write_direct_chunk((0, 0, 0), buf)
        return ds[0]

    def available(self, frame) -> bool:
        return any(lo <= frame < lo + n for lo, n, _ in self._blocks)

    def bitshuffled_ok(self) -> bool:
        ok = getattr(self, "_bshuf_ok", None)
        if ok is None:
            # recovered blocks (damaged files) are raw bitshuffle-LZ4 chunks, structurally checked when recovered
            ok = (np.issubdtype(self.dtype, np.unsignedinteger) and self.dtype.itemsize in (2, 4) and not self.offset
                  and self.dark is None and bslz4.available()
                  and all(np.dtype(dt) == self.dtype for _, _, dt in self._recovered.values()))
            for _, _, key in self._blocks if ok else ():
                if key in self._recovered:
                    continue
                ds = self._file()["/entry/data"][key]
                pl = ds.id.get_create_plist()
                filt = [pl.get_filter(i) for i in range(pl.get_nfilters())]
                # one chunk per frame, the bitshuffle filter alone, LZ4 compression (cd_values[4] == 2)
                ok = (ds.chunks == (1, self.ny, self.nx) and ds.dtype == self.dtype and len(filt) == 1
                      and filt[0][0] == bslz4.BSLZ4_FILTER_ID and len(filt[0][2]) >= 5 and filt[0][2][4] == 2)
                if not ok:
                    break
            self._bshuf_ok = bool(ok)
        return ok

    def read_chunk(self, frame):
        """The frame's stored chunk (bytes) when HDF5 applied its filter (or a recovered raw chunk), else None."""
        for lo, n, key in self._blocks:
            if lo <= frame < lo + n:
                if key in self._recovered:
                    path, chunks, _ = self._recovered[key]
                    off, length = chunks[frame - lo]
                    with open(path, "rb") as fh:
                        fh.seek(off)
                        return fh.read(length)
                fmask, buf = self._file()["/entry/data"][key].id.read_direct_chunk((frame - lo, 0, 0))
                return buf if fmask == 0 else None
        return None

    def _file(self):
        import h5py
        f = getattr(self._tl, "f", None)
        if f is None:
            f = h5py.File(self.master, "r")
            self._tl.f = f
        return f

    def _read_native(self, frame):
        for lo, n, key in self._blocks:
            if lo <= frame < lo + n:
                if key in self._recovered:
                    path, chunks, dtype = self._recovered[key]
                    raw = self._decode_chunk(path, chunks[frame - lo], dtype)
                else:
                    raw = self._file()["/entry/data"][key][frame - lo]
                if np.issubdtype(raw.dtype, np.unsignedinteger):
                    return raw, raw == np.iinfo(raw.dtype).max       # saturated / masked pixels are 2^n - 1
                return raw, None
        raise IndexError(frame)

    def read_raw(self, frame):
        raw, flagged = self._read_native(frame)
        if flagged is None:
            return raw
        out = raw.astype(np.int64)
        out[flagged] = -1
        return out


def _decode_procs() -> int:
    v = os.environ.get("XPG_DECODE_PROCS")
    if v is not None:
        return max(int(v), 0)
    try:
        ncpu = len(os.sched_getaffinity(0))      # CPUs this job may use (SLURM / cgroups), not the node's total
    except AttributeError:
        ncpu = os.cpu_count() or 2
    return max(min(16, ncpu - 2), 0)


_W = {}     # decode-worker globals: image set, shared buffer view


def _worker_init(iset, shm_name, nslots, shape, pad=0, mval_name=None):
    from multiprocessing import shared_memory
    iset.cache_size = 0
    shm = shared_memory.SharedMemory(name=shm_name)
    _W.update(iset=iset, shm=shm, buf=np.ndarray((nslots,) + shape, np.float32, buffer=shm.buf), pad=pad, mval=None)
    if mval_name is not None:
        ms = shared_memory.SharedMemory(name=mval_name)
        _W.update(mshm=ms, mval=np.ndarray((iset.ny, iset.nx), np.float32, buffer=ms.buf))


def _worker_decode(frame, slot):
    iset, p = _W["iset"], _W["pad"]
    if not p:
        iset.get(frame, out=_W["buf"][slot])
        return frame
    # padded layout (iter_frames_padded): the frame inside a fixed -1 border, the per-pixel ceiling applied
    dst = iset.get(frame, out=_W["buf"][slot, p:p + iset.ny, p:p + iset.nx])
    if _W["mval"] is not None:
        np.minimum(dst, _W["mval"], out=dst)
    return frame


def _worker_bitshuffled(frame, slot):
    """LZ4 stage of a bitshuffle-LZ4 frame into the slot's bytes; returns the block size in elements, or 0 when the
    slot holds the fully decoded frame instead (missing frame, chunk stored without its filter, other layout)."""
    iset = _W["iset"]
    raw = iset.read_chunk(frame) if iset.available(frame) else None
    if raw is not None:
        try:
            return bslz4.lz4_stage(raw, _W["buf"][slot].reshape(-1).view(np.uint8), iset.dtype.itemsize)
        except ValueError:
            pass
    iset.get(frame, out=_W["buf"][slot])
    return 0


_POOL = {}   # the persistent decode pool: one per process, rebuilt when the image set or worker count changes


def _close_pool():
    P = _POOL.pop("cur", None)
    if P is None:
        return
    try:
        P["pool"].terminate()
        P["pool"].join()
    finally:
        del P["buf"]
        P["shm"].close()
        P["shm"].unlink()
        if P.get("mshm") is not None:
            P["mshm"].close()
            P["mshm"].unlink()


def _get_pool(iset, nproc, layout=None):
    """Decode workers + shared frame ring, kept across calls (the pipeline reads the sweep several times: spot finding,
    sigma estimation, integration, pass 2; spawning 14 workers costs ~5 s each time). layout = (pad, mask): frames
    written inside a pad-pixel border of -1 with masked pixels at min(value, -1) (iter_frames_padded); the pool is
    rebuilt when the layout changes (keyed by the mask object)."""
    import atexit
    import hashlib
    import pickle
    import multiprocessing as mp
    from multiprocessing import shared_memory
    pad, mask = layout if layout is not None else (0, None)
    key = (hashlib.sha1(pickle.dumps(iset)).hexdigest(), int(pad), id(mask) if mask is not None else None)
    P = _POOL.get("cur")
    if P is not None and P["key"] == key and P["nproc"] == nproc and P.get("mask") is mask:
        return P
    _close_pool()
    shape = (iset.ny + 2 * pad, iset.nx + 2 * pad)
    nslots = nproc + 4
    shm = shared_memory.SharedMemory(create=True, size=nslots * shape[0] * shape[1] * 4)
    buf = np.ndarray((nslots,) + shape, np.float32, buffer=shm.buf)
    mshm = None
    if pad:
        buf[...] = -1.0                              # the border stays masked; workers write the interiors
        if mask is not None and not np.asarray(mask).all():
            mshm = shared_memory.SharedMemory(create=True, size=iset.ny * iset.nx * 4)
            np.ndarray((iset.ny, iset.nx), np.float32, buffer=mshm.buf)[...] = np.where(
                np.asarray(mask), np.float32(np.inf), np.float32(-1.0))
    env_keep = {k: os.environ.get(k) for k in ("XPG_DECODE_WORKER", "CUDA_VISIBLE_DEVICES")}
    os.environ["XPG_DECODE_WORKER"] = "1"       # spawned workers inherit this: I/O modules only, no torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # and never a CUDA context (one per worker would cost GPU memory)
    # spawned workers must not re-run the parent's main script (they only need this module): hide it while the
    # pool starts (a main read from stdin / a notebook cannot even be re-imported)
    import sys
    main = sys.modules.get("__main__")
    main_file, main_spec = getattr(main, "__file__", None), getattr(main, "__spec__", None)
    try:
        if main is not None:
            if main_file is not None:
                del main.__file__
            main.__spec__ = None
        pool = mp.get_context("spawn").Pool(nproc, _worker_init, (iset, shm.name, nslots, shape, int(pad),
                                                                 mshm.name if mshm is not None else None))
    finally:
        if main is not None:
            if main_file is not None:
                main.__file__ = main_file
            main.__spec__ = main_spec
        for k, v in env_keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if not _POOL.get("atexit"):
        atexit.register(_close_pool)
        _POOL["atexit"] = True
    P = dict(key=key, nproc=nproc, pool=pool, shm=shm, buf=buf, nslots=nslots, mshm=mshm, mask=mask)
    _POOL["cur"] = P
    return P


def _iter_frames_mp(iset, frames, nproc, copy=True, layout=None, bitshuffled=False):
    """(frame, image) in order; images decoded by ``nproc`` spawned processes into a shared ring of float32 frames
    (copy=True: copied out, consumers may keep them; copy=False: a view valid until the next frame is requested).
    bitshuffled: (frame, buffer, block), see ImageSet.iter_frames_bitshuffled."""
    import multiprocessing as mp
    P = _get_pool(iset, nproc, layout)
    pool, buf, nslots = P["pool"], P["buf"], P["nslots"]
    pending = OrderedDict()
    free = list(range(nslots))
    it = iter(frames)

    def submit():
        f = next(it, None)
        if f is not None:
            slot = free.pop()
            pending[f] = (slot, pool.apply_async(_worker_bitshuffled if bitshuffled else _worker_decode, (f, slot)))
    try:
        for _ in range(nslots):
            submit()
        lent = None                                # slot handed out as a view (copy=False): freed on the next request
        while pending:
            f, (slot, res) = pending.popitem(last=False)
            try:
                block = res.get(timeout=600)       # a dead / stuck worker must not hang the pipeline
            except mp.TimeoutError:
                _close_pool()
                raise RuntimeError(f"image decoding of frame {f} timed out (decode worker died?); "
                                   "set XPG_DECODE_PROCS=0 to decode in-process")
            if lent is not None:
                free.append(lent)
                submit()
                lent = None
            if copy:
                img = buf[slot].copy()
                free.append(slot)
                submit()
            else:
                img, lent = buf[slot], slot
            if bitshuffled:
                yield f, (img.reshape(-1).view(np.uint8) if block else img), block
            else:
                yield f, img
    finally:
        # a consumer that stops early leaves decodes in flight: let them finish before the ring is reused
        for _, res in pending.values():
            res.wait(timeout=600)
        if any(not r.ready() for _, r in pending.values()) and "cur" in _POOL:
            _close_pool()


def _valid_bslz4(buf, elem_size) -> bool:
    """Full structural check of a bitshuffle/LZ4 chunk (every LZ4 block decoded by python-lz4, which raises on
    malformed input).  The bitshuffle C filter has no bounds checks and crashes on damaged chunks, so recovered
    chunks only reach it after passing this test."""
    import lz4.block
    if len(buf) < 12:
        return False
    total = int.from_bytes(buf[:8], "big")
    bs_bytes = int.from_bytes(buf[8:12], "big")
    if total % elem_size or bs_bytes == 0 or bs_bytes % elem_size:
        return False
    n, bs = total // elem_size, bs_bytes // elem_size
    last = (n % bs) - (n % bs) % 8
    sizes = [bs * elem_size] * (n // bs) + ([last * elem_size] if last else [])
    pos = 12
    try:
        for want in sizes:
            if pos + 4 > len(buf):
                return False
            c = int.from_bytes(buf[pos:pos + 4], "big")
            pos += 4
            if pos + c > len(buf):
                return False
            if len(lz4.block.decompress(buf[pos:pos + c], uncompressed_size=want)) != want:
                return False
            pos += c
    except Exception:
        return False
    return pos + (n % 8) * elem_size <= len(buf)


def _scan_signature(path, sig, piece=1 << 26, min_gap=12):
    """Byte offsets of every non-overlapping occurrence of ``sig`` in a file."""
    starts = []
    with open(path, "rb") as fh:
        base, tail = 0, b""
        while True:
            buf = fh.read(piece)
            if not buf:
                break
            data = tail + buf
            j = data.find(sig)
            while j >= 0:
                pos = base - len(tail) + j
                if not starts or pos - starts[-1] >= min_gap:
                    starts.append(pos)
                j = data.find(sig, j + 1)
            tail = data[-(len(sig) - 1):]
            base += len(buf)
    return starts


def _h5get(h5, path, default=None):
    try:
        v = h5[path][()]
    except Exception:
        return default
    if isinstance(v, bytes):
        v = v.decode()
    return v


def read_nexus_header(h5) -> dict:
    det = "/entry/instrument/detector"
    h = {
        "qx": _h5get(h5, f"{det}/x_pixel_size"),
        "qy": _h5get(h5, f"{det}/y_pixel_size"),
        "orgx": _h5get(h5, f"{det}/beam_center_x"),
        "orgy": _h5get(h5, f"{det}/beam_center_y"),
        "distance": _h5get(h5, f"{det}/detector_distance"),
        "sensor_thickness": _h5get(h5, f"{det}/sensor_thickness"),
        "sensor_material": _h5get(h5, f"{det}/sensor_material"),
        "count_cutoff": _h5get(h5, f"{det}/detectorSpecific/countrate_correction_count_cutoff"),
        "nx": _h5get(h5, f"{det}/detectorSpecific/x_pixels_in_detector"),
        "ny": _h5get(h5, f"{det}/detectorSpecific/y_pixels_in_detector"),
        "description": _h5get(h5, f"{det}/description"),
        "wavelength": _h5get(h5, "/entry/instrument/beam/incident_wavelength"),
    }
    if h["wavelength"] is None:
        h["wavelength"] = _h5get(h5, "/entry/instrument/monochromator/wavelength")
    tth = _h5get(h5, f"{det}/goniometer/two_theta")
    h["two_theta"] = float(np.atleast_1d(tth)[0]) if tth is not None else None
    for k in ("qx", "qy", "distance", "sensor_thickness"):
        if h[k] is not None:
            h[k] = float(h[k]) * 1000.0  # m -> mm
    pm = _h5get(h5, f"{det}/detectorSpecific/pixel_mask")
    h["pixel_mask"] = np.asarray(pm) if pm is not None else None
    for ax in ("omega", "phi", "chi", "kappa"):
        v = _h5get(h5, f"/entry/sample/goniometer/{ax}")
        inc = _h5get(h5, f"/entry/sample/goniometer/{ax}_range_average")
        if inc is None:
            inc = _h5get(h5, f"/entry/sample/goniometer/{ax}_increment_set")
        if v is not None:
            v = np.atleast_1d(np.asarray(v, float))
            h[ax] = v
            if inc is not None:
                h[f"{ax}_increment"] = float(np.asarray(inc).ravel()[0])
            elif v.size > 1:
                h[f"{ax}_increment"] = float(np.median(np.diff(v)))
    return h


def open_images(template: str, workdir=".", first=None, last=None, min_valid=0.0, overload=None,
                offset=0.0, dark_current_image=None) -> ImageSet:
    iset = _open_images(template, workdir, first, last, min_valid, overload)
    iset.offset = float(offset or 0.0)
    if dark_current_image:
        dpath = resolve_template(dark_current_image, workdir)
        if dpath.endswith((".h5", ".nxs")):
            dk = _open_images(dpath, workdir)
            dark = np.mean([dk.read_raw(f).astype(np.float32) for f in range(dk.first, min(dk.last, dk.first + 9) + 1)], 0)
        else:
            dark, _ = read_cbf(dpath)
            dark = dark.astype(np.float32)
        if dark.shape != (iset.ny, iset.nx):
            raise ValueError(f"DARK_CURRENT_IMAGE {dpath}: shape {dark.shape} != detector {(iset.ny, iset.nx)}")
        iset.dark = dark
    return iset


def _open_images(template: str, workdir=".", first=None, last=None, min_valid=0.0, overload=None) -> ImageSet:
    t = resolve_template(template, workdir)
    if t.endswith(".h5") or t.endswith(".nxs"):
        master = t
        if "?" in t or not t.endswith("_master.h5"):
            base = re.sub(r"_?\?+\.h5$", "", t)
            base = re.sub(r"_(data_\d+|master)$", "", base)
            cand = base + "_master.h5"
            if os.path.exists(cand):
                master = cand
            elif not os.path.exists(master):
                hits = glob.glob(base + "*master*.h5")
                if not hits:
                    raise FileNotFoundError(f"no master file for template {template}")
                master = hits[0]
        return EigerImageSet(master, first, last, min_valid=min_valid, overload=overload)
    first_file = next(iter(sorted(glob.glob(t.replace("?", "[0-9]")))), None)
    if first_file is not None and is_smv(first_file):
        return SMVImageSet(t, first, last, min_valid=min_valid, overload=overload)
    return CBFImageSet(t, first, last, min_valid=min_valid, overload=overload)

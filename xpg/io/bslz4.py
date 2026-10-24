"""Native bitshuffle-LZ4 (HDF5 filter 32008) chunk decoder.

Lets Eiger/NeXus frames be read with H5Dread_chunk (raw compressed bytes) and decoded in
parallel threads: lz4 releases the GIL and the bit-unshuffle is vectorised numpy. This avoids
the global HDF5 lock of h5py filter pipelines (and HDF5 reader plugins).

Chunk layout (bitshuffle 'bslz4'): 8-byte big-endian uncompressed size, 4-byte big-endian
block size in bytes, then per block a 4-byte big-endian compressed size + LZ4 block. Each
decompressed block holds ``block_size // elem_size`` elements bit-transposed in groups of 8;
trailing elements (n % 8) of the last block are stored unshuffled.
"""
from __future__ import annotations

import numpy as np

try:
    import lz4.block as _lz4
except Exception:  # pragma: no cover
    _lz4 = None

BSLZ4_FILTER_ID = 32008


def available():
    return _lz4 is not None


def _bitunshuffle(buf: np.ndarray, n: int, esize: int) -> np.ndarray:
    """Undo bitshuffle for n elements (n multiple of 8) of esize bytes."""
    bits = np.unpackbits(buf.reshape(8 * esize, n // 8), axis=1, bitorder="little")  # (8e, n)
    return np.packbits(bits.T, axis=1, bitorder="little").reshape(-1)              # (n, e) bytes


def lz4_stage(raw, out: np.ndarray, esize: int) -> int:
    """LZ4 stage only: the chunk's blocks decompressed into out (uint8, >= the chunk's size), left bit-shuffled; the
    device undoes the shuffle (unshuffle_frame_torch), which is most of the decoding cost on CPU. Returns the block
    size in elements."""
    mv = memoryview(raw)
    total = int.from_bytes(mv[0:8], "big")
    block = int.from_bytes(mv[8:12], "big") or 8192
    if total % esize or total > out.size:
        raise ValueError("not a bitshuffle-LZ4 chunk of this frame size")
    nelem_total = total // esize
    belem = block // esize
    belem -= belem % 8
    pos = 12
    done = 0
    while done < nelem_total:
        n = min(belem, nelem_total - done)
        n8 = n - n % 8
        if n8 > 0:
            csize = int.from_bytes(mv[pos:pos + 4], "big")
            pos += 4
            out[done * esize:(done + n8) * esize] = np.frombuffer(
                _lz4.decompress(mv[pos:pos + csize], uncompressed_size=n8 * esize), np.uint8)
            pos += csize
        if n8 < n:
            rest = (n - n8) * esize
            out[(done + n8) * esize:(done + n) * esize] = np.frombuffer(mv[pos:pos + rest], np.uint8)
            pos += rest
        done += n
    return belem


# 8 x 8 bit-matrix transpose of a 64-bit word (bit 8 r + c <-> 8 c + r) by three delta swaps; arithmetic right shifts
# of int64 only fill bits the masks clear
_T8 = ((7, 0x00AA00AA00AA00AA), (14, 0x0000CCCC0000CCCC), (28, 0x00000000F0F0F0F0))


def _unshuffle_blocks_torch(x, nb, e, m):
    """x: uint8, nb blocks of 8 e bit planes of m bytes (8 m elements of e bytes) -> the elements' bytes."""
    import torch
    y = x.view(nb, e, 8, m).permute(0, 1, 3, 2).contiguous().view(torch.int64)       # (nb, e, m) 8 x 8 bit matrices
    for s, msk in _T8:
        t = (y ^ (y >> s)) & msk
        y = y ^ t ^ (t << s)
    return y.view(torch.uint8).view(nb, e, m, 8).permute(0, 2, 3, 1).reshape(-1)


def unshuffle_frame_torch(b, nel, esize, block, overload=None, min_valid=0.0, static_bad=None):
    """float32 frame (nel pixels) from its bit-shuffled bytes b (uint8 device tensor, nel * esize, from lz4_stage),
    identical to ImageSet.get for unsigned counts: 2^n - 1 (saturated / masked), counts below min_valid and pixels
    outside the static mask (static_bad True) -1, counts >= overload -2."""
    import math
    import torch
    u = torch.empty(nel * esize, dtype=torch.uint8, device=b.device)
    nf = nel // block
    n0 = nf * block * esize
    if nf:
        u[:n0] = _unshuffle_blocks_torch(b[:n0], nf, esize, block // 8)
    rem = nel - nf * block
    r8 = rem - rem % 8
    if r8:
        u[n0:n0 + r8 * esize] = _unshuffle_blocks_torch(b[n0:n0 + r8 * esize], 1, esize, r8 // 8)
    if rem % 8:
        u[n0 + r8 * esize:] = b[n0 + r8 * esize:nel * esize]
    s = u.view(torch.int32 if esize == 4 else torch.int16)
    bad = s == -1
    v = s.to(torch.int64) & ((1 << (8 * esize)) - 1)
    img = v.to(torch.float32)
    # integer counts: v >= o <=> v >= ceil(o) (no float rounding of large counts in the comparison)
    if min_valid > 0:
        bad |= v < math.ceil(min_valid)
    if static_bad is not None:
        bad |= static_bad
    if overload is not None:
        img.masked_fill_(v >= math.ceil(overload), -2.0)
    return img.masked_fill_(bad, -1.0)

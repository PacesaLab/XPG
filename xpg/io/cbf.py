"""Minimal, dependency-free CBF (byte-offset compressed, miniCBF/PILATUS) reader & writer."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_MARK = b"\x0c\x1a\x04\xd5"


def decode_byte_offset(buf: bytes | np.ndarray, n: int) -> np.ndarray:
    b = np.frombuffer(buf, np.uint8) if not isinstance(buf, np.ndarray) else buf
    i8 = b.view(np.int8)
    out = np.empty(n, np.int64)
    esc = np.flatnonzero(b == 0x80)
    pos = 0
    i = 0
    nb = b.size
    while i < n:
        k = np.searchsorted(esc, pos, side="left")
        j = esc[k] if k < esc.size else nb
        cnt = min(j - pos, n - i)
        if cnt > 0:
            out[i:i + cnt] = i8[pos:pos + cnt]
            i += cnt
            pos += cnt
            if i >= n:
                break
        # escape at pos
        v = int(b[pos + 1]) | (int(b[pos + 2]) << 8)
        if v != 0x8000:
            out[i] = v - 0x10000 if v >= 0x8000 else v
            pos += 3
        else:
            v = int.from_bytes(bytes(b[pos + 3:pos + 7]), "little", signed=True)
            if v != -0x80000000:
                out[i] = v
                pos += 7
            else:
                out[i] = int.from_bytes(bytes(b[pos + 7:pos + 15]), "little", signed=True)
                pos += 15
        i += 1
    return np.cumsum(out)


def encode_byte_offset(data: np.ndarray) -> bytes:
    d = np.diff(data.ravel().astype(np.int64), prepend=0)
    ad = np.abs(d)
    small = ad <= 127
    mid = (~small) & (d >= -32767) & (d <= 32767)
    large = ~(small | mid)
    if np.any(large & ((d < -2 ** 31 + 1) | (d > 2 ** 31 - 1))):
        raise ValueError("values too large for 32-bit byte offset encoding")
    lens = np.where(small, 1, np.where(mid, 3, 7))
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    buf = np.zeros(int(lens.sum()), np.uint8)
    buf[off[small]] = d[small].astype(np.int8).view(np.uint8)
    om = off[mid]
    buf[om] = 0x80
    v16 = d[mid].astype(np.int16).view(np.uint16)
    buf[om + 1] = v16 & 0xFF
    buf[om + 2] = v16 >> 8
    ol = off[large]
    buf[ol] = 0x80
    buf[ol + 1] = 0x00
    buf[ol + 2] = 0x80
    v32 = d[large].astype(np.int32).view(np.uint32)
    for s in range(4):
        buf[ol + 3 + s] = (v32 >> (8 * s)) & 0xFF
    return buf.tobytes()


def read_cbf(path: str | Path, header_only=False):
    """Return (data[int32 (ny, nx)] or None, header dict). Gzip / bzip2-compressed files (.cbf.gz, .cbf.bz2, as
    often archived) are decompressed transparently."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)
    elif raw[:3] == b"BZh":
        import bz2
        raw = bz2.decompress(raw)
    k = raw.find(_MARK)
    if k < 0:
        raise ValueError(f"{path}: not a binary CBF")
    head = raw[:k].decode("latin-1")
    hdr = parse_cbf_header(head)
    if header_only:
        return None, hdr
    n = hdr["n_elements"]
    nx, ny = hdr["nx"], hdr["ny"]
    if "x-cbf_byte_offset" not in head.lower():
        raise ValueError(f"{path}: only x-CBF_BYTE_OFFSET compression is supported")
    size = hdr.get("binary_size")
    buf = raw[k + 4: k + 4 + size] if size else raw[k + 4:]
    data = decode_byte_offset(buf, n).astype(np.int32)
    return data.reshape(ny, nx), hdr


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


def parse_cbf_header(head: str) -> dict:
    h: dict = {}
    m = re.search(r"X-Binary-Number-of-Elements:\s*(\d+)", head)
    h["n_elements"] = int(m.group(1)) if m else None
    m = re.search(r"X-Binary-Size-Fastest-Dimension:\s*(\d+)", head)
    h["nx"] = int(m.group(1)) if m else None
    m = re.search(r"X-Binary-Size-Second-Dimension:\s*(\d+)", head)
    h["ny"] = int(m.group(1)) if m else None
    m = re.search(r"X-Binary-Size:\s*(\d+)", head)
    h["binary_size"] = int(m.group(1)) if m else None
    pats = {
        "pixel_size": r"#\s*Pixel_size\s+([\d.eE+-]+)\s*m\s*x\s*([\d.eE+-]+)\s*m",
        "wavelength": r"#\s*Wavelength\s+([\d.eE+-]+)",
        "distance": r"#\s*Detector_distance\s+([\d.eE+-]+)\s*m",
        "beam_xy": r"#\s*Beam_xy\s*\(\s*([\d.eE+-]+),\s*([\d.eE+-]+)\s*\)",
        "start_angle": r"#\s*Start_angle\s+([\d.eE+-]+)",
        "angle_increment": r"#\s*Angle_increment\s+([\d.eE+-]+)",
        "count_cutoff": r"#\s*Count_cutoff\s+([\d.eE+-]+)",
        "sensor_thickness": r"thickness\s+([\d.eE+-]+)\s*m",
        "detector": r"#\s*Detector:\s*([^,\r\n]+)",
        "exposure_time": r"#\s*Exposure_time\s+([\d.eE+-]+)",
        "phi": r"#\s*Phi\s+([\d.eE+-]+)",
        "omega": r"#\s*Omega\s+([\d.eE+-]+)",
        "kappa": r"#\s*Kappa\s+([\d.eE+-]+)",
        "chi": r"#\s*Chi\s+([\d.eE+-]+)",
        "alpha": r"#\s*Alpha\s+([\d.eE+-]+)",
        "two_theta": r"#\s*Detector_2theta\s+([\d.eE+-]+)",
        "oscillation_axis": r"#\s*Oscillation_axis\s+([^\r\n]+)",
    }
    for k, pat in pats.items():
        m = re.search(pat, head)
        if not m:
            continue
        if k in ("detector", "oscillation_axis"):
            h[k] = m.group(1).strip()
        elif m.lastindex and m.lastindex > 1:
            h[k] = tuple(float(g) for g in m.groups())
        else:
            h[k] = _num(m.group(1))
    return h


def write_cbf(path: str | Path, data: np.ndarray, header: dict | None = None):
    """Write a PILATUS-style miniCBF readable by standard viewers (adxv, albula, fabio, ...)"""
    header = header or {}
    data = np.asarray(data)
    ny, nx = data.shape
    payload = encode_byte_offset(data)
    hl = [
        f"# Detector: {header.get('detector', 'PILATUS 6M, S/N 60-0000')}",
        "# 2026-01-01T00:00:00.000",
        f"# Pixel_size {header.get('qx', 0.172) * 1e-3:.6e} m x {header.get('qy', 0.172) * 1e-3:.6e} m",
        f"# Silicon sensor, thickness {header.get('thickness', 0.45) * 1e-3:.6f} m",
        f"# Exposure_time {header.get('exposure', 0.1):.6f} s",
        f"# Exposure_period {header.get('exposure', 0.1):.6f} s",
        "# Tau = 0 s",
        f"# Count_cutoff {int(header.get('count_cutoff', 1048500))} counts",
        "# Threshold_setting: 6000 eV",
        f"# Wavelength {header.get('wavelength', 1.0):.5f} A",
        f"# Detector_distance {header.get('distance', 200.0) * 1e-3:.5f} m",
        f"# Beam_xy ({header.get('orgx', nx / 2):.2f}, {header.get('orgy', ny / 2):.2f}) pixels",
        f"# Start_angle {header.get('start_angle', 0.0):.4f} deg.",
        f"# Angle_increment {header.get('osc', 0.1):.4f} deg.",
        "# Oscillation_axis X, CW",
        "# N_oscillations 1",
    ]
    head = (
        "###CBF: VERSION 1.5, CBFlib v0.7.8 - PILATUS detectors\r\n\r\n"
        f"data_{Path(path).stem}\r\n\r\n"
        '_array_data.header_convention "PILATUS_1.2"\r\n'
        "_array_data.header_contents\r\n;\r\n" + "\r\n".join(hl) + "\r\n;\r\n\r\n"
        "_array_data.data\r\n;\r\n"
        "--CIF-BINARY-FORMAT-SECTION--\r\n"
        "Content-Type: application/octet-stream;\r\n"
        '     conversions="x-CBF_BYTE_OFFSET"\r\n'
        "Content-Transfer-Encoding: BINARY\r\n"
        f"X-Binary-Size: {len(payload)}\r\n"
        "X-Binary-ID: 1\r\n"
        'X-Binary-Element-Type: "signed 32-bit integer"\r\n'
        "X-Binary-Element-Byte-Order: LITTLE_ENDIAN\r\n"
        f"X-Binary-Number-of-Elements: {nx * ny}\r\n"
        f"X-Binary-Size-Fastest-Dimension: {nx}\r\n"
        f"X-Binary-Size-Second-Dimension: {ny}\r\n"
        "X-Binary-Size-Padding: 4095\r\n\r\n"
    ).encode("latin-1")
    tail = b"\x00" * 4095 + b"\r\n--CIF-BINARY-FORMAT-SECTION----\r\n;\r\n\r\n"
    Path(path).write_bytes(head + _MARK + payload + tail)

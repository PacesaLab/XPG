"""Structured processing events for live monitoring (``xpg serve``).

The pipeline's log lines are mirrored to ``<workdir>/events.jsonl`` (one JSON object per line):
  {"t": unix time, "type": "stage_start" | "stage_end" | "log" | "flag" | "run_start" | "run_end", ...}
Stages are recognised from the '=== NAME ===' headers the pipeline already prints; WARNING lines become flags
(problems for the dashboard to show with their stage). Writing never raises: monitoring must not break processing.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

_STAGE = re.compile(r"^\s*=== (.+?) ===\s*$")
_FLAG = re.compile(r"WARNING|!!! ERROR|not written|failed", re.I)
_NOISE = re.compile(r"RuntimeWarning|UserWarning|FutureWarning|warnings\.warn|^\s*File \"")   # Python-internal


def _clean(v):
    """JSON-safe: non-finite floats -> None (browsers reject NaN / Infinity), numpy scalars -> Python."""
    import math
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except Exception:
            return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class EventLog:
    def __init__(self, workdir):
        self.path = Path(workdir) / "events.jsonl"
        self.stage = None
        self.t_stage = None
        try:
            self.f = open(self.path, "a")
        except OSError:
            self.f = None

    def emit(self, type_, **data):
        if self.f is None:
            return
        data.setdefault("stage", self.stage)
        try:
            self.f.write(json.dumps(_clean({"t": round(time.time(), 3), "type": type_, **data}), default=str,
                                    allow_nan=False) + "\n")
            self.f.flush()
        except Exception:  # pragma: no cover
            pass

    def line(self, text):
        """Mirror one log line; stage headers and warnings become structured events."""
        for ln in str(text).splitlines() or [""]:
            m = _STAGE.match(ln)
            if m:
                self.end_stage()
                self.stage, self.t_stage = m.group(1), time.time()
                self.emit("stage_start", stage=self.stage)
                continue
            if not ln.strip():
                continue
            self.emit("log", stage=self.stage, text=ln)
            if _FLAG.search(ln) and not _NOISE.search(ln):
                self.emit("flag", stage=self.stage, text=ln.strip())

    def end_stage(self):
        if self.stage is not None:
            self.emit("stage_end", stage=self.stage, seconds=round(time.time() - self.t_stage, 1))
        self.stage = None


def read_events(workdir, offset=0):
    """Events from byte ``offset`` on; returns (events, new offset). Partial last lines are left for later."""
    p = Path(workdir) / "events.jsonl"
    if not p.exists():
        return [], offset
    with open(p, "rb") as fh:
        fh.seek(offset)
        data = fh.read()
    cut = data.rfind(b"\n") + 1
    out = []
    for ln in data[:cut].splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out, offset + cut

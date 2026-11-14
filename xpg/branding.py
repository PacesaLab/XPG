"""The XPG logo, shipped with the package (xpg/assets) for the README, the reports and the dashboard."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")
LOGO_SVG = ASSETS / "XPG_logo.svg"
LOGO_PNG = ASSETS / "XPG_logo_small.png"   # 360 px, for embedding in HTML and the PDF


@lru_cache(maxsize=None)
def logo_bytes(kind: str = "png") -> bytes:
    """The logo file's bytes ("png" or "svg"); empty if the asset is missing, so a report never fails over it."""
    p = LOGO_SVG if kind == "svg" else LOGO_PNG
    try:
        return p.read_bytes()
    except OSError:
        return b""


def logo_data_uri(kind: str = "png") -> str:
    """A data: URI of the logo, for self-contained HTML; empty if the asset is missing."""
    b = logo_bytes(kind)
    if not b:
        return ""
    mime = "image/svg+xml" if kind == "svg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"

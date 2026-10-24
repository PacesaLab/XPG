"""One-glance PDF summary of a run (``xpg_report.pdf``), written at the end of processing and again after ``xpg mr``.

Page 1: the data set, the key results, a diffraction image with resolution rings, the statistics table, the possible
problems (warnings, close calls, MR without a solution) highlighted, the decisions, the MR result and the stage times.
Page 2: the analysis graphs (resolution shells, per-batch statistics, spots, spot widths, cell, repeats vs symmetry
mates). Everything comes from the run directory (events.jsonl, xpg_summary.json, xpg_geometry.json, xpg_mr.json); the
image is read from the run's images when they are reachable. ``xpg report RUN_DIR`` rewrites it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .events import read_events

# reference palette (light, print): series, status (always with an icon and a label), ink
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
STATUS = {"critical": ("#d03b3b", "✕"), "warning": ("#fab219", "!"), "close": ("#ec835a", "◆"), "good": ("#0ca30c", "✓")}
INK, INK2, MUTE, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ACC = SERIES[0]


def _load(run_dir):
    d = Path(run_dir)
    j = lambda n: json.loads((d / n).read_text()) if (d / n).exists() else {}
    evs = read_events(d)[0]
    runs = [e for e in evs if e["type"] == "run_start"]
    ends = [e for e in evs if e["type"] == "run_end"]
    decisions, metrics, flags = {}, {}, {}
    for e in evs:
        if e["type"] == "decision":
            decisions[e["name"]] = e
        elif e["type"] == "metric":
            metrics[e["name"]] = e
        elif e["type"] == "flag":
            key = "".join(c for c in e["text"] if not c.isdigit())
            flags[key] = e
    return dict(dir=d, summary=j("xpg_summary.json"), geometry=j("xpg_geometry.json"), mr=j("xpg_mr.json"),
                run=runs[-1] if runs else {}, end=ends[-1] if ends else {}, decisions=decisions, metrics=metrics,
                flags=list(flags.values()))


def _image_from_run(d, run):
    """First frame of the run's images: from its xpg.toml, else the template of its events (None if unreachable)."""
    from .io.images import open_images
    try:
        from .inputs import read_input
        p = read_input(d / "xpg.toml")
        im = open_images(p.name_template, p.workdir, p.data_range[0], p.data_range[1], p.min_valid_pixel, p.overload)
        return im.get(p.data_range[0])
    except Exception:  # noqa: BLE001
        pass
    try:
        f0, f1 = run["frames"]
        return open_images(run["template"], d, f0, f1).get(f0)
    except Exception:  # noqa: BLE001 - the report is drawn without the image
        return None


def _problems(D):
    """(level, title, text) of what to check, most severe first."""
    out = []
    for f in D["flags"]:
        t = f["text"].strip()
        lvl = "critical" if ("ERROR" in t or "failed" in t) else "warning"
        for pre in ("WARNING:", "!!! ERROR !!!"):
            if pre in t:
                t = t.split(pre, 1)[1].strip()
        out.append((lvl, (f.get("stage") or "").split(" ")[0] or "run", t))
    for name, e in D["decisions"].items():
        if e.get("low_margin"):
            alts = "; ".join(e.get("alternatives") or [])
            out.append(("close", f"close call: {name}", f"{e['choice']}" + (f" (alternatives: {alts})" if alts else "")))
    pa = (D["summary"] or {}).get("pathologies") or {}
    tn = pa.get("tncs") or {}
    if tn.get("present"):
        v = tn["vector"]
        out.append(("warning", "tNCS", f"native-Patterson peak {100 * tn['ratio']:.0f}% of the origin at "
                    f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}): weak reflection classes; MR searches tNCS pairs"))
    tw = pa.get("twinning") or {}
    laws = [w for w in tw.get("laws", []) if w["alpha_h"] >= 0.05]
    if laws and tw.get("alpha_l", 0) >= 0.05:
        w = max(laws, key=lambda w: w["alpha_h"])
        out.append(("warning", "twinning", f"twin law {w['law']}: alpha {w['alpha_h']:.2f} (H-test), "
                    f"{w['alpha_britton']:.2f} (Britton); L-test {tw['alpha_l']:.2f}"))
    mr = D["mr"]
    sol = [v for v in mr.values() if isinstance(v, dict) and v.get("solved")]
    if mr and not sol:
        from .mr import SOLVED_Z
        tf = max((v.get("TFZ", 0) for v in mr.values() if isinstance(v, dict) and "TFZ" in v), default=0)
        out.append(("warning", "MR", f"no solution (best TFZ {tf:.1f} < {SOLVED_Z:g})"))
    order = {"critical": 0, "warning": 1, "close": 2}
    return sorted(out, key=lambda x: order[x[0]])


def _style(ax, title=None, xlabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTE, labelsize=6.5, length=2, width=0.5)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=8, color=INK, fontweight="bold", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=6.5, color=MUTE, labelpad=2)


def _resolution_axis(ax, s2):
    """x = 1/d^2 with ticks labelled in Angstrom."""
    span, ds = float(np.nanmax(s2) - np.nanmin(s2)) or 1.0, []
    for d in (10, 6, 4, 3, 2.5, 2, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8):
        if np.nanmin(s2) <= 1 / d ** 2 <= np.nanmax(s2) * 1.02 and (not ds or 1 / d ** 2 - 1 / ds[-1] ** 2 > 0.08 * span):
            ds.append(d)
    ax.set_xticks([1 / d ** 2 for d in ds])
    ax.set_xticklabels([f"{d:g}" for d in ds])


def _image_panel(ax, img, geom, d_cut):
    ax.set_axis_off()
    if img is None:
        ax.text(0.5, 0.5, "diffraction image\nnot reachable", ha="center", va="center", color=MUTE, fontsize=8,
                transform=ax.transAxes)
        return
    ny, nx = img.shape
    b = max(1, int(np.ceil(max(nx, ny) / 700)))
    a = np.asarray(img, np.float32)[:ny // b * b, :nx // b * b].reshape(ny // b, b, nx // b, b)
    thumb = a.max(axis=(1, 3))                       # max-pool: spots stay visible
    masked = thumb < 0
    pos = thumb[~masked & (thumb > 0)]
    bg = float(np.median(pos)) if pos.size else 0.0          # background level of the pooled image
    top = float(np.percentile(pos, 99.7)) - bg if pos.size else 1.0
    show = np.ma.masked_where(masked, np.arcsinh(np.clip(thumb - bg, 0, None) / max(top / 20, 1e-6)))
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("Greys").with_extremes(bad="#dfe8f5")
    ax.imshow(show, cmap=cmap, vmin=0, vmax=float(np.arcsinh(20.0)), interpolation="nearest")
    det = geom.get("detector") or {}
    if not det or not geom.get("wavelength"):
        return
    cx, cy = (det["orgx"] - 1) / b, (det["orgy"] - 1) / b
    lam, D, q = geom["wavelength"], det["distance"], det["pixel_x"]
    rmax = np.hypot(max(cx, nx / b - cx), max(cy, ny / b - cy))
    rings = [d for d in (8, 4, 3, 2.5, 2, 1.6, 1.4, 1.2, 1.0) if not d_cut or abs(d - d_cut) > 0.15]
    for i, d in enumerate(rings + ([d_cut] if d_cut else [])):
        r = D * np.tan(2 * np.arcsin(min(lam / (2 * d), 1.0))) / q / b
        if r > rmax:
            continue
        cut = d == d_cut
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, lw=1.0 if cut else 0.5, ls="-" if cut else (0, (3, 3)),
                                color=SERIES[1] if cut else ACC, alpha=0.95 if cut else 0.7))
        ang = np.radians(-50 if cut else (-30 if i % 2 else -15))
        x_, y_ = cx + r * np.cos(ang), cy + r * np.sin(ang)
        if 0 < x_ < nx / b and 0 < y_ < ny / b:
            ax.text(x_, y_, f"{d:.2f} Å" if cut else f"{d:g} Å", fontsize=6, color=SERIES[1] if cut else ACC,
                    fontweight="bold" if cut else "normal", ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc=(1, 1, 1, 0.75), ec="none"))
    ax.set_xlim(0, nx / b)
    ax.set_ylim(ny / b, 0)


def _tile(fig, x, y, w, h, label, value, note=""):
    ax = fig.add_axes([x, y, w, h])
    ax.set_axis_off()
    ax.add_patch(__import__("matplotlib.patches", fromlist=["FancyBboxPatch"]).FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06", fc=SURF, ec=GRID, lw=0.6, transform=ax.transAxes))
    ax.text(0.08, 0.72, label, fontsize=6.5, color=INK2, transform=ax.transAxes)
    ax.text(0.08, 0.30, value, fontsize=13, color=INK, fontweight="bold", transform=ax.transAxes)
    if note:
        ax.text(0.08, 0.08, note, fontsize=5.5, color=MUTE, transform=ax.transAxes)


def _panel(fig, rect, title=None):
    """Axes for text and tables: axis off, both coordinates 0..1."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title:
        ax.text(0, 1.0, title, fontsize=8.5, fontweight="bold", color=INK, va="bottom")
    return ax


def _f(v, fmt):
    try:
        return format(float(v), fmt) if v is not None and np.isfinite(float(v)) else "–"
    except (TypeError, ValueError):
        return "–"


def _page1(pdf, D, img):
    import matplotlib.pyplot as plt
    S, run = D["summary"], D["run"]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    # header band
    hb = fig.add_axes([0, 0.945, 1, 0.055])
    hb.set_axis_off()
    hb.add_patch(plt.Rectangle((0, 0), 1, 1, color="#184f95", transform=hb.transAxes))
    name = Path(run.get("template", "") or str(D["dir"])).name
    hb.text(0.04, 0.58, "XPG processing summary", color="white", fontsize=14, fontweight="bold", transform=hb.transAxes)
    hb.text(0.04, 0.2, f"{name}   ·   frames {'–'.join(map(str, run.get('frames', [])))}   ·   "
            f"{time.strftime('%Y-%m-%d %H:%M')}", color="#cde2fb", fontsize=7.5, transform=hb.transAxes)
    from . import __version__
    hb.text(0.96, 0.58, f"XPG {__version__}", color="#cde2fb", fontsize=7.5, ha="right", transform=hb.transAxes)
    if D["end"].get("seconds"):
        hb.text(0.96, 0.2, f"processed in {D['end']['seconds'] / 60:.1f} min", color="#cde2fb", fontsize=7.5,
                ha="right", transform=hb.transAxes)
    # key results
    st = (S.get("statistics") or {})
    ov, outer = st.get("overall") or S.get("overall") or {}, st.get("outer") or {}
    pc = lambda v: 100 * v if v is not None and v <= 1.5 else v
    tiles = [("Space group", S.get("space_group") or "–", ""),
             ("Resolution (CC½ = 0.3)", f"{_f(S.get('cc_half_cutoff'), '.2f')} Å",
              ("anisotropic " + " / ".join(f"{v:.2f}" for v in S["aniso"]["limits"]) + " Å")
              if (S.get("aniso") or {}).get("limits") else ""),
             ("ISa", _f(S.get("isa"), ".1f"), ""),
             ("CC½ overall / outer", f"{_f(ov.get('cc_half'), '.3f')} / {_f(outer.get('cc_half'), '.2f')}", "")
             if outer else ("CC½ (all data)", _f(ov.get("cc_half"), ".3f"), ""),
             ("Completeness", f"{_f(pc(ov.get('completeness')), '.1f')} %",
              (f"outer {_f(pc(outer.get('completeness')), '.1f')} %" if outer else "") +
              (f" · ellipsoidal {_f(100 * ea, '.1f')} %" if (ea := (S.get("statistics_aniso") or {})
                                                            .get("ellipsoidal_completeness")) is not None else "")),
             ("Multiplicity", _f(ov.get("multiplicity"), ".1f"), "")]
    for i, (lab, val, note) in enumerate(tiles):
        _tile(fig, 0.05 + (i % 3) * 0.163, 0.855 - (i // 3) * 0.078, 0.15, 0.068, lab, val, note)
    cell = S.get("cell")
    fig.text(0.05, 0.758, ("Cell  " + "  ".join(f"{v:.2f}" for v in cell[:3]) + "  Å    " +
                           "  ".join(f"{v:.1f}" for v in cell[3:]) + "°") if cell else "", fontsize=7.5, color=INK2)
    # diffraction image
    _image_panel(fig.add_axes([0.56, 0.752, 0.39, 0.172]), img, D["geometry"], S.get("cc_half_cutoff"))
    if img is not None:
        fig.text(0.56, 0.742, "first frame; rings: resolution; orange: the CC½ = 0.3 limit", fontsize=5.8, color=MUTE)
    # statistics table
    sa = S.get("statistics_aniso") or {}
    ax = _panel(fig, [0.05, 0.505, 0.46, 0.21], "Statistics")
    cols = [st.get("overall"), st.get("inner"), st.get("outer")] if st else [S.get("overall") or {}]
    cols += [sa.get("overall"), sa.get("outer")] if sa else []
    rows = [("Resolution (Å)", lambda s: f"{_f(s.get('dmax'), '.0f' if (s.get('dmax') or 0) >= 10 else '.1f')}–"
             f"{_f(s.get('dmin'), '.2f')}"),
            ("Unique reflections", lambda s: f"{int(s['n_uniq']):,}" if s.get("n_uniq") else "–"),
            ("Completeness (%)", lambda s: _f(pc(s.get("completeness")), ".1f")),
            ("Multiplicity", lambda s: _f(s.get("multiplicity"), ".1f")),
            ("I/σ", lambda s: _f(s.get("i_over_sigma"), ".1f")), ("Rmeas", lambda s: _f(s.get("rmeas"), ".3f")),
            ("Rpim", lambda s: _f(s.get("rpim"), ".3f")), ("CC½", lambda s: _f(s.get("cc_half"), ".3f")),
            ("CCano", lambda s: _f(s.get("cc_ano"), ".3f"))]
    n_iso = 3 if st else 1
    hdr = ["", "overall", "inner", "outer"][:n_iso + 1] + (["overall", "outer"] if sa else [])
    body = [[lab] + [fn(c or {}) for c in cols] for lab, fn in rows]
    if sa and sa.get("ellipsoidal_completeness") is not None:
        body.append(["Ellipsoidal compl. (%)"] + [""] * n_iso + [_f(100 * sa["ellipsoidal_completeness"], ".1f"), ""])
    xs = [0.0] + [0.42 + 0.12 * j + (0.04 if j >= n_iso else 0) for j in range(len(cols))]
    n = len(body) + 2
    groups = [("isotropic" if st else "all data", 1, n_iso)] + ([("anisotropic", n_iso + 1, n_iso + 2)] if sa else [])
    for lab, c0, c1 in groups:                       # column groups
        ax.text((xs[c0] - 0.08 + xs[c1]) / 2, 1 - 0.55 / n, lab, fontsize=6.3, ha="center", va="center", color=INK2,
                fontweight="bold")
        ax.plot([xs[c0] - 0.09, xs[c1]], [1 - 1.05 / n] * 2, color=BASE, lw=0.6)
    for r, row in enumerate([hdr] + body):
        y = 1 - (r + 1.7) / n
        for c, txt in enumerate(row):
            ax.text(xs[c], y, txt, fontsize=6.6, ha="left" if c == 0 else "right", va="center",
                    color=(MUTE if r == 0 or c == 0 else INK), fontweight="bold" if r == 0 else "normal")
        ax.plot([0, 1], [y - 0.5 / n] * 2, color=GRID, lw=0.4)
    # problems
    probs = _problems(D)
    ax = _panel(fig, [0.56, 0.505, 0.39, 0.21], "Check")
    if not probs:
        probs = [("good", "no problems flagged", "no warnings, no close calls")]
    shown = probs[:5]
    import textwrap
    for i, (lvl, title, text) in enumerate(shown):
        col, icon = STATUS[lvl]
        y = 0.94 - i * 0.19
        ax.text(0.0, y, icon, fontsize=8, color=col, fontweight="bold", va="top")
        ax.text(0.05, y, title, fontsize=6.6, color=INK, fontweight="bold", va="top")
        body = textwrap.wrap(text, 66)
        ax.text(0.05, y - 0.055, "\n".join(body[:2]) + ("…" if len(body) > 2 else ""), fontsize=5.8, color=INK2,
                va="top", linespacing=1.3)
    if len(probs) > len(shown):
        ax.text(0.05, 0.0, f"+ {len(probs) - len(shown)} more in xpg.log", fontsize=5.8, color=MUTE)
    # decisions
    ax = _panel(fig, [0.05, 0.285, 0.9, 0.19], "Decisions")
    dec = sorted(D["decisions"].values(), key=lambda e: not e.get("low_margin"))[:10]
    for i, e in enumerate(dec):
        y = 1 - (i + 0.8) / 10
        ax.text(0.0, y, e["name"], fontsize=6.8, color=INK2)
        m = e.get("margin")
        ax.text(0.27, y, (e["choice"][:80]) + (f"   (margin {m:.3g} {e.get('unit', '')})" if m is not None else ""),
                fontsize=6.8, color=INK)
        if e.get("low_margin"):
            ax.text(0.93, y, "◆ close call", fontsize=6.3, color=STATUS["close"][0], fontweight="bold", ha="right")
        ax.plot([0, 1], [y - 0.3 / 10] * 2, color=GRID, lw=0.4)
    # MR
    mr = D["mr"]
    rows_mr = [(k, v) for k, v in mr.items() if isinstance(v, dict) and "space_group_data" in v]
    ax = _panel(fig, [0.05, 0.13, 0.9, 0.12], "Molecular replacement")
    if not rows_mr:
        ax.text(0.0, 0.8, "not run (xpg mr RUN --model model.pdb, or --sequence content.fasta)", fontsize=6.8, color=MUTE)
    else:
        hdr = ["data", "space group", "copies", "TFZ", "LLG", "R / R-free (placed)", ""]
        xs = [0.0, 0.14, 0.34, 0.46, 0.56, 0.7, 0.86]
        for c, t in enumerate(hdr):
            ax.text(xs[c], 0.82, t, fontsize=6.5, color=MUTE, fontweight="bold")
        ch = (mr.get("choice") or {}).get("data")
        for i, (k, v) in enumerate(rows_mr[:4]):
            y = 0.82 - (i + 1) * 0.17
            vals = [k, v.get("sg") or v.get("space_group_data"), str(v.get("placed", "–")), _f(v.get("TFZ"), ".1f"),
                    _f(v.get("LLG"), ".0f"), f"{_f(v.get('Rwork'), '.3f')} / {_f(v.get('Rfree'), '.3f')}"]
            for c, t in enumerate(vals):
                ax.text(xs[c], y, t, fontsize=6.8, color=INK, fontweight="bold" if k == ch else "normal")
            col, icon = STATUS["good"] if v.get("solved") else STATUS["warning"]
            ax.text(xs[6], y, f"{icon} {'solved' if v.get('solved') else 'not solved'}", fontsize=6.5, color=col,
                    fontweight="bold")
        if mr.get("choice"):
            ax.text(0.0, 0.82 - (min(len(rows_mr), 4) + 1) * 0.17, f"chosen: {mr['choice']['data']} "
                    f"({mr['choice'].get('space_group')}): {mr['choice'].get('reason', '')}"[:150], fontsize=6.2,
                    color=INK2)
    # stage times: one bar, one hue, segments separated by surface gaps
    tm = D["end"].get("timing") or {}
    if tm:
        ax = fig.add_axes([0.05, 0.065, 0.9, 0.035])
        ax.set_axis_off()
        tot, x0 = sum(tm.values()), 0.0
        for k, v in tm.items():
            w = v / tot
            ax.add_patch(plt.Rectangle((x0, 0.25), max(w - 0.003, 0), 0.5, color="#86b6ef", lw=0))
            if w > 0.06:
                ax.text(x0 + w / 2, 0.5, f"{k.title()} {v / 60:.1f}m" if v >= 60 else f"{k.title()} {v:.0f}s",
                        fontsize=5.6, color=INK, ha="center", va="center")
            x0 += w
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0, 0.95, "Stage times", fontsize=6.5, color=INK2, fontweight="bold")
    fig.text(0.05, 0.03, "Outputs: xpg_merged_iso.mtz (merged), xpg_unmerged.mtz (scaled, unmerged), "
             "xpg_summary.html (full statistics), xpg.log", fontsize=6, color=MUTE)
    pdf.savefig(fig)
    plt.close(fig)


def _page2(pdf, D):
    import matplotlib.pyplot as plt
    M, S = D["metrics"], D["summary"]
    fig, axs = plt.subplots(3, 3, figsize=(8.27, 11.69))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.06, hspace=0.45, wspace=0.3)
    fig.text(0.07, 0.955, "Analysis", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.07, 0.938, "per resolution shell (top), along the sweep (middle), geometry and systematic errors "
             "(bottom)", fontsize=7, color=MUTE)
    dcut = S.get("cc_half_cutoff")

    def line(ax, x, y, col=ACC, label=None, gaps=False):
        x, y = np.asarray(x, float), np.asarray([np.nan if v is None else v for v in y], float)
        if gaps and x.size > 3:                  # no line across missing batches (e.g. frames removed by a filter)
            dx = np.diff(x)
            cut = np.flatnonzero(dx > 3 * np.median(dx)) + 1
            x, y = np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)
        ax.plot(x, y, color=col, lw=1.2, label=label)
        return x, y

    def shells(ax, key, title, ref=None):
        m = M.get("shells")
        if not m or key not in m["series"]:
            return _empty(ax, title)
        x, _ = line(ax, m["x"], m["series"][key])
        _style(ax, title, "resolution (Å)")
        _resolution_axis(ax, x)
        if dcut:
            ax.axvline(1 / dcut ** 2, color=SERIES[1], lw=0.8, ls=(0, (3, 2)))
            ax.text(1 / dcut ** 2, ax.get_ylim()[1], f" {dcut:.2f} Å", fontsize=6, color=SERIES[1], va="top")
        if ref is not None:
            ax.axhline(ref, color=MUTE, lw=0.6, ls=":")

    def sweep(ax, name, key, title, col=ACC):
        m = M.get(name)
        if not m or key not in m["series"]:
            return _empty(ax, title)
        line(ax, m["x"], m["series"][key], col, gaps=name == "batch_stats")
        _style(ax, title, m.get("xlabel") or "frame")

    shells(axs[0, 0], "CC1/2", "CC½", 0.3)
    shells(axs[0, 1], "<I/sigma>", "<I/σ>")
    shells(axs[0, 2], "completeness (%)", "Completeness (%)")
    sweep(axs[1, 0], "spots_per_frame", "spots", (M.get("spots_per_frame") or {}).get("title", "Spots per frame"))
    sweep(axs[1, 1], "batch_stats", "<I/sigma>", "<I/σ> per batch")
    sweep(axs[1, 2], "batch_stats", "Rmeas (%)", "Rmeas (%) per batch")
    sweep(axs[2, 0], "block_sigmas", "sigma_M (deg)", "Rocking width σM (°) per block")
    for ax, name, title in ((axs[2, 1], "cell_trajectory", "Cell along the sweep (% from median)"),
                            (axs[2, 2], "shared_error", "Repeats vs symmetry mates (CC)")):
        m = M.get(name)
        if not m:
            _empty(ax, title)
            continue
        for i, (k, v) in enumerate(list(m["series"].items())[:3]):
            line(ax, m["x"], v, SERIES[i], k)
        _style(ax, title, "resolution (Å)" if name == "shared_error" else (m.get("xlabel") or "frame"))
        if name == "shared_error":
            _resolution_axis(ax, np.asarray(m["x"], float))
        ax.legend(fontsize=6, frameon=False, loc="best", handlelength=1.2)
    pdf.savefig(fig)
    plt.close(fig)


def _empty(ax, title):
    _style(ax, title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=7, color=MUTE, transform=ax.transAxes)


def write_pdf_report(run_dir, image=None, log=print):
    """run_dir/xpg_report.pdf; image: a frame (else read from the run's images when reachable). Never raises."""
    run_dir = Path(run_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        log("xpg: PDF report not written (matplotlib not installed)")
        return None
    try:
        import matplotlib.pyplot as plt
        D = _load(run_dir)
        img = image if image is not None else _image_from_run(run_dir, D["run"])
        with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
            with PdfPages(run_dir / "xpg_report.pdf", metadata={"Title": "XPG processing summary",
                                                                "Creator": "XPG"}) as pdf:
                _page1(pdf, D, img)
                _page2(pdf, D)
        log(f"xpg: PDF summary in {run_dir / 'xpg_report.pdf'}")
        return run_dir / "xpg_report.pdf"
    except Exception as e:  # noqa: BLE001 - a report must never fail the run
        log(f"xpg: PDF report not written ({type(e).__name__}: {e})")
        return None

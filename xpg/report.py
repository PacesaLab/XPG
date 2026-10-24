"""Run summary: xpg_summary.md and xpg_summary.html from the pipeline results.

What was processed, the decisions taken (space group, lattices, excluded frames and ice rings, resolution
limits), data quality (ISa, merging statistics per shell, anisotropy, twinning, tNCS) and the output files.
"""
from __future__ import annotations

import html
import time

import numpy as np

OUTPUTS = [("xpg_merged_iso.mtz", "merged, French-Wilson amplitudes, isotropic CC1/2 cut-off"),
           ("xpg_anisotropic.mtz", "merged, anisotropic cut-off and anisotropic-prior French-Wilson"),
           ("xpg_unmerged.mtz", "unmerged scaled intensities"),
           ("xpg_geometry.json", "refined experimental geometry and crystal model"),
           ("xpg_deposit.cif", "PDB deposition: merged + unmerged reflections with data-collection statistics"),
           ("xpg_all_lattices_unmerged.mtz", "unmerged, all lattices scaled together"),
           ("xpg_all_lattices_merged_iso.mtz", "all lattices, merged"),
           ("xpg_lattice1_merged_iso.mtz", "first lattice alone, merged"),
           ("xpg_detwinned_merged_iso.mtz", "detwinned merged data"),
           ("xpg_integrated.npz", "integrated reflections (pass 1; xpg_integrated_pass2.npz after pass 2)"),
           ("xpg_spots.txt", "strong spots"),
           ("xpg_summary.json", "machine-readable summary"),
           ("xpg.log", "full log"),
           ("XDS_ASCII.HKL", "legacy-format unmerged data (--export-legacy)")]


def _ranges(frames):
    """[1, 2, 3, 7, 8] -> '1-3, 7-8'."""
    frames = sorted(set(int(f) for f in frames))
    if not frames:
        return ""
    out, a = [], frames[0]
    for p, q in zip(frames, frames[1:] + [None]):
        if q != p + 1:
            out.append(f"{a}-{p}" if p > a else f"{a}")
            a = q
    return ", ".join(out)


def _f(x, fmt):
    try:
        return format(float(x), fmt) if x is not None and np.isfinite(float(x)) else "-"
    except (TypeError, ValueError):
        return "-"


def build_markdown(pipe) -> str:
    p, st, r = pipe.p, pipe.st, pipe.st.results
    g = st.exp.geom
    L = ["# XPG summary", "",
         f"Processed {time.strftime('%Y-%m-%d %H:%M')}; data `{p.name_template}`, frames {p.data_range[0]}-{p.data_range[1]} "
         f"({abs(g.osc):g} deg each), wavelength {g.wavelength:.4f} A, detector {g.nx} x {g.ny} px."]
    if getattr(pipe, "timing", None):
        tot = sum(pipe.timing.values())
        L += ["", f"Run time {tot / 60:.1f} min (" + ", ".join(f"{k} {v:.0f} s" for k, v in pipe.timing.items()) + ")."]
    try:
        from .backend import describe
        L += [f"Compute backend: {describe()}."]
    except Exception:  # pragma: no cover
        pass

    # automatic decisions (low-margin ones first)
    dec = r.get("decisions") or []
    if dec:
        L += ["", "## Automatic decisions", "",
              "| decision | choice | margin | alternatives |", "|---|---|---|---|"]
        for d in sorted(dec, key=lambda d: not d.get("low_margin")):
            m = "" if d.get("margin") is None else f"{d['margin']:.3g} {d.get('unit', '')}".strip()
            flag = " **(close call: check)**" if d.get("low_margin") else ""
            L += [f"| {d['name']} | {d['choice']}{flag} | {m} | {'; '.join(d.get('alternatives') or [])} |"]
        hints = [d for d in dec if d.get("hint")]
        if hints:
            L += [""] + [f"- *{d['name']}*: {d['hint']}" for d in hints]

    # symmetry and lattices
    L += ["", "## Symmetry and lattices", ""]
    sg = st.space_group.xhm() if st.space_group else "?"
    cell = st.exp.crystal.cell
    L += [f"- Space group **{sg}**, cell {' '.join(f'{c:.2f}' for c in cell[:3])} A, "
          f"{' '.join(f'{c:.1f}' for c in cell[3:])} deg"]
    alts = [a for a in (r.get("space_group_alternatives") or []) if a != sg]
    if alts:
        L += [f"- Space groups not distinguished by the data (same point group, screw axes ambiguous): {', '.join(alts)}"]
    hz = r.get("hidden_twin")
    for h in ([hz] if isinstance(hz, dict) else (hz or [])):
        L += [f"- **Twinning warning**: the true space group may be {h['space_group']} with twin law {h['law']} "
              f"(alpha ~{0.5 * (h['alpha_h'] + h['alpha_britton']):.2f}); data in that setting: "
              f"`xpg_unmerged_{h['space_group'].replace(' ', '')}.mtz`"]
    tw = r.get("twinning")
    if tw is not None:
        L += [f"- Wilson B {_f(tw.wilson_b, '.1f')} A^2; L-test <|L|> {_f(tw.l_mean, '.3f')} (0.500 untwinned, "
              f"0.375 perfect twin)"]
        for op, h, a, n, ab in (tw.twin_laws or []):
            L += [f"  - twin law {op}: alpha {a:.3f} (H-test), {ab:.3f} (Britton)"]
    tn = r.get("tncs")
    if tn:
        L += [f"- Native Patterson: largest off-origin peak {100 * tn['ratio']:.1f}% of the origin at "
              f"({', '.join(f'{v:+.3f}' for v in tn['vector'])})" + (" -> **tNCS likely**" if tn["ratio"] > 0.2 else "")]
    li = r.get("lattice_isomorphism")
    if li:
        n = len(li["cc_half"])
        L += [f"- {n} lattices integrated; merging reference: lattice {li['reference']}; primary output: "
              f"{r.get('primary_mtz', 'first lattice')}"]

    # decisions on the data
    L += ["", "## Data selection", ""]
    rg = r.get("removed_image_groups")
    gs = r.get("image_group_size")
    if rg and gs:
        frames = [p.data_range[0] + b * gs + k for b in rg for k in range(gs)]
        L += [f"- Frames excluded as inconsistent (Delta-CC1/2 image-group filter): {_ranges(frames)} "
              f"({len(frames)} frames)"]
    else:
        L += ["- No frames excluded by the image-group filter"]
    if p.exclude_data_range:
        L += ["- Frames excluded by the input (EXCLUDE_DATA_RANGE): " + ", ".join(f"{a}-{b}" for a, b in p.exclude_data_range)]
    rings = r.get("ice_rings_excluded")
    if rings:
        L += ["- Ice-ring shells excluded: " + ", ".join(f"{max(a, b):.2f}-{min(a, b):.2f} A" for a, b in rings)]
    elif r.get("ice_rings"):
        L += ["- Ice rings detected but kept (not worse than ice-free shells)"]

    # quality
    L += ["", "## Data quality", ""]
    L += [f"- ISa {_f(r.get('isa_two_param', r.get('isa')), '.1f')} (two-parameter a, b error model); empirical (strong-reflection "
          f"scatter) {_f(r.get('isa_empirical'), '.1f')}"]
    L += [f"- Resolution where CC1/2 = 0.3: {_f(r.get('cc_half_cutoff'), '.2f')} A"]
    an = r.get("aniso")
    if an and an.get("limits"):
        dirs = an.get("directions_hkl") or []
        parts = []
        for lim, d in zip(an["limits"], dirs):
            dd = np.asarray(d, float)
            dd = dd / np.max(np.abs(dd)) if np.max(np.abs(dd)) > 0 else dd
            parts.append(f"{lim:.2f} A along {' '.join(f'{x:+.2f}' for x in dd)}")
        L += ["- Anisotropic diffraction limits (local <I/sigma> = 1.2): " + "; ".join(parts)]
        try:
            from .aniso import ellipsoidal_completeness
            mg = r.get("merged")
            ec = ellipsoidal_completeness(mg["hkl"], st.exp.crystal.cell, st.space_group, an["limits"], dirs) \
                if mg is not None and st.space_group is not None else None
            if ec is not None:
                L += [f"- Ellipsoidal completeness (inside the anisotropic cut-off surface): {100 * ec[2]:.1f}% "
                      f"({ec[0]} of {ec[1]} unique reflections)"]
        except Exception:  # pragma: no cover (report only)
            pass
    L += [f"- Spot model: sigma_D {_f(st.model.sigma_d, '.4f')} deg, sigma_M {_f(st.model.sigma_m, '.4f')} deg"]
    shells, ov = r.get("shells"), r.get("overall")
    if shells and ov is not None:
        # summary: overall, inner (lowest-resolution) and outer (highest-resolution) shells,
        # inside the CC1/2 = 0.3 limit when SCALE computed those statistics
        cut = r.get("shells_cut") is not None and r.get("overall_cut") is not None
        shells_s, ov_s = (r["shells_cut"], r["overall_cut"]) if cut else (shells, ov)
        inner, outer = shells_s[0], shells_s[-1]
        rows = [("Resolution (A)", lambda s: f"{s.dmax:.2f}-{s.dmin:.2f}"), ("N obs", lambda s: f"{int(s.n_obs)}"),
                ("N unique", lambda s: f"{int(s.n_uniq)}"),
                ("Completeness %", lambda s: _f(100 * s.completeness if s.completeness <= 1.0 else s.completeness, '.1f')),
                ("Multiplicity", lambda s: _f(s.multiplicity, '.1f')), ("I/sigma", lambda s: _f(s.i_over_sigma, '.1f')),
                ("Rmeas", lambda s: _f(s.rmeas, '.3f')), ("Rpim", lambda s: _f(s.rpim, '.3f')),
                ("CC1/2", lambda s: _f(s.cc_half, '.3f')), ("CCano", lambda s: _f(s.cc_ano, '.3f'))]
        # the anisotropic data (inside the cut-off surface, xpg_anisotropic.mtz) next to the isotropic ones
        an = r.get("shells_aniso") and r.get("overall_aniso") is not None
        cols = [ov_s, inner, outer] + ([r["overall_aniso"], r["shells_aniso"][0], r["shells_aniso"][-1]] if an else [])
        head = "| | overall | inner shell | outer shell |" + (" aniso overall | aniso inner | aniso outer |" if an else "")
        L += ["", "Summary" + (" (isotropic: to the CC1/2 = 0.3 limit" if cut else " (isotropic: all data") +
              ("; anisotropic: inside the cut-off surface, xpg_anisotropic.mtz)" if an else ")") + ":", "",
              head, "|---" * (len(cols) + 1) + "|"]
        for name, fn in rows:
            L += ["| " + name + " | " + " | ".join(fn(c) for c in cols) + " |"]
        if an and r.get("ellipsoidal_completeness") is not None:
            L += ["| Ellipsoidal completeness % | | | | " + _f(100 * r["ellipsoidal_completeness"], '.1f') + " | | |"]
        L += ["", "All shells (to the detector edge):"]
        L += ["", "| Resolution (A) | N obs | N uniq | Compl % | Mult | I/sigma | Rmeas | Rpim | CC1/2 | CCano |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in list(shells) + [ov]:
            lab = f"{s.dmax:.2f}-{s.dmin:.2f}" if s is not ov else f"**overall {s.dmax:.2f}-{s.dmin:.2f}**"
            L += [f"| {lab} | {int(s.n_obs)} | {int(s.n_uniq)} | {_f(100 * s.completeness if s.completeness <= 1.0 else s.completeness, '.1f')} | {_f(s.multiplicity, '.1f')} "
                  f"| {_f(s.i_over_sigma, '.1f')} | {_f(s.rmeas, '.3f')} | {_f(s.rpim, '.3f')} | {_f(s.cc_half, '.3f')} "
                  f"| {_f(s.cc_ano, '.3f')} |"]

    # files
    L += ["", "## Output files", ""]
    for fn, what in OUTPUTS:
        if (pipe.wd / fn).exists():
            L += [f"- `{fn}`: {what}"]
    return "\n".join(L) + "\n"


def _md_to_html(md: str) -> str:
    """The subset of Markdown written above (headings, bullets, tables, bold, code) as a small styled page."""
    import re

    def inline(t):
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    out, in_ul, rows = [], 0, []

    def flush_table():
        if rows:
            out.append("<table>")
            for i, rw in enumerate(rows):
                tag = "th" if i == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{inline(c.strip())}</{tag}>" for c in rw) + "</tr>")
            out.append("</table>")
            rows.clear()
    for line in md.splitlines():
        if line.startswith("|"):
            cells = line.strip().strip("|").split("|")
            if not all(set(c.strip()) <= set("-") for c in cells):
                rows.append(cells)
            continue
        flush_table()
        ind = len(line) - len(line.lstrip())
        if line.lstrip().startswith("- "):
            lvl = ind // 2 + 1
            while in_ul < lvl:
                out.append("<ul>"); in_ul += 1
            while in_ul > lvl:
                out.append("</ul>"); in_ul -= 1
            out.append(f"<li>{inline(line.lstrip()[2:])}</li>")
            continue
        while in_ul:
            out.append("</ul>"); in_ul -= 1
        if line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush_table()
    while in_ul:
        out.append("</ul>"); in_ul -= 1
    css = ("body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:60rem;margin:2rem auto;"
           "padding:0 1rem;color:#1d1d1f;background:#fff;line-height:1.45}h1{font-size:1.5rem}h2{font-size:1.15rem;"
           "margin-top:1.6rem;border-bottom:1px solid #ddd}table{border-collapse:collapse;font-size:.85rem;"
           "font-variant-numeric:tabular-nums}td,th{border-bottom:1px solid #e5e5e5;padding:.25rem .6rem;text-align:right}"
           "td:first-child,th:first-child{text-align:left}tr:last-child td{font-weight:600}code{font-size:.85em}"
           "@media(prefers-color-scheme:dark){body{background:#161618;color:#e8e8ea}td,th{border-color:#333}"
           "h2{border-color:#333}}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>XPG summary</title><style>{css}</style></head><body>" + "\n".join(out) + "</body></html>\n")


def write_summary(pipe):
    md = build_markdown(pipe)
    (pipe.wd / "xpg_summary.md").write_text(md)
    (pipe.wd / "xpg_summary.html").write_text(_md_to_html(md))
    return md

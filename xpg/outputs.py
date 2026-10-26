"""Merged outputs: isotropic and anisotropic merged MTZ files with statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io.mtz import add_free_flags, write_merged_mtz

def _isa_txt(isa):
    """Limiting I/sigma of the three-parameter model, 1/(SdFac SdAdd). With SdAdd ~ 0 (the strongest reflections'
    scatter is explained by the linear SdB term) I/sigma grows as sqrt(I) without a plateau: no finite limit to quote;
    the reported ISa is always the two-parameter one (Diederichs 2010)."""
    return f"limit I/sigma {isa:.1f}" if isa < 500 else "SdAdd ~ 0: I/sigma does not level off in the measured range"

def write_merged_outputs(merged, cell, sg, dcut, wavelength, stem: Path, log=print, anisotropy=True,
                         free_r_reference=None):
    """Write <stem>_merged_iso.mtz (French-Wilson, isotropic prior, CC1/2 cut-off) and
    <stem>_anisotropic.mtz (anisotropic cut-off + anisotropic-prior French-Wilson). Returns aniso summary."""
    from .aniso import french_wilson, run_anisotropic
    stem = Path(stem)
    m = dict(merged)
    m["free"] = add_free_flags(m["hkl"], sg, cell=cell)
    if free_r_reference:
        from .analysis import transfer_free_flags
        try:
            m["free"], nk, nn = transfer_free_flags(m["hkl"], sg, free_r_reference)
            log(f"  FREE-R: {nk} flags copied from {free_r_reference}, {nn} new reflections flagged")
        except Exception as e:
            log(f"  FREE-R: transfer failed ({e}); using new random flags")
    iso_prior = _isotropic_prior(m["d"], m["I"], m["eps"])
    F, sF, Ip, sIp = french_wilson(m["I"], m["sig"], iso_prior, m["centric"])
    m["F"], m["SIGF"] = F, sF
    for tag in ("plus", "minus"):
        Fa, sFa, _, _ = french_wilson(m[f"I{tag}"], m[f"sigI{tag}"], iso_prior, m["centric"])
        m[f"F{tag}"], m[f"SIGF{tag}"] = Fa, sFa
    m["DANO"] = m["Fplus"] - m["Fminus"]
    m["SIGDANO"] = np.sqrt(m["SIGFplus"] ** 2 + m["SIGFminus"] ** 2)
    cols = [("IMEAN", "J", "I"), ("SIGIMEAN", "Q", "sig"), ("I(+)", "K", "Iplus"), ("SIGI(+)", "M", "sigIplus"),
            ("I(-)", "K", "Iminus"), ("SIGI(-)", "M", "sigIminus"), ("F", "F", "F"), ("SIGF", "Q", "SIGF"),
            ("F(+)", "G", "Fplus"), ("SIGF(+)", "L", "SIGFplus"), ("F(-)", "G", "Fminus"), ("SIGF(-)", "L", "SIGFminus"),
            ("DANO", "D", "DANO"), ("SIGDANO", "Q", "SIGDANO"), ("FreeR_flag", "I", "free")]
    sel = np.ones(len(m["I"]), bool) if dcut is None else m["d"] >= dcut
    mi = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape else v) for k, v in m.items()}
    write_merged_mtz(f"{stem}_merged_iso.mtz", sg, cell, wavelength, mi, cols, history=[f"XPG isotropic cut-off {dcut}"])
    if not anisotropy:
        return None
    try:
        res, surf, model = run_anisotropic(m, cell, sg, log=log)
        ma = dict(m)
        for k in ("F", "SIGF", "Fplus", "SIGFplus", "Fminus", "SIGFminus", "DANO", "SIGDANO",
                  "F_aniso", "SIGF_aniso", "Ipost", "SIGIpost"):
            ma[k] = res[k]
        ins = res["inside"]
        ma = {k: (v[ins] if isinstance(v, np.ndarray) and v.shape[:1] == ins.shape else v) for k, v in ma.items()}
        cols_a = cols + [("FANISO", "F", "F_aniso"), ("SIGFANISO", "Q", "SIGF_aniso"),
                         ("IPOST", "J", "Ipost"), ("SIGIPOST", "Q", "SIGIpost")]
        write_merged_mtz(f"{stem}_anisotropic.mtz", sg, cell, wavelength, ma, cols_a,
                         history=["XPG anisotropic cut-off (<I/sigI> >= 1.2 along each direction)",
                                  f"principal limits {surf.limits.round(2).tolist()}"])
        return dict(limits=surf.limits.tolist(), directions_hkl=surf.directions_hkl.tolist(),
                    B_eigen=model.eigen_B.tolist(), n_inside=int(ins.sum()), inside_hkl=ma["hkl"])
    except Exception as e:
        log(f"  ANISO: failed ({e})")
        return None

def _isotropic_prior(d, I, eps, nbins=30):
    s2 = 1.0 / d ** 2
    edges = np.quantile(s2, np.linspace(0, 1, nbins + 1))
    b = np.clip(np.searchsorted(edges, s2) - 1, 0, nbins - 1)
    m = np.array([np.mean((I / eps)[b == i]) if np.any(b == i) else np.nan for i in range(nbins)])
    m = np.where(np.isfinite(m) & (m > 0), m, np.nanmin(m[m > 0]) if np.any(m > 0) else 1.0)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return eps * np.exp(np.interp(s2, centers, np.log(m)))

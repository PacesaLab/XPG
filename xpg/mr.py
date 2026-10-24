"""Molecular replacement on a finished XPG run, with XPG's own search on the GPU.

    xpg mr RUN_DIR [--sequence content.fasta] [--model model.pdb] [--copies N] [--alternatives "P 31 1 2,P 31"]

* data: the run's merged MTZ (xpg_merged_iso.mtz: F, SIGF, FreeR_flag), resolution as decided by the run
* content: the FASTA file lists one copy of the asymmetric-unit content (protein, RNA, DNA chains, ligands; see
  matthews.read_components); with --model alone it is read from the model's polymer chains
* copies: most probable Matthews copy number (Kantardjieff & Rupp 2003) and its neighbours
* data pathologies, found by the processing analyses run on each data set searched (data_pathologies): tNCS
  (native-Patterson peak > 20 % of the origin) -> copies searched as tNCS pairs (single copies as well when the
  pairs give no clear solution); twinning (a twin law with its H-test fraction, when the L-test, which tNCS does not
  bias, also shows twinning) -> twinned LLG, and the twin fraction refined against the placed model; anisotropy
  and Wilson outliers are handled in the targets (xpg.llg, xpg.mrsearch)
* check: R / R-free of the placed model (bulk solvent, overall scale and B; no atomic refinement); after a close
  point-group call, the solutions are compared on the same data (choose_point_group)

Results: RUN_DIR/mr/<label>/..., RUN_DIR/xpg_mr.json, and a summary in RUN_DIR/xpg_mr.log."""
from __future__ import annotations

import json
from pathlib import Path

import gemmi
import numpy as np


def prepare_model(model, out):
    """Search model: polymer chains only (waters, ligands and hydrogens removed). Predicted models (B column =
    pLDDT, no CRYST1): residues with pLDDT < 70 dropped, B from the expected coordinate error
    (rmsd = 1.5 exp(4 (0.7 - pLDDT/100)), Oeffner et al. 2022); experimental models keep their B factors.
    Returns (predicted, n_residues, n_chains)."""
    st = gemmi.read_structure(str(model))
    st.setup_entities()
    st.remove_ligands_and_waters()             # the search model: polymer chains only
    st.remove_hydrogens()
    st.remove_empty_chains()
    bs = np.array([at.b_iso for ch in st[0] for res in ch for at in res])
    predicted = bool((not st.cell.is_crystal()) and bs.size and bs.max() <= 100.0 and np.median(bs) > 50.0)
    if predicted:
        for ch in st[0]:
            for i in range(len(ch) - 1, -1, -1):
                pl = np.mean([at.b_iso for at in ch[i]])
                if pl < 70.0:
                    del ch[i]
                    continue
                rmsd = 1.5 * np.exp(4.0 * (0.7 - pl / 100.0))
                for at in ch[i]:
                    at.b_iso = float(8.0 * np.pi ** 2 / 3.0 * rmsd ** 2)
        st.remove_empty_chains()
    st.write_pdb(str(out))
    return predicted, sum(len(ch) for ch in st[0]), len(st[0])


def merge_in(run_dir, sg_alt, out_stem, log=print, reindex=None, cell=None, dmin=None):
    """Merge the run's unmerged data in another space group -> <out_stem>_merged_iso.mtz. Same cell basis unless
    reindex (integer M, h' = M h) and the new conventional cell are given; cut at the run's CC1/2 limit unless dmin
    is given."""
    from .io.xpgfiles import read_unmerged_mtz
    from .merging import merging_statistics
    from .outputs import write_merged_outputs
    from .symops import to_asu
    run_dir = Path(run_dir)
    u = read_unmerged_mtz(run_dir / "xpg_unmerged.mtz")
    sg = gemmi.SpaceGroup(sg_alt)
    hkl = u["hkl"] if reindex is None else u["hkl"] @ np.asarray(reindex).T
    cell = np.array(u["cell"] if cell is None else cell, float)
    system = sg.crystal_system_str()
    if system in ("trigonal", "hexagonal"):
        a = 0.5 * (cell[0] + cell[1])
        cell = np.array([a, a, cell[2], 90.0, 90.0, 120.0])
    elif system == "tetragonal":
        a = 0.5 * (cell[0] + cell[1])
        cell = np.array([a, a, cell[2], 90.0, 90.0, 90.0])
    elif system == "cubic":
        a = float(np.mean(cell[:3]))
        cell = np.array([a, a, a, 90.0, 90.0, 90.0])
    elif system == "orthorhombic":
        cell = np.r_[cell[:3], 90.0, 90.0, 90.0]
    elif system == "monoclinic":
        cell = np.r_[cell[:3], 90.0, cell[4], 90.0]
    ok = np.isfinite(u["I"]) & (u["sig"] > 0)
    hasu, _, plus = to_asu(hkl[ok], sg)
    _, _, merged = merging_statistics(hasu, plus, u["I"][ok], u["sig"][ok], cell, sg, nbins=10, anomalous=True)
    summ = json.loads((run_dir / "xpg_summary.json").read_text()) if (run_dir / "xpg_summary.json").exists() else {}
    dcut = dmin if dmin is not None else summ.get("cc_half_cutoff")
    wl = float((u.get("meta") or {}).get("wavelength", 1.0))
    write_merged_outputs(merged, cell, sg, dcut, wl, Path(out_stem), log=log, anisotropy=False)
    return Path(f"{out_stem}_merged_iso.mtz")


def _same_basis_alternatives(sg_run, alt_labels):
    """Point-group alternatives (from the run's decision record, e.g. 'hP 3 (0.005)') that share the cell basis of
    the run's lattice -> reference space groups to merge in."""
    from .lattice import _SG_FOR, _rotations
    out = []
    cen = sg_run.hm[0]
    for a in alt_labels:
        parts = a.split()
        if len(parts) < 2:
            continue
        bravais, pg = parts[0], parts[1]
        if bravais[0] != {"hexagonal": "h", "trigonal": "h", "tetragonal": "t", "cubic": "c"}.get(
                sg_run.crystal_system_str(), "?") or bravais[1:] != cen:
            continue
        hm = _SG_FOR.get((pg, cen))
        if hm and _rotations(hm) != _rotations(sg_run.hm):     # point-group name alone: P321 and P312 are both '32'
            out.append(hm)
    return out


# TFZ: z-score of the last copy placed against random positions of its orientation (fast LLG, xpg.mrsearch); solved
# from 8 (correct solutions: 10-34, with tNCS 10; wrong search models: up to 6)
SOLVED_Z = 8.0


def data_pathologies(mtz_path, log=print, label=""):
    """tNCS and twinning of a merged data set, with the analyses of the processing (xpg.analysis) run on the data MR
    uses: tNCS when the largest off-origin native-Patterson peak is above 20 % of the origin; a twin law with its
    H-test fraction when the L-test (not biased by tNCS) also shows twinning (alpha_L >= 0.05)."""
    from .analysis import l_test_alpha, tncs_patterson, twinning_analysis
    m = gemmi.read_mtz_file(str(mtz_path))
    lab = m.column_labels()
    col = lambda k: np.asarray(m.column_with_label(k), float)
    hkl = np.asarray(m.make_miller_array(), np.int64)
    if "IMEAN" in lab and "SIGIMEAN" in lab:
        I, sig = col("IMEAN"), col("SIGIMEAN")
    else:
        F, sF = col("F"), col("SIGF")
        I, sig = F ** 2, 2 * F * sF
    ok = np.isfinite(I) & np.isfinite(sig) & (sig > 0)
    hkl, I, sig = hkl[ok], I[ok], sig[ok]
    cell, sg = tuple(m.cell.parameters), m.spacegroup
    out = dict(tncs=None, twin=None)
    ratio, vec, dist = tncs_patterson(hkl, I, cell, sg)
    if np.isfinite(ratio) and ratio > 0.2:
        out["tncs"] = dict(vector=[float(v) for v in vec], ratio=float(ratio), distance=float(dist))
        log(f"MR [{label}]: tNCS: native-Patterson peak {100 * ratio:.0f}% of the origin at "
            f"({vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f}), {dist:.1f} A: copies searched as tNCS pairs")
    try:
        tw = twinning_analysis(hkl, I, sig, cell, sg, tncs=(vec, min(ratio, 0.95)) if out["tncs"] else None)
    except Exception as e:  # noqa: BLE001
        log(f"MR [{label}]: twinning analysis failed ({e})")
        return out
    a_l = l_test_alpha(tw.l_mean)
    out["l_test"] = dict(l_mean=float(tw.l_mean), alpha=a_l)
    if tw.twin_laws:
        op, _, a_h, _, a_b = max(tw.twin_laws, key=lambda t: t[2])
        out["twin_test"] = dict(law=op, alpha_h=float(a_h), alpha_britton=float(a_b))
        use = a_l >= 0.05 and a_h >= 0.05
        if use:
            out["twin"] = dict(law=op, alpha=float(min(a_h, 0.45)))
        log(f"MR [{label}]: twin law {op}: alpha {a_h:.2f} (H-test), {a_b:.2f} (Britton); L-test alpha {a_l:.2f} -> "
            + ("twinned LLG" if use else "not used (the L-test shows no twinning)"))
    return out


def _xpg_search(wd, data_mtz, model_pdb, n_copies, tag, log=print, patho=None):
    """Own MR search (xpg.mrsearch): space-group scan of the point group, sequential copies (tNCS pairs), sigma_A
    LLG (twinned with a twin law)."""
    from .mrsearch import search_space_groups
    patho = patho or {}
    tn, tw = patho.get("tncs"), patho.get("twin")
    tvec = tn["vector"] if tn and n_copies >= 2 else None
    sol, table = search_space_groups(data_mtz, model_pdb, copies=n_copies, workdir=wd / f"xpg_{tag}", log=log,
                                     tncs=tvec, twin=tw)
    z = lambda s_: float(s_.get("LLGZ", s_.get("tz", np.nan)))
    if tvec is not None and not z(sol) >= SOLVED_Z:
        log(f"MR: tNCS pairs give no clear solution (Z {z(sol):.1f}); single copies as well")
        sol2, table2 = search_space_groups(data_mtz, model_pdb, copies=n_copies, workdir=wd / f"xpg_{tag}_single",
                                           log=log, twin=tw)
        if sol2.get("LLG_all", -np.inf) > sol.get("LLG_all", -np.inf):
            sol, table = sol2, table2
    return dict(n=n_copies, placed=sol.get("copies"), TFZ=z(sol), LLG=float(sol.get("LLG_all", np.nan)),
                sg=sol.get("sg"), pdb=sol.get("pdb"), sg_table=table, tncs_pairs=int(sol.get("tncs_pairs", 0)))


def _reindex_candidates(cell_run, cell_alt, tol=0.02):
    """Integer index transformations h_alt = R h_run between two cells of one lattice, from their metrics (for runs
    whose decision audit did not record it). R and -R are equivalent for intensities: one of each pair."""
    import itertools
    G = lambda c: np.linalg.inv(np.array(gemmi.UnitCell(*c).metric_tensor().as_mat33().tolist()))
    Gr, Ga = G(cell_run), G(cell_alt)
    out = []
    for e in itertools.product((-1, 0, 1), repeat=9):
        R = np.array(e).reshape(3, 3)
        if abs(round(np.linalg.det(R))) != 1 or any(np.array_equal(-R, q) for q in out):
            continue
        if np.abs(R.T @ Ga @ R - Gr).max() <= tol * np.abs(Gr).max():
            out.append(R)
    return out


def expand_model(pdb, M, cell, sg, out):
    """The placed model (all copies of its space group) in another cell setting of the same lattice: fractional
    coordinates x' = M x, copies related by the new space group's operators dropped. -> out (PDB)."""
    import string
    st = gemmi.read_structure(str(pdb))
    st.setup_entities()
    st.remove_ligands_and_waters()
    F = np.array(st.cell.frac.mat.tolist())
    atoms = [at for ch in st[0] for res in ch for at in res]
    X = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms]) @ F.T
    uc = gemmi.UnitCell(*cell)
    O = np.array(uc.orth.mat.tolist())
    ops_new = [(np.array(o.rot) / o.DEN, np.array(o.tran) / o.DEN) for o in gemmi.SpaceGroup(sg).operations()]
    keep = []
    for o in gemmi.SpaceGroup(st.spacegroup_hm).operations():
        Y = (X @ (np.array(o.rot) / o.DEN).T + np.array(o.tran) / o.DEN) @ M.T
        c = Y.mean(0)
        same = False
        for K in keep:
            for W, t in ops_new:
                dv = c - (K.mean(0) @ W.T + t)
                if np.linalg.norm(O @ (dv - np.round(dv))) < 2.0:
                    same = True
        if not same:
            keep.append(Y)
    names = iter(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    new = gemmi.Structure()
    new.cell, new.spacegroup_hm = uc, gemmi.SpaceGroup(sg).hm
    mdl = gemmi.Model("1")
    for Y in keep:
        P = Y @ O.T
        k = 0
        for ch in st[0]:
            c = gemmi.Chain(next(names))
            for res in ch:
                r = gemmi.Residue()
                r.name, r.seqid, r.entity_type = res.name, res.seqid, res.entity_type
                for at in res:
                    a = at.clone()
                    a.pos = gemmi.Position(*P[k])
                    k += 1
                    r.add_atom(a)
                c.add_residue(r)
            mdl.add_chain(c)
    new.add_model(mdl)
    new.setup_entities()
    new.write_pdb(str(out))
    return Path(out)


def _is_subgroup(sg_low, sg_high, M):
    """Every rotation of sg_low (in its setting) is one of sg_high's, with x_low = M x_high."""
    Mi = np.linalg.inv(M)
    high = {tuple(np.round(np.array(o.rot) / o.DEN).astype(int).ravel()) for o in gemmi.SpaceGroup(sg_high).operations()}
    return all(tuple(np.round(Mi @ (np.array(o.rot) / o.DEN) @ M).astype(int).ravel()) in high
               for o in gemmi.SpaceGroup(sg_low).operations())


def compare_on_same_data(high, low, wd):
    """The higher-symmetry solution expanded into the lower-symmetry data's setting, and the lower-symmetry
    solution, both scored on the lower-symmetry data (same reflections, same free set). high / low: dicts with
    pdb, mtz, sg, reindex (run -> data, None: unknown). Returns dict(rfree_high, rfree_low, rwork_high, rwork_low)
    or None when the lower group is not a subgroup."""
    from .llg import model_r_factors
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    cell_low = gemmi.read_mtz_file(str(low["mtz"])).cell.parameters
    if high["reindex"] is not None and low["reindex"] is not None:
        Rs = [np.asarray(low["reindex"])]
        Rh = np.asarray(high["reindex"])
    else:                          # older runs: the transformation from the cells
        Rh = np.eye(3, dtype=int)
        Rs = _reindex_candidates(gemmi.read_mtz_file(str(high["mtz"])).cell.parameters, cell_low)
    best = None
    for i, R in enumerate(Rs):
        M = np.linalg.inv(R).T @ Rh.T                  # fractional coordinates: high setting -> low setting
        if not _is_subgroup(low["sg"], high["sg"], M):
            continue
        exp = expand_model(high["pdb"], M, cell_low, low["sg"], wd / f"expanded_{i}.pdb")
        rw, rf = model_r_factors(low["mtz"], exp, sg=low["sg"])
        if best is None or rf < best[1]:
            best = (rw, rf)
    if best is None:
        return None
    rw_l, rf_l = model_r_factors(low["mtz"], low["pdb"], sg=low["sg"])
    return dict(rwork_high=best[0], rfree_high=best[1], rwork_low=rw_l, rfree_low=rf_l)


def choose_point_group(scored, same_data=None, rfree_noise=0.01):
    """Data / space group of the MR results {label: result}. LLG and R-free are not comparable between data merged
    in different point groups (other reflections, other free sets), so a lower-symmetry solution is compared with the
    higher-symmetry one on its own data (compare_on_same_data: same_data[(high, low)]). Among the solutions
    (TFZ >= SOLVED_Z) the highest symmetry is kept unless a lower one has an R-free lower by at least rfree_noise there;
    without a solution, the highest TFZ. Returns (label, reason)."""
    order = {k: len(gemmi.SpaceGroup(v["sg"]).operations().sym_ops) for k, v in scored.items()}
    solved = [k for k, v in scored.items() if v.get("solved")]
    if not solved:
        best = max(scored, key=lambda k: scored[k]["TFZ"])
        return best, f"no solution (TFZ < {SOLVED_Z:g}); highest TFZ {scored[best]['TFZ']:.1f}"
    best = max(solved, key=lambda k: (order[k], scored[k]["TFZ"]))
    same_data = same_data or {}
    better = [(c["rfree_high"] - c["rfree_low"], low, c) for (h, low), c in same_data.items()
              if h == best and low in solved and c and c["rfree_low"] <= c["rfree_high"] - rfree_noise]
    if better:
        _, low, c = max(better)
        return low, (f"on the {scored[low]['sg']} data R-free {c['rfree_low']:.3f} vs {c['rfree_high']:.3f} for the "
                     f"{scored[best]['sg']} solution: the lower symmetry fits better")
    cmp = [f"on the {scored[low]['sg']} data R-free {c['rfree_high']:.3f} vs {c['rfree_low']:.3f}"
           for (h, low), c in same_data.items() if h == best and c]
    return best, "highest symmetry among the solutions" + (f" ({'; '.join(cmp)})" if cmp else "")


def run_mr(run_dir, model, sequence, copies=None, alternatives=None, log=print, msa=True):
    from .matthews import matthews_table, model_components, mw_of_components, read_components, write_components
    run_dir = Path(run_dir).resolve()
    wd = run_dir / "mr"
    wd.mkdir(exist_ok=True)
    if model is None and sequence is None:
        raise ValueError("xpg mr needs a model (--model) or the content to predict one (--sequence)")
    if model is None:
        from .predict import predict_model
        model = predict_model(sequence, wd / "predict", msa=msa, log=log)
    comps = read_components(sequence) if sequence else model_components(model)
    if not sequence:                  # composition from the model's chains (Matthews)
        sequence = write_components(comps, wd / "model_content.fasta")
        log("MR: content from the model: " + ", ".join(f"{n} ({t}, {len(q)})" for n, t, q in comps))
    pdb = wd / "search_model.pdb"
    predicted, n_res, n_ch = prepare_model(model, pdb)
    log(f"MR: search model {Path(model).name}: {n_res} residues in {n_ch} chain(s)"
        f"{' (predicted: pLDDT >= 70, B from the expected error)' if predicted else ''}")
    data = {"run": run_dir / "xpg_merged_iso.mtz"}
    reindex = {"run": np.eye(3, dtype=int)}              # run indices -> each data set's indices
    m0 = gemmi.read_mtz_file(str(data["run"]))
    sg_run = m0.spacegroup
    # alternative point groups: explicit, else the run's decision audit (the close call's alternatives already merged,
    # on any cell basis), else the same-basis alternatives of the close call in the run's events
    alts = [a.strip() for a in alternatives.split(",")] if alternatives else []
    summ = json.loads((run_dir / "xpg_summary.json").read_text()) if (run_dir / "xpg_summary.json").exists() else {}
    audit = [] if alternatives else [a for a in summ.get("audit_merges", []) if (run_dir / a["file"]).exists()]
    for a in audit:
        lab = Path(a["file"]).name[len("xpg_"):-len("_merged_iso.mtz")]
        log(f"MR: also searching {a['label']} in {a['space_group']} (decision audit, {a['file']})")
        data[lab] = run_dir / a["file"]
        reindex[lab] = np.array(a["reindex"]) if "reindex" in a else None
    if not alts and not audit and (run_dir / "events.jsonl").exists():
        for line in (run_dir / "events.jsonl").read_text().splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "decision" and ev.get("name") == "point group" and ev.get("low_margin"):
                alts = _same_basis_alternatives(sg_run, ev.get("alternatives", []))
    if not alternatives:
        # subgroups the twinning check could not tell apart (xpg_unmerged_<SG>.mtz: true space group may be a
        # subgroup twinned to look like the merged one)
        for f in sorted(run_dir.glob("xpg_unmerged_*.mtz")):
            try:
                a = gemmi.read_mtz_file(str(f)).spacegroup.hm
            except Exception:  # noqa: BLE001
                continue
            if a not in alts:
                alts.append(a)
                log(f"MR: twinning check lists {a} as a possible true space group ({f.name})")
    from .lattice import _rotations
    for a in alts:
        lab = a.replace(" ", "")
        if _rotations(a) == _rotations(sg_run.hm) or lab in data:
            continue
        log(f"MR: also merging in {a} (alternative point group)")
        data[lab] = merge_in(run_dir, a, wd / f"xpg_{lab}", log=log)
        reindex[lab] = np.eye(3, dtype=int)
    mw = mw_of_components(comps)
    results = {}
    for lab, mtz in data.items():
        m = gemmi.read_mtz_file(str(mtz))
        rows = matthews_table(m.cell.parameters, m.spacegroup, mw, resolution=m.resolution_high())
        best = max(rows, key=lambda r: r["p"])["n"] if rows else 1
        ns = [copies] if copies else sorted({max(1, best - 1), best, best + 1} & ({r["n"] for r in rows} | {best}))
        log(f"MR [{lab}]: {m.spacegroup.xhm()}, {m.resolution_high():.2f} A, Matthews: " +
            ", ".join(f"{r['n']} copies {r['solvent']:.0f}% ({r['p']:.2f})" for r in rows if r["p"] > 0.01))
        lw = wd / lab
        lw.mkdir(exist_ok=True)
        patho = data_pathologies(mtz, log=log, label=lab)
        tried = []
        for n in ns:
            try:
                tried.append(_xpg_search(lw, mtz, pdb, n, f"n{n}", log=log, patho=patho))
            except RuntimeError as e:
                tried.append(dict(n=n, error=str(e)[:120]))
            t = tried[-1]
            log(f"MR [{lab}]: {n} copies -> placed {t.get('placed')} in {t.get('sg')}, Z {t.get('TFZ', float('nan')):.1f}, LLG {t.get('LLG', float('nan')):.1f}")
        ok = [t for t in tried if t.get("pdb")]
        res = dict(space_group_data=m.spacegroup.xhm(), dmin=m.resolution_high(), searches=tried, pathologies=patho)
        if ok:
            top = max(ok, key=lambda t: t["LLG"] if np.isfinite(t["LLG"]) else -1e9)
            from .llg import model_r_factors, refine_twin_fraction
            twin = None
            if patho.get("twin_test"):
                law = patho["twin_test"]["law"]
                a_ml, gain = refine_twin_fraction(mtz, top["pdb"], law)
                res["twin_fraction_model"] = a_ml
                log(f"MR [{lab}]: twin law {law}: fraction {a_ml:.3f} against the placed model (LLG {gain:+.1f})")
                twin = dict(law=law, alpha=a_ml) if a_ml > 0 else None
            rw, rf = model_r_factors(mtz, top["pdb"], twin=twin)
            res.update(top, Rwork=rw, Rfree=rf, solved=bool(top["TFZ"] >= SOLVED_Z))
            ok_z = f"  (solved: TFZ >= {SOLVED_Z:g})" if top["TFZ"] >= SOLVED_Z else ""
            log(f"MR [{lab}]: best {top['placed']} copies in {top['sg']}: TFZ {top['TFZ']:.1f}, LLG {top['LLG']:.0f}; "
                f"R {rw:.3f} / R-free {rf:.3f} (placed model){ok_z}")
        results[lab] = res
    scored = {k: v for k, v in results.items() if np.isfinite(v.get("LLG", np.nan))}
    order = {k: len(gemmi.SpaceGroup(v["sg"]).operations().sym_ops) for k, v in scored.items()}
    same = {}
    for h in [k for k in scored if scored[k].get("solved")]:
        for low in [k for k in scored if scored[k].get("solved") and order[k] < order[h]]:
            c = compare_on_same_data(dict(pdb=scored[h]["pdb"], mtz=data[h], sg=scored[h]["sg"], reindex=reindex[h]),
                                     dict(pdb=scored[low]["pdb"], mtz=data[low], sg=scored[low]["sg"],
                                          reindex=reindex[low]), wd / f"compare_{h}_{low}")
            if c:
                same[(h, low)] = c
                results[low].setdefault("same_data", {})[h] = c
                log(f"MR: {scored[h]['sg']} solution on the {scored[low]['sg']} data: R-free {c['rfree_high']:.3f} "
                    f"vs {c['rfree_low']:.3f} for the {scored[low]['sg']} solution")
    if len(scored) > 1:
        best, why = choose_point_group(scored, same)
        results["choice"] = dict(data=best, space_group=scored[best].get("sg"), reason=why)
        log(f"MR: best data / space group: {best} ({scored[best].get('sg')}): {why}; others: " +
            ", ".join(f"{k} {v.get('sg')} TFZ {v['TFZ']:.1f}"
                      for k, v in scored.items() if k != best))
    (run_dir / "xpg_mr.json").write_text(json.dumps(dict(format="xpg-mr/1", **results), indent=1, default=str))
    from .pdf_report import write_pdf_report
    write_pdf_report(run_dir, log=log)            # the one-page summary, now with the MR result
    return results

"""xpg.llg: LLG of a placed model against amplitudes simulated from it (> 0, sigma_A near 1) and against amplitudes of
an unrelated model (~ 0)."""
import gemmi
import numpy as np

from xpg.llg import placed_model_llg


def _model(seed, n=300, cell=(40.0, 45.0, 50.0, 90, 90, 90)):
    rng = np.random.default_rng(seed)
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*cell)
    st.spacegroup_hm = "P 1"
    m, ch = gemmi.Model("1"), gemmi.Chain("A")
    for i in range(n):
        r = gemmi.Residue()
        r.name, r.seqid = "GLY", gemmi.SeqId(i + 1, " ")
        a = gemmi.Atom()
        a.name, a.element, a.occ, a.b_iso = "CA", gemmi.Element("C"), 1.0, 20.0
        a.pos = gemmi.Position(*(rng.uniform(0.2, 0.8, 3) * np.array(cell[:3])))
        r.add_atom(a)
        ch.add_residue(r)
    m.add_chain(ch)
    st.add_model(m)
    st.setup_entities()
    return st


def _write_data(st, path, dmin=2.5, seed=0):
    dc = gemmi.DensityCalculatorX()
    dc.d_min = dmin
    dc.set_grid_cell_and_spacegroup(st)
    dc.set_refmac_compatible_blur(st[0])
    dc.put_model_density_on_grid(st[0])
    asu = gemmi.transform_map_to_f_phi(dc.grid).prepare_asu_data(dmin=dmin, with_000=False, with_sys_abs=False)
    hkl = np.asarray(asu.miller_array, float)
    inv_d2 = np.asarray(asu.make_1_d2_array())
    F = np.abs(np.asarray(asu.value_array)) * np.exp(dc.blur * inv_d2 / 4)
    rng = np.random.default_rng(seed)
    sig = 0.05 * F.mean() * np.ones_like(F)
    Fo = np.abs(F + rng.normal(0, 1, F.size) * sig)
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = st.find_spacegroup()
    mtz.set_cell_for_all(st.cell)
    mtz.add_dataset("d")
    mtz.add_column("F", "F")
    mtz.add_column("SIGF", "Q")
    mtz.set_data(np.c_[hkl, Fo, sig].astype(np.float32))
    mtz.write_to_file(str(path))


def test_llg_true_vs_unrelated(tmp_path):
    st = _model(1)
    _write_data(st, tmp_path / "d.mtz")
    st.write_pdb(str(tmp_path / "true.pdb"))
    _model(2).write_pdb(str(tmp_path / "other.pdb"))
    good = placed_model_llg(tmp_path / "d.mtz", tmp_path / "true.pdb", solvent=False, aniso=False, nbins=6)
    bad = placed_model_llg(tmp_path / "d.mtz", tmp_path / "other.pdb", solvent=False, aniso=False, nbins=6)
    assert good["LLG"] > 1000 and np.median(good["sigmaA"]) > 0.8
    assert abs(bad["LLG"]) < 0.05 * good["LLG"]


def test_mr_point_group_choice():
    """Solutions in different point groups are compared on the lower-symmetry data: the higher symmetry is kept
    unless the lower one's R-free there is lower by >= 0.01."""
    from xpg.mr import choose_point_group
    res = {"run": dict(sg="P 1 21 1", TFZ=37.5, LLG=784.0, solved=True), "P1": dict(sg="P 1", TFZ=14.9, LLG=995.5,
                                                                                   solved=True)}
    same = {("run", "P1"): dict(rfree_high=0.421, rfree_low=0.418, rwork_high=0.430, rwork_low=0.438)}
    assert choose_point_group(res, same)[0] == "run"
    same[("run", "P1")]["rfree_low"] = 0.40
    assert choose_point_group(res, same)[0] == "P1"
    res = {"run": dict(sg="P 1 21 1", TFZ=5.0, LLG=784.0, solved=False), "P1": dict(sg="P 1", TFZ=6.0, LLG=995.5,
                                                                                    solved=False)}
    assert choose_point_group(res)[0] == "P1"


def test_mr_content(tmp_path):
    """MR content: protein / RNA / DNA chains and ligands from FASTA (type in the header or recognised), masses for
    Matthews, OpenFold3 query chains; a model alone gives its polymer chains."""
    import gemmi
    from xpg.matthews import model_components, mw_of_components, read_components
    from xpg.predict import query_chains
    f = tmp_path / "c.fasta"
    f.write_text(">A protein\nMSTIEELIKAAKEL\n>B\nGGACUUCGGUCC\n>C dna\nATGCATGCAT\n>L ligand\nATP\n>M\nsmiles:CC(=O)O\n")
    comps = read_components(f)
    assert [t for _, t, _ in comps] == ["protein", "rna", "dna", "ligand", "ligand"]
    assert 4000 < mw_of_components(comps) < 9500
    q = query_chains(comps)
    assert q[1] == {"molecule_type": "rna", "chain_ids": ["B"], "description": "B", "sequence": "GGACUUCGGUCC"}
    assert q[3]["ccd_codes"] == ["ATP"] and q[4]["smiles"] == "CC(=O)O"
    st = gemmi.Structure()
    m = gemmi.Model("1")
    ch = gemmi.Chain("A")
    for i, name in enumerate(("MET", "SER", "THR")):
        r = gemmi.Residue()
        r.name, r.seqid = name, gemmi.SeqId(i + 1, " ")
        a = gemmi.Atom()
        a.name, a.element, a.pos = "CA", gemmi.Element("C"), gemmi.Position(3.8 * i, 0, 0)
        r.add_atom(a)
        ch.add_residue(r)
    m.add_chain(ch)
    st.add_model(m)
    st.write_pdb(str(tmp_path / "m.pdb"))
    assert model_components(tmp_path / "m.pdb") == [("A", "protein", "MST")]


def test_tncs_modulation():
    """The tNCS modulation 1 + f cos(2 pi h.t) of the expected intensities is recovered from normalised intensities."""
    from xpg.llg import tncs_modulation
    rng = np.random.default_rng(3)
    hkl = rng.integers(-20, 21, (20000, 3))
    t = np.array([0.5, 0.0, 0.25])
    true = 1 + 0.6 * np.cos(2 * np.pi * (hkl @ t))
    m = tncs_modulation(hkl, rng.exponential(1.0, len(hkl)) * true, rng.uniform(0.01, 0.1, len(hkl)), t)
    assert np.corrcoef(m, true)[0, 1] > 0.99 and np.abs(m - true).mean() < 0.05


def test_twinned_llg(tmp_path):
    """Twinned data: the twin mates pair up, the twinned target scores the true model higher than the untwinned one,
    and the twin fraction refined against the model is close to the true one."""
    import gemmi
    from xpg.llg import placed_model_llg, refine_twin_fraction, twin_mates
    st = _model(1, cell=(40.0, 45.0, 50.0, 90, 90, 90))
    _write_data(st, tmp_path / "d.mtz", dmin=2.5)
    st.write_pdb(str(tmp_path / "true.pdb"))
    m = gemmi.read_mtz_file(str(tmp_path / "d.mtz"))
    hkl = np.asarray(m.make_miller_array(), np.int64)
    j = twin_mates(hkl, m.spacegroup, "-x,-y,z")
    ok = (j >= 0) & (j[np.maximum(j, 0)] == np.arange(len(j)))
    assert ok.mean() > 0.9
    a = np.array(m, copy=True)
    F = a[:, 3].astype(float)
    a[:, 3] = np.where(j >= 0, np.sqrt(0.7 * F ** 2 + 0.3 * F[np.maximum(j, 0)] ** 2), F)
    m.set_data(a)
    m.write_to_file(str(tmp_path / "tw.mtz"))
    kw = dict(solvent=False, aniso=False, nbins=6)
    plain = placed_model_llg(tmp_path / "tw.mtz", tmp_path / "true.pdb", **kw)["LLG"]
    twin = placed_model_llg(tmp_path / "tw.mtz", tmp_path / "true.pdb", twin=dict(law="-x,-y,z", alpha=0.3), **kw)["LLG"]
    assert twin > 1.2 * plain
    alpha, gain = refine_twin_fraction(tmp_path / "tw.mtz", tmp_path / "true.pdb", "-x,-y,z")
    assert abs(alpha - 0.3) <= 0.1 and gain > 0


def test_mr_tncs_pair(tmp_path):
    """tNCS: a crystal of two copies related by a translation; with the search model in the right orientation, the
    placement ranked best by the fast LLG is the true pair (up to symmetry and origin, on the translation grid)."""
    import gemmi
    from xpg.mrsearch import model_transform, rank_placements, rotation_search, translation_search
    rng = np.random.default_rng(5)
    cell, t = (70.0, 50.0, 45.0, 90, 90, 90), np.array([0.5, 0.05, 0.1])
    xyz = rng.normal(0, 5.0, (250, 3)) + np.array([15.0, 12.0, 10.0])
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*cell)
    st.spacegroup_hm = "P 21 21 21"
    O = np.array(st.cell.orth.mat.tolist())
    model = gemmi.Model("1")
    for name, shift in (("A", 0.0), ("B", 1.0)):
        ch = gemmi.Chain(name)
        for i, x in enumerate(xyz + shift * O @ t):
            r = gemmi.Residue()
            r.name, r.seqid = "GLY", gemmi.SeqId(i + 1, " ")
            at = gemmi.Atom()
            at.name, at.element, at.occ, at.b_iso, at.pos = "CA", gemmi.Element("C"), 1.0, 20.0, gemmi.Position(*x)
            r.add_atom(at)
            ch.add_residue(r)
        model.add_chain(ch)
    st.add_model(model)
    st.setup_entities()
    _write_data(st, tmp_path / "d.mtz", dmin=3.0)
    one = st.clone()
    one[0].remove_chain("B")
    one.write_pdb(str(tmp_path / "one.pdb"))
    _, _, info = rotation_search(tmp_path / "d.mtz", tmp_path / "one.pdb", sg="P 21 21 21", tncs=t, log=lambda *a: None)
    info["Gc"], info["Lc"], _, _ = model_transform(tmp_path / "one.pdb", 3.5, box_pad=2.5)
    peaks = translation_search(np.eye(3), info, top=40)
    _, p, tn = rank_placements(np.eye(3), info, peaks, [t, -t])[0]
    a = np.linalg.solve(O, xyz.mean(0))
    true = (a, a + t)
    ops = [(np.array(o.rot) / o.DEN, np.array(o.tran) / o.DEN) for o in st.find_spacegroup().operations()]
    shifts = np.array(np.meshgrid([0, .5], [0, .5], [0, .5])).reshape(3, -1).T
    dist = lambda u, v: np.linalg.norm(O @ ((u - v + 0.5) % 1.0 - 0.5))
    best = min(max(dist(S @ x + tau + o, y) for x, y in zip(pair, true))
               for S, tau in ops for o in shifts for pair in ((p, p + tn), (p + tn, p)))
    assert best < 1.5

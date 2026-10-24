"""Regression tests for edge cases found on the held-out validation set (2026-10-17)."""
import numpy as np


def _hex_long_axis(seed=1):
    from scipy.spatial.transform import Rotation
    Rm = Rotation.random(random_state=seed).as_matrix()
    A = np.array([[37.0, 0, 0], [-18.5, 32.04, 0], [0, 0, 478.0]]) @ Rm.T     # real-space rows (9QHA-like)
    return A, np.linalg.inv(A)


def test_fft1d_finds_long_axis():
    """9QHA (c = 478 A): the 1D FFTs over directions recover the long axis that a 3D grid smears."""
    from xpg.indexing import fft1d_candidate_vectors
    A, Astar = _hex_long_axis()
    h = np.array(np.meshgrid(np.arange(-11, 12), np.arange(-11, 12), np.arange(-140, 141), indexing="ij")).reshape(3, -1).T
    p = h @ Astar.T
    d2 = np.einsum("ij,ij->i", p, p)
    p = p[(d2 > 0) & (d2 <= 1 / 3.5 ** 2)]
    p = p[np.random.default_rng(0).random(len(p)) < 0.05]
    t, v, _ = fft1d_candidate_vectors(p, 3.5, 700.0)
    L = np.linalg.norm(t, axis=1)
    cos_c = np.abs(t @ A[2] / np.linalg.norm(A[2])) / np.maximum(L, 1e-9)
    assert np.any((np.abs(L - 478.0) < 15.0) & (cos_c > 0.99))


def test_match_basis_to_main_setting():
    """An extra lattice of the same crystal form found in another basis (GLF-18: all-acute 75.8 / 75.9 / 68.9 deg)
    is put into the main lattice's setting (same metric)."""
    from xpg.indexing import _match_basis
    from xpg.geometry import cell_from_Astar
    R_main = np.array([[37.6, 0.0, 0.0], [-18.4, 48.5, 0.0], [-14.6, -9.3, 55.8]])     # triclinic, reduced-like
    Astar_main = np.linalg.inv(R_main)
    T0 = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])                                 # another basis of the lattice
    from scipy.spatial.transform import Rotation
    Q = Rotation.random(random_state=7).as_matrix()
    R_sub = (T0 @ R_main) @ Q.T                                                       # rotated crystal, other basis
    T = _match_basis(np.linalg.inv(R_sub), Astar_main)
    cell_new = cell_from_Astar(np.linalg.inv(T @ R_sub))
    np.testing.assert_allclose(cell_new, cell_from_Astar(Astar_main), atol=1e-6)
    assert round(abs(np.linalg.det(T))) == 1


def test_decode_bins_flags():
    """Binned shoebox sums on IEEE-encoded frames: an overload in any frame wins (-2), then masked (-1); empty bins
    are unrecorded (stored 0); values are stored + SB_OFF."""
    from xpg.shoebox import SB_OFF, _decode_bins
    arr = np.array([[[[5.0, np.inf], [np.nan, 0.0]], [[2.0, 3.0], [4.0, 1.0]]]], np.float32)   # (1, B=2, 2, 2)
    cnt = np.array([[2, 0]], np.int16)
    _decode_bins(arr, cnt)
    np.testing.assert_allclose(arr[0, 0], np.array([[5.0 + SB_OFF, SB_OFF - 1.0], [SB_OFF - 2.0, SB_OFF]], np.float32))
    assert np.all(arr[0, 1] == 0.0)
    # IEEE propagation the gather relies on: nan + inf = nan (overload wins), inf + finite = inf
    assert np.isnan(np.float32(np.nan) + np.float32(np.inf)) and np.isinf(np.float32(np.inf) + np.float32(3.0))


def test_remove_supercell_empty_class():
    """7AO2 (tNCS): a 2x supercell whose odd class holds no spot at all (no stray spots to deplete) is still reduced."""
    from xpg.indexing import remove_supercell
    from xpg.geometry import cell_from_Astar
    R_true = np.array([[63.3, 0.0, 0.0], [0.0, 96.7, 0.0], [0.0, 0.0, 106.9]])
    rng = np.random.default_rng(3)
    h = rng.integers(-15, 16, size=(4000, 3))
    p = h @ np.linalg.inv(R_true).T                      # reciprocal vectors of the true lattice points
    R_super = R_true.copy(); R_super[0] *= 2.0           # index basis with a doubled a axis
    Rn = remove_supercell(R_super, p)
    np.testing.assert_allclose(sorted(cell_from_Astar(np.linalg.inv(Rn))[:3]), [63.3, 96.7, 106.9], atol=1e-6)
    # a correct basis is left alone
    np.testing.assert_allclose(remove_supercell(R_true, p), R_true)


def test_bitshuffled_frames_finished_on_device(tmp_path):
    """Workers' LZ4 stage + the torch bit unshuffle give exactly ImageSet.get: a frame size that is not a multiple
    of the bitshuffle block or of 8, saturated / overloaded / masked pixels, uint32 (Eiger) and uint16."""
    from types import SimpleNamespace
    import h5py
    import hdf5plugin
    import torch
    from xpg.io.bslz4 import lz4_stage, unshuffle_frame_torch
    from xpg.io.images import open_images
    from xpg.simulate import _open_eiger
    ny, nx = 37, 61                                   # 2257 pixels: one 2048-element block + 208 + 1
    rng = np.random.default_rng(3)
    mask = rng.random((ny, nx)) < 0.05                # True = masked
    spec = SimpleNamespace(nframes=3, ny=ny, nx=nx, qx=0.075, orgx=30.0, orgy=18.0, distance=200.0,
                           wavelength=1.0, phi0=0.0, osc=0.1)
    fd, ds = _open_eiger(tmp_path, spec, mask)
    for k in range(3):
        fr = rng.poisson(5.0 * (k + 1), (ny, nx)).astype(np.uint32)
        fr.flat[rng.choice(ny * nx, 20, replace=False)] = 2 ** 32 - 1          # saturated / module gaps
        fr.flat[rng.choice(ny * nx, 20, replace=False)] = 2 ** 20 + 7          # above the count cut-off
        ds[k] = fr
    fd.close()
    im = open_images(str(tmp_path / "sim_master.h5"))
    im.cache_size = 0
    assert im.bitshuffled_ok()
    nel = ny * nx
    sbad = torch.as_tensor(~np.asarray(im.static_mask)).reshape(-1)
    for f in (1, 2, 3):
        buf = np.zeros(nel * 4, np.uint8)
        block = lz4_stage(im.read_chunk(f), buf, 4)
        img = unshuffle_frame_torch(torch.from_numpy(buf), nel, 4, block, im.overload, im.min_valid, sbad)
        np.testing.assert_array_equal(img.numpy().reshape(ny, nx), im.get(f))
    # uint16 at the codec level
    with h5py.File(tmp_path / "u16.h5", "w") as h:
        a = rng.integers(0, 2 ** 16, (ny, nx), dtype=np.uint16)
        d16 = h.create_dataset("d", data=a[None], chunks=(1, ny, nx), **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        _, raw = d16.id.read_direct_chunk((0, 0, 0))
    buf = np.zeros(nel * 2, np.uint8)
    block = lz4_stage(raw, buf, 2)
    img = unshuffle_frame_torch(torch.from_numpy(buf), nel, 2, block)
    ref = a.astype(np.float32)
    ref[a == 2 ** 16 - 1] = -1.0
    np.testing.assert_array_equal(img.numpy().reshape(ny, nx), ref)

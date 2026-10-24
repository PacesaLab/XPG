# XPG — X-tal data Processing on GPU

XPG takes your diffraction images and gives you merged, scaled data ready for refinement. It does spot finding,
indexing, integration, symmetry, scaling and merging on the GPU, and logs every decision it makes so you can check
it. It runs on NVIDIA, AMD and Intel GPUs, on Apple silicon, or on the CPU, and needs no other crystallographic
software.

https://github.com/user-attachments/assets/1729523e-c3cc-421e-9505-a4be930f7b00

## Installation

```bash
git clone https://github.com/PacesaLab/XPG.git
cd XPG
bash install.sh
```

This creates a Python environment in `XPG/.venv` with the right PyTorch build for your machine and puts the `xpg`
command into `~/.local/bin`. On a cluster, clone into scratch and run the GPU check on a compute node with
`bash install.sh --srun "--gpus=1 --time=00:10:00"`. All options are in the [guide](docs/GUIDE.md).

## Running XPG

```bash
xpg --images /data/sample_master.h5 -o run1    # Eiger / NeXus, CBF or SMV, geometry read from the headers
xpg XDS.INP -o run1                             # an XDS.INP works too, it is converted
xpg serve run1                                  # live dashboard at http://127.0.0.1:8765
```

What you get in `run1/`:

| File | Content |
|---|---|
| `xpg_merged_iso.mtz` | merged data with free-R flags, ready for refinement |
| `xpg_anisotropic.mtz` | the same, with an anisotropic cut-off |
| `xpg_unmerged.mtz` | scaled unmerged data |
| `xpg_report.pdf` | two-page summary with images, graphs and anything worth checking |
| `xpg_summary.html` | full statistics and decisions |
| `xpg.log` | the full log |

If a run stops, pick it up from the last checkpoint: `xpg xpg.toml -o run1 --jobs INTEGRATE SCALE`.

## Merging and molecular replacement

```bash
xpg merge merge.toml                    # scale and merge several data sets
xpg mr run1 --model model.pdb           # molecular replacement on a finished run
xpg mr run1 --sequence content.fasta    # same, with a model predicted by OpenFold3 (install.sh --openfold3)
```

`content.fasta` holds one copy of what is in the asymmetric unit: protein, RNA and DNA chains and ligands (as SMILES
or CCD codes). Everything is predicted together as one complex. If you give a model, nothing is predicted.

All settings, outputs and things to look out for are in the [guide](docs/GUIDE.md). The input and output files
are described in [FORMAT](docs/FORMAT.md).

https://github.com/user-attachments/assets/dcce2021-bfa2-46aa-983d-8ae64aff17d3


## References

XPG is our own implementation of published methods. It builds on:

* **Geometry, integration and scaling**: Kabsch, W. (1988). *J. Appl. Cryst.* 21, 916–924; Kabsch, W. (2010).
  *Acta Cryst.* D66, 125–132 and 133–144.
* **Spot finding** (dispersion): Winter, G. et al. (2018). *Acta Cryst.* D74, 85–97.
* **Symmetry, scaling model and data quality**: Evans, P. (2006). *Acta Cryst.* D62, 72–82; Evans, P. R. (2011).
  *Acta Cryst.* D67, 282–292; Evans, P. R. & Murshudov, G. N. (2013). *Acta Cryst.* D69, 1204–1214.
* **Data quality**: Diederichs, K. (2010). *Acta Cryst.* D66, 733–740 (ISa); Karplus, P. A. & Diederichs, K. (2012).
  *Science* 336, 1030–1033 (CC1/2); Assmann, G., Brehm, W. & Diederichs, K. (2016). *J. Appl. Cryst.* 49,
  1021–1028 (ΔCC1/2).
* **Radiation damage**: Diederichs, K. (2006). *Acta Cryst.* D62, 96–101 (zero-dose extrapolation).
* **Amplitudes**: French, S. & Wilson, K. (1978). *Acta Cryst.* A34, 517–525.
* **Anisotropy**: the anisotropic cut-off follows Tickle, I. J. et al. (2018), Global Phasing Ltd.
* **Twinning and tNCS**: Yeates, T. O. (1997). *Methods Enzymol.* 276, 344–358; Padilla, J. E. & Yeates, T. O.
  (2003). *Acta Cryst.* D59, 1124–1130; Read, R. J., Adams, P. D. & McCoy, A. J. (2013). *Acta Cryst.* D69, 176–183.
* **Copies in the asymmetric unit**: Matthews, B. W. (1968). *J. Mol. Biol.* 33, 491–497; Kantardjieff, K. A. &
  Rupp, B. (2003). *Protein Sci.* 12, 1865–1871.
* **Molecular replacement**: Read, R. J. (1986). *Acta Cryst.* A42, 140–149; Read, R. J. (2001). *Acta Cryst.* D57,
  1373–1382; McCoy, A. J. et al. (2007). *J. Appl. Cryst.* 40, 658–674; Oeffner, R. D. et al. (2022). *Acta Cryst.*
  D78, 1303–1314 (B factors of predicted models).
* **Variable projection**: Golub, G. H. & Pereyra, V. (1973). *SIAM J. Numer. Anal.* 10, 413–432; Kaufman, L.
  (1975). *BIT* 15, 49–57.
* **Bitshuffle**: Masui, K. et al. (2015). *Astron. Comput.* 12, 181–190.

## Software

XPG is written in Python on [PyTorch](https://pytorch.org) (Paszke, A. et al. (2019). *NeurIPS* 32), with
[Triton](https://triton-lang.org) kernels on CUDA. It uses [NumPy](https://numpy.org) (Harris, C. R. et al. (2020).
*Nature* 585, 357–362), [SciPy](https://scipy.org) (Virtanen, P. et al. (2020). *Nat. Methods* 17, 261–272),
[gemmi](https://gemmi.readthedocs.io) (Wojdyr, M. (2022). *J. Open Source Softw.* 7, 4200),
[h5py](https://www.h5py.org), [hdf5plugin](https://github.com/silx-kit/hdf5plugin),
[lz4](https://github.com/python-lz4/python-lz4) and [matplotlib](https://matplotlib.org) (Hunter, J. D. (2007).
*Comput. Sci. Eng.* 9, 90–95). To predict MR search models it can use [OpenFold3](https://github.com/aqlaboratory/openfold-3)
with alignments from the [ColabFold](https://github.com/sokrypton/ColabFold) MMseqs2 server (Mirdita, M. et al.
(2022). *Nat. Methods* 19, 679–682). X-ray attenuation coefficients come from NIST XCOM.

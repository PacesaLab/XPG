# XPG guide

Everything you need to run XPG and understand what it does. For the quick start, see the [README](../README.md); the
input and output files are described in [FORMAT](FORMAT.md).

XPG does the whole processing itself: spot finding, indexing, integration, symmetry, scaling, merging, French-Wilson
amplitudes, the MTZ output and molecular replacement. The only optional extra is OpenFold3, if you want XPG to predict
your MR search model.

## 1. Installation

```bash
git clone https://github.com/PacesaLab/XPG.git
cd XPG
bash install.sh
```

The installer creates a Python environment in `XPG/.venv` with the newest PyTorch for your machine, links `xpg` into
`~/.local/bin` and runs a quick GPU test. To update, run `git pull` and `bash install.sh` again.

| Option | What it does |
|---|---|
| `--prefix DIR` | where to put the environment (default `XPG/.venv`) |
| `--cpu` | CPU-only PyTorch (by default CUDA if there is an NVIDIA driver; on a Mac the Apple GPU is used anyway) |
| `--cuda 126\|128\|130` | pick a CUDA build (cu130 needs driver >= 580, cu128 >= 570, cu126 >= 525) |
| `--rocm 6.4` / `--xpu` | AMD or Intel GPUs |
| `--python EXE` | Python to use (3.10 or newer) |
| `--srun "SLURM OPTIONS"` | run the GPU test on a compute node |
| `--openfold3` | also install OpenFold3 and download its weights (~2 GB) for predicted MR models |
| `--dev` | also install pytest and run the tests |
| `--no-test`, `--no-link` | skip the GPU test / the `~/.local/bin` link |

For CUDA you only need the NVIDIA driver, the rest comes with the wheels.

**On a cluster** clone into scratch and run XPG as a job, never on the login node:

```bash
cd /scratch/$USER
git clone https://github.com/PacesaLab/XPG.git
cd XPG
bash install.sh --srun "--gpus=1 --time=00:10:00"
```

```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
source /scratch/$USER/XPG/.venv/bin/activate
xpg xpg.toml -o run1
```

**Hardware.** NVIDIA, AMD and Intel GPUs, Apple silicon or the CPU. Each step runs on the GPU unless the CPU is faster
for it. Memory use adapts to what the machine has, so the same data set processes on a 32 GB laptop and on an H100.

## 2. Running

```bash
xpg --images /data/x_master.h5 -o run1     # geometry from the image headers (writes run1/xpg.toml)
xpg XDS.INP -o run1                         # an XDS.INP is converted (the xpg.toml is kept)
xpg xpg.toml -o run1                        # from an XPG input file
xpg serve run1                              # dashboard at http://127.0.0.1:8765
```

`--images` takes an Eiger / NeXus master file, a CBF template (`x_????.cbf`) or SMV images and reads the geometry
from the headers, including multi-axis goniometers. XPG prints the geometry it found: check it the first time you
use a beamline.

| Option | What it does |
|---|---|
| `-o DIR` | output directory |
| `--jobs STAGE ...` | run only these stages (`DETECTOR BACKGROUND SPOTS INDEX MASK INTEGRATE SCALE`, `ALL`), or `PASS2` to redo the second pass |
| `--pass2 auto\|on\|off` | second integration pass (default auto); `--single-pass` = off |
| `--export-legacy` | also write the legacy-format text files older programs read |
| `--no-aniso` | skip the anisotropy analysis |
| `--allow-centrosymmetric` | small molecules |
| `--cpu` | ignore the GPU |
| `xpg convert XDS.INP` | only convert, write the matching xpg.toml |
| `xpg report RUN` | rewrite the PDF summary |

**Resuming.** Every stage writes a checkpoint. If a run stops, continue with the remaining stages, for example
`xpg xpg.toml -o run1 --jobs INTEGRATE SCALE`. To change a setting, edit `run1/xpg.toml` and rerun from the first
stage it affects, or use the rerun form in the dashboard.

## 3. Outputs

| File | Content |
|---|---|
| `xpg_merged_iso.mtz` | merged data to the isotropic CC1/2 = 0.3 limit: I, F (French-Wilson), anomalous columns, free-R flags |
| `xpg_anisotropic.mtz` | merged data inside the anisotropic cut-off surface, with anisotropic French-Wilson amplitudes |
| `xpg_unmerged.mtz` + `xpg_geometry.json` | scaled unmerged data and the refined geometry |
| `xpg_report.pdf` | two-page summary: key numbers, a diffraction image, statistics, what to check, graphs, MR |
| `xpg_summary.html` / `.md` / `.json` | full statistics, decisions and warnings |
| `xpg_deposit.cif` | statistics for PDB deposition |
| `xpg.log`, `events.jsonl` | the log, and the events the dashboard reads |
| `xpg_<SG>_merged_iso.mtz` | the other space group of a close call (section 5) |
| `xpg_all_lattices_*.mtz`, `xpg_lattice1_*.mtz` | several lattices: all merged, or the first one alone |
| `xpg_detwinned_merged_iso.mtz` | detwinned data, when XPG finds twinning |

The statistics of the isotropic and the anisotropic data are reported side by side.

## 4. How XPG processes your data

| Stage | What happens |
|---|---|
| DETECTOR | detector model: panels, distortion tables, sensor parallax |
| BACKGROUND | background and mask: gaps, untrusted regions, shadows, hot and dead pixels |
| SPOTS | strong-pixel search and spot clustering, touching spots split |
| INDEX | 3D FFT indexing on the GPU with automatic retries, beam-centre search, extra lattices |
| MASK | mask of the indexed experiment |
| INTEGRATE | per-block geometry, then profile fitting with learnt spot profiles; overlapping spots are fitted together; on crowded data a pixel-level model fits all spots and the background at once |
| SCALE | ice rings, symmetry, twinning and tNCS, scaling and error model, frame filters, merging, resolution limit, anisotropy |
| pass 2 | INTEGRATE and SCALE again with the refined geometry and spot widths |

**Second pass.** With `--pass2 auto` it is skipped when the first-pass predictions are already accurate and the data
have a single lattice.

**Crowded data.** `XPG_INTEGRATE_MODE=auto` (the default) switches to the pixel-level forward model when at least
half of the reflections overlap with a neighbour. `fm` forces it, `shoebox` turns it off.

**Several lattices.** Other crystals in the beam and split crystals are indexed one after the other and integrated
together. A lattice is kept only if merging it doesn't cost resolution. If the merge of all lattices is better, it
becomes the main output and the first lattice alone is written as `xpg_lattice1_*.mtz`.

**Bad frames.** Image ranges that lower CC1/2, and stretches without signal, are removed before the final scaling.
You can switch this off with `prune_image_groups`, `prune_chunks` and `prune_fitness_threshold`.

**Where XPG differs from classical processing.**

* Damaged Eiger files are read from their raw compressed chunks instead of failing.
* Indexing keeps going after the first lattice: leftover spots are indexed as further lattices.
* Overlapping reflections are deconvolved instead of rejected.
* The background follows a scaled template (rings, solvent) instead of a flat local plane.
* Frames that hurt the data are found and removed automatically.
* Close symmetry calls are flagged, both answers are merged, and MR can decide between them.
* Molecular replacement, with predicted models, is built in and runs on the GPU too.

## 5. Decisions and how to check them

Every decision is logged with its margin, evidence and alternatives (`xpg.log`, `xpg_summary`, the dashboard, the
PDF). When the margin is small it is marked as a close call and comes with a hint:

| Decision | What to do with a close call |
|---|---|
| point group | the other point group is merged too (`xpg_<SG>_merged_iso.mtz`) and `xpg mr` searches both; to impose one, set `[crystal] space_group` and rerun from INTEGRATE |
| space group | the candidates differ only in screw axes: search them all in MR |
| frame filter | compare with the filter off (`prune_image_groups = false`), or exclude a range yourself |
| lattice merge | both outputs are written: try both in refinement |

Screw-axis enantiomorphs (P41 / P43, P3121 / P3221, ...) can't be told apart from intensities: XPG lists them all
and you need MR to decide.

## 6. The input file (`xpg.toml`)

Every run has an `xpg.toml`: the experiment, and the settings you changed. Everything is optional except
`data.images`. Units are pixels, mm, degrees and Å.

```toml
format = 1

[data]
images = "/data/x_master.h5"      # template with ???? or an HDF5 master file
frames = [1, 3600]
exclude_frames = [[1201, 1260]]

[detector]
beam_x = 2071.3                   # pixels
beam_y = 2190.8
distance = 180.2                  # mm

[crystal]
space_group = 19                  # impose a space group ...
unit_cell = [36.4, 44.2, 124.9, 90, 90, 90]    # ... and cell
friedels_law = false              # anomalous data

[processing]
resolution = [50.0, 1.6]
prune_image_groups = false
```

The switches you are most likely to need, all in `[processing]` (default in brackets): `prune_image_groups` [true],
`prune_chunks` [true], `prune_fitness_threshold` [0.3], `extra_lattices` [true], `min_lattice_separation` [0.5 deg],
`ice_validate` [true], `zero_dose` [false], `strong_pixel` [3], `min_pixels_per_spot` [6]. XPG refuses keys it
doesn't know, so a typo can't go unnoticed. All sections, keys and defaults are in [FORMAT](FORMAT.md).

## 7. Environment variables

The defaults work for almost everything. These are for special data or machines.

| Variable | Default | Use |
|---|---|---|
| `XPG_DEVICE` | auto | `cuda`, `xpu` or `cpu` |
| `XPG_TORCH_THREADS` | all CPUs you have | PyTorch CPU threads |
| `XPG_DECODE_PROCS` | min(16, CPUs - 2) | image decoding processes |
| `XPG_GATHER_THREADS` | min(8, CPUs) | threads that collect the shoeboxes |
| `XPG_SPOTS_CLUSTER_PROCS` | up to 3 on CUDA | spot clustering processes |
| `XPG_INTEGRATE_MEM_GB` | 70% of free memory | memory for the integration buffers |
| `XPG_INTEGRATE_QUEUE_GB` | 8 | finished blocks waiting for the GPU |
| `XPG_SB_DEVICE_GB` | min(8 GB, 1/4 of the GPU) | shoebox memory on the GPU |
| `XPG_CHUNK_ELEMENTS` | auto | voxels per kernel call (lower it on small GPUs) |
| `XPG_FM_MEM_GB`, `XPG_FM_MAX_GPU_GB` | auto, 40 | forward-model memory (host / GPU) |
| `XPG_FM_WINDOW`, `XPG_FM_TILES` | auto | forward-model frame window and detector tiles |
| `XPG_INTEGRATE_MODE` | auto | `auto`, `fm` or `shoebox` |
| `XPG_FM_MULTI_LATTICE` | 0 | `1`: forward model on all lattices together |
| `XPG_FM_SUPERPIXEL` | 0 | forward-model fit on summed pixels (very oversampled spots) |
| `XPG_POS_PRIOR` | automatic | per-block detector position prior `sd_x,sd_y,sd_dist`; `0` = free |
| `XPG_PRIOR_GRID` | 0.01,0.03,0.1,0.3 | widths of the absorption / modulation restraints to choose from |
| `XPG_TNCS_MODE` | scale | tNCS weak class: `scale`, `sum` or `off` |
| `XPG_WEAK_REGIONS` | on | `0` keeps detector regions with systematically weak data |
| `XPG_LATTICE_TRIAGE`, `XPG_LATTICES_PRIMARY` | on | `0` keeps all lattices / keeps the first lattice as the main output |
| `XPG_BAYES_MERGE` | 0 | `1` also writes a Bayesian merge |
| `XPG_GATHER_PROCESS`, `XPG_SPOTS_PROCESS`, `XPG_GPU_UNSHUFFLE`, `XPG_FM_TRITON`, `XPG_COMPILE` | on | `0` switches off the platform-specific fast paths (same results) |
| `XPG_OPENFOLD3` | installed by `--openfold3` | the `run_openfold` executable |

## 8. Dashboard

`xpg serve run1` shows a run live at http://127.0.0.1:8765: the stages, the integration progress, the result and
statistics, the decisions with their hints, the output files for download, the MR result and the graphs. On a
cluster, open an SSH tunnel (`ssh -L 8765:localhost:8765 cluster`) instead of opening the port.

From the dashboard you can also rerun from any stage with changed settings. Each rerun gets its own folder in
`run1/reruns/` and shows up next to the original run, so you can compare them.

## 9. Merging several data sets

```toml
[merge]
output = "merged"
friedels_law = true
reject_nonisomorphous = false      # drop data sets that don't fit the rest

[[merge.input]]
file = "run1/xpg_unmerged.mtz"
reference = true
[[merge.input]]
file = "run2/xpg_unmerged.mtz"
resolution = [50.0, 2.0]
```

`xpg merge merge.toml -o out` sorts out indexing ambiguities between the data sets, scales them together and merges
them. XSCALE.INP files are converted. All keys are in [FORMAT](FORMAT.md).

## 10. Molecular replacement

```bash
xpg mr run1 --model model.pdb                   # with your model
xpg mr run1 --sequence content.fasta            # with a model predicted by OpenFold3
```

The FASTA file lists one copy of the asymmetric unit, one entry per chain or ligand. Put the type in the header,
otherwise nucleic acids are recognised by their letters:

```text
>A protein
MSTIEELIKAAKELKSLGIEN...
>B rna
GGACUUCGGUCC
>C dna
ATGCATGCAT
>L ligand
ATP                      (a CCD code, or "smiles:CC(=O)O")
```

* XPG searches with its own likelihood engine on the GPU, in every space group of the point group, with the most
  likely number of copies and its neighbours (`--copies N` to fix it). Translation peaks are ranked by likelihood,
  and the final solution is rigid-body refined copy by copy. TFZ is the z-score of the last copy's likelihood
  against random positions; from 8 on it counts as solved (wrong models stayed below 6 in our tests).
* Pathologies of the data are taken into account. MR runs the same tNCS and twinning analyses as the processing on
  each data set it searches:
  * **tNCS**: the copies are placed as pairs related by the tNCS translation, and the rotation function is corrected
    for the modulation it causes.
  * **Twinning**: when the L-test confirms it, the LLG compares the model through its twinned intensities, and the
    twin fraction is refined against the placed model.
  * **Anisotropy and outliers**: anisotropy is removed from the targets, and reflections that are implausible under
    the Wilson distribution are left out.
* Without `--model`, the whole FASTA content is predicted as one complex and the best of five models is used, trimmed
  to the confident parts. The alignment comes from the public ColabFold server, so **your sequence is sent there**.
  Use `--no-msa` to keep it local (weaker models for natural proteins).
* With `--model`, nothing is predicted and the content is read from the model.
* XPG doesn't refine the solution. It reports R and R-free of the placed model (overall scale, B and a flat bulk
  solvent fitted to the working set), which is enough to tell solutions apart. Take the placed model into your
  refinement.
* After a close point-group call both point groups are searched. The higher-symmetry solution is expanded into the
  lower symmetry and both are scored against the same data. The higher symmetry wins unless the lower one has an
  R-free at least 0.01 lower.
* The placed models go to `run1/mr/`, the results to `run1/xpg_mr.json` and the PDF.

## 11. Things to keep in mind

* Check the geometry the first time you use a beamline: a wrong axis or beam centre is the usual reason indexing
  fails.
* On a laptop, process one large data set at a time.
* A close call isn't an error. Compare both outputs in refinement or MR before you choose.
* GPU runs aren't bit-reproducible. Don't read much into differences below 0.01 in R-free on a single data set.
* With several lattices, check in the summary which merge is the main output.
* For anomalous data set `friedels_law = false`.
* If you impose a space group or cell, XPG uses it as given: check the indexing and merging statistics.

## 12. Conventions and defaults

The pixel centre of array element `ix` is at `x = ix + 1`, frame `i` covers `z` in `[i-1, i)`, and rotations are
right-handed.

| Parameter | Value |
|---|---|
| strong / background pixel | 3 / 2 sigma |
| minimum spot size, max centroid offset | 6 pixels, 3 px |
| shoebox | +-5 sigma_D, +-4 sigma_M |
| summation foreground | 3 sigma (pixel size included) |
| smallest observed profile part, smallest zeta | 75%, 0.05 |
| lattice tolerance, symmetry I/sigma limit | 2 deg, 6 |
| outlier rejection, error model | 6 sigma; SdFac, SdB, SdAdd |
| scale knots, absorption | 5 deg (scale), 20 deg (B), harmonics up to l = 6 |
| anisotropic cut-off | local I/sigma >= 1.2 |
| ΔCC1/2 outliers | < 0 or < mean - 3 sd (Assmann et al. 2016) |
| tNCS | Patterson peak > 20% of the origin, > 15 Å |

## 13. Code map

| Module | Role |
|---|---|
| `cli.py`, `inputs.py`, `config.py`, `setup.py` | command line, input files, settings, header geometry |
| `pipeline.py` | runs the stages, checkpoints, decisions |
| `stage_index.py`, `stage_integrate.py`, `stage_scale.py`, `stage_filters.py`, `stage_lattices.py` | the stages |
| `detector.py`, `geometry.py`, `goniometer.py`, `models.py` | detector and diffraction geometry |
| `init.py`, `spotfind.py`, `spotcluster.py` | background, mask, spot finding |
| `indexing.py`, `lattice.py`, `refine.py` | indexing, lattices, refinement |
| `integrate.py`, `integrate_kernels.py`, `shoebox.py`, `forward/` | integration and the forward model |
| `scaling.py`, `symmetry.py`, `symops.py`, `merging.py`, `analysis.py`, `aniso.py`, `ice.py`, `bayes_merge.py` | scaling, symmetry, statistics, twinning, anisotropy |
| `outputs.py`, `report.py`, `pdf_report.py`, `deposit.py`, `io/` | output files and reports |
| `multiset.py` | `xpg merge` |
| `mr.py`, `mrsearch.py`, `llg.py`, `matthews.py`, `predict.py` | molecular replacement |
| `serve.py`, `events.py` | dashboard |
| `backend.py`, `tnp.py` | devices and array helpers |
| `simulate.py` | simulated data for the tests |

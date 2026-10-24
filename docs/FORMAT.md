# XPG file formats

XPG reads one input file, `xpg.toml`, and `merge.toml` for merging. It writes standard MTZ files and a few JSON
files of its own. All XPG formats carry a version number, so we can add to them without breaking your old files.

## Versions and adding to the formats

* `xpg.toml` and `merge.toml` start with `format = 1`. You can leave it out: a file without it is read as version 1.
  The JSON outputs carry it as `"format": "xpg-geometry/1"`, `"xpg-summary/1"` and `"xpg-mr/1"`.
* New keys don't change the version. The version only goes up when an existing key changes its meaning or unit.
* XPG reads every older version. A newer file is refused with a message to update XPG, so it is never misread.
* Input files are strict: a key XPG doesn't know is an error, so a typo can't be silently ignored.
* Output files are open: the readers skip keys they don't know, so you can add your own keys to a JSON file and XPG
  still reads it.

**Adding a setting.** Add a field with its default to `Params` in `xpg/config.py`. It is then a `[processing]` key
under the same name, it is written to `xpg.toml` whenever it differs from the default, and the dashboard rerun form
takes it. If it describes the experiment rather than the processing, give it a key in one of the other sections in
`SECTIONS` in `xpg/inputs.py`. A new section also goes into `SECTIONS`.

## `xpg.toml`

The input of a run: the experiment, and the processing settings that differ from the defaults. `xpg --images`
writes it from the image headers, and every run keeps a copy in its output directory. Edit that copy to rerun with
other settings.

```toml
format = 1

[data]
images = "/data/x_master.h5"
frames = [1, 3600]
exclude_frames = [[1201, 1260]]
oscillation = 0.1

[beam]
wavelength = 0.9763

[goniometer]
rotation_axis = [1.0, 0.0, 0.0]

[detector]
type = "EIGER"
nx = 4148
ny = 4362
pixel_x = 0.075
pixel_y = 0.075
beam_x = 2071.3
beam_y = 2190.8
distance = 180.2

[crystal]
space_group = 19
unit_cell = [36.4, 44.2, 124.9, 90, 90, 90]
friedels_law = false

[processing]
resolution = [50.0, 1.6]
prune_image_groups = false
```

Everything is optional except `data.images`. Lengths are in mm on the detector and in Å in the crystal, angles in
degrees and detector positions in pixels. Vectors are in the laboratory frame and don't need to be unit length.

### `[data]`

| Key | Meaning |
|---|---|
| `images` | image template with `????` for the frame number (`x_????.cbf`), or an HDF5 master file |
| `frames` | `[first, last]` frame to process |
| `exclude_frames` | `[[a, b], ...]` frame ranges to leave out |
| `spot_frames` | `[[a, b], ...]` frames searched for spots (default: all) |
| `background_frames` | `[a, b]` frames for the background model |
| `starting_frame`, `starting_angle` | the rotation angle (deg) at the start of this frame |
| `oscillation` | rotation per frame (deg) |

### `[beam]`, `[goniometer]`

| Key | Meaning |
|---|---|
| `wavelength` | Å |
| `direction` | direction of the incident beam |
| `polarization_fraction` | fraction of the beam polarised along the normal below (0.5 = unpolarised, ~0.99 at a synchrotron) |
| `polarization_normal` | normal of the polarisation plane |
| `rotation_axis` | rotation axis, positive for a right-handed rotation |

### `[detector]`

| Key | Meaning |
|---|---|
| `type` | detector name (`EIGER`, `PILATUS`, ...) |
| `nx`, `ny` | pixels along x and y |
| `pixel_x`, `pixel_y` | pixel size (mm) |
| `beam_x`, `beam_y` | point on the detector closest to the crystal (pixels); the beam position when the beam is normal to the detector |
| `distance` | crystal to detector (mm) |
| `detector_x_axis`, `detector_y_axis` | directions of the detector x and y axes |
| `sensor_material`, `sensor_thickness` | `Si`, `CdTe` or `GaAs`; thickness in mm (for the parallax correction) |
| `sensor_mu` | attenuation of the sensor (1/mm); computed from the material and wavelength when missing |
| `overload` | counts at which a pixel is saturated |
| `min_valid_pixel` | lower values mark a pixel as untrusted |
| `offset` | constant pedestal subtracted from every pixel |
| `gain` | counts per photon (estimated when missing) |
| `dark_current_image` | image of the non-X-ray background, subtracted |
| `trusted_region` | `[inner, outer]` radius of the usable area, relative to half the detector width |
| `trusted_values` | `[low, high]`: pixels whose background is below low / 10000 of the expected level are shadowed (beam stop) |
| `untrusted_rectangles` | `[[x1, x2, y1, y2], ...]` areas to mask |
| `untrusted_ellipses` | `[[x1, x2, y1, y2], ...]` ellipses inside these rectangles to mask |
| `untrusted_quadrilaterals` | `[[x1, y1, x2, y2, x3, y3, x4, y4], ...]` areas to mask |
| `x_distortion`, `y_distortion` | detector distortion tables (CBF images, 1/100 pixel) |
| `two_theta` | detector swing (deg), for information: it is already in the detector axes |

### `[crystal]`

| Key | Meaning |
|---|---|
| `space_group` | space group number to impose |
| `unit_cell` | `[a, b, c, alpha, beta, gamma]` to impose (Å, deg) |
| `friedels_law` | `false` for anomalous data: Friedel mates are kept apart |
| `reindex` | 3x4 reindexing matrix applied to the result |
| `reference_data` | MTZ to match the indexing and scale to |
| `free_r_reference` | MTZ whose free set is copied and extended |
| `max_cell` | longest cell edge to expect when indexing (Å) |

### `[processing]`

`stages`, `resolution = [low, high]` (Å), `exclude_resolution = [[low, high], ...]` (Å), and which parameters are
refined: `refine_index`, `refine_integrate` and `refine_scale` take lists of `BEAM`, `AXIS`, `ORIENTATION`, `CELL` and
`POSITION`. On top of these, every processing parameter (default in brackets):

| Parameter | Meaning |
|---|---|
| `strong_pixel` [3.0] | a strong pixel is this many sigma above the background |
| `background_pixel` [2.0] | pixels below this many sigma count as background |
| `min_pixels_per_spot` [6] | smallest spot |
| `spot_maximum_centroid` [3.0] | largest offset between a spot's peak and its centroid (pixels) |
| `cluster_radius` [2] | strong pixels this close belong to one spot (pixels) |
| `min_fraction_indexed` [0.5] | fraction of spots a solution must index |
| `lattice_tolerance` [2.0] | metric tolerance of the lattice symmetry (deg) |
| `extra_lattices` [true] | index and integrate further lattices |
| `min_lattice_separation` [0.5] | lattices closer than this are integrated as one (deg) |
| `beam_divergence`, `reflecting_range` [estimated] | spot width perpendicular to and along the rotation (deg, e.s.d.) |
| `delphi` [5.0] | rotation range of one geometry block in integration (deg) |
| `minpk` [75.0] | smallest observed part of a profile for a reflection to be kept (%) |
| `minimum_zeta` [0.05] | reflections closer to the rotation axis are left out |
| `background_template` [true] | background as a scaled template (rings, solvent) instead of a flat plane |
| `combine_intensities` [true] | combine profile-fitted and summation intensities |
| `smooth_orientation` [true] | smooth the crystal orientation from block to block |
| `ice_validate` [true] | exclude an ice ring only if its data are measurably worse |
| `isig_limit_symmetry` [6.0] | I/sigma of the reflections used to find the symmetry |
| `corrections` [DECAY, MODULATION, ABSORP] | scaling corrections |
| `strict_absorption` [auto] | keep Friedel mates apart in the absorption correction (auto: when `friedels_law = false`) |
| `minimum_i_sigma` [3.0] | I/sigma of the reflections used for post-refinement |
| `snrc` [0.0] | I/sigma of the reflections used to fit the corrections |
| `wfac1` [1.0] | outlier threshold factor |
| `nbatch` [auto] | number of scale batches (auto: one per 5 deg) |
| `fixed_scale_range` [none] | `[first, last, factor]` frames whose scale is fixed |
| `prune_image_groups` [true] | remove image ranges that lower CC1/2 |
| `prune_chunks` [true] | remove 5% chunks of the sweep that lower the high-resolution CC1/2 |
| `prune_fitness_threshold` [0.3] | remove stretches with I/sigma below this fraction of the median (0 = off) |
| `zero_dose` [false] | extrapolate every reflection to zero dose |
| `export_legacy` [false] | also write legacy-format text files for older programs |

## `merge.toml`

Input of `xpg merge`: several unmerged data sets, scaled and merged together.

```toml
format = 1

[merge]
output = "merged"                    # merged_unmerged.mtz, merged_merged_iso.mtz, ...
friedels_law = true
reject_nonisomorphous = false        # drop data sets that don't fit the rest

[[merge.input]]
file = "run1/xpg_unmerged.mtz"
reference = true                     # match the indexing to this one
crystal = "xtal1"                    # data sets from one crystal share their scales
[[merge.input]]
file = "run2/xpg_unmerged.mtz"
resolution = [50.0, 2.0]
```

| Key | Meaning |
|---|---|
| `output` | stem of the output files |
| `space_group`, `unit_cell` | impose them (default: from the inputs) |
| `friedels_law`, `corrections`, `nbatch`, `snrc`, `wfac1` | as in `xpg.toml` |
| `zero_dose` | significance level of the zero-dose extrapolation (0 = off) |
| `resolution_shells` | upper limits of the statistics shells (Å) |
| `reject_nonisomorphous` | drop data sets with negative or outlying ΔCC1/2 |
| `input.file` | unmerged MTZ from XPG (with its geometry JSON next to it) |
| `input.reference`, `input.crystal`, `input.resolution` | see the example |
| `input.weight` | factor on the sigmas of this data set |
| `input.reindex` | 3x4 reindexing matrix for this data set |
| `input.starting_dose`, `input.dose_rate` | dose of the first frame and per frame, for the zero-dose extrapolation |

## Output files

### MTZ

| File | Columns |
|---|---|
| `xpg_unmerged.mtz` | `M/ISYM`, `BATCH` (frame), `I`, `SIGI`, `XDET`, `YDET` (pixels), `ROT` (deg), `FRACTIONCALC`, `LP` |
| `xpg_merged_iso.mtz` | `IMEAN`, `SIGIMEAN`, `I(+)`, `SIGI(+)`, `I(-)`, `SIGI(-)`, `F`, `SIGF`, `F(+)`, `SIGF(+)`, `F(-)`, `SIGF(-)`, `DANO`, `SIGDANO`, `FreeR_flag` |
| `xpg_anisotropic.mtz` | the same, inside the anisotropic cut-off surface, plus `FANISO`, `SIGFANISO`, `IPOST`, `SIGIPOST` |

Amplitudes are French-Wilson estimates. `FreeR_flag` runs from 0 to 19 and 0 is the free set. The flags come from a
hash of the reflection's index under the lattice symmetry, so they are the same in every run of the same crystal and
twin-related reflections share their flag.

### `xpg_geometry.json`

The refined experiment, written next to every unmerged MTZ so that the MTZ is all you need to scale and merge again.

```json
{"format": "xpg-geometry/1",
 "wavelength": 0.9763, "beam": [0, 0, 1], "rotation_axis": [1, 0, 0],
 "detector": {"orgx": 2071.3, "orgy": 2190.8, "distance": 180.2, "x_axis": [1, 0, 0], "y_axis": [0, 1, 0],
              "pixel_x": 0.075, "pixel_y": 0.075, "nx": 4148, "ny": 4362, "sensor_thickness": 0.45, "mu": 3.9},
 "polarization": {"fraction": 0.99, "normal": [0, 1, 0]},
 "scan": {"phi_start": 0.0, "osc": 0.1, "starting_frame": 1, "frames": [1, 3600]},
 "crystal": {"space_group_number": 19, "cell": [36.4, 44.2, 124.9, 90, 90, 90],
             "a_star": [[...], [...], [...]], "system": "orthorhombic"},
 "friedel_law": false, "merged": false}
```

`a_star` holds the reciprocal axes a\*, b\*, c\* (1/Å) as the columns of a 3x3 matrix at the rotation angle zero.

### Other files

| File | Content |
|---|---|
| `xpg_summary.json` | space group and alternatives, cell, ISa, resolution limit, anisotropy, error model, the merging statistics (`statistics`: overall, inner and outer shell to the CC1/2 limit; `statistics_aniso`: the same inside the anisotropic surface, with the ellipsoidal completeness) and `pathologies` (`tncs`: Patterson peak ratio, vector, distance; `twinning`: L-test and every twin law with its H-test and Britton fractions; `hidden_twin`) |
| `xpg_mr.json` | one entry per data set searched (space group, copies, tNCS pairs, TFZ, LLG, R and R-free of the placed model, the `pathologies` MR found and used, the twin fraction refined against the model) and `choice`, the data set and space group picked |
| `events.jsonl` | one JSON object per line: `t` (time), `type` (`run_start`, `stage_start`, `stage_end`, `log`, `metric`, `decision`, `flag`, `run_end`) and `stage`; the dashboard and the PDF read it |
| `xpg_spots.txt` | strong spots: `x y frame intensity` and, once indexed, `h k l` |
| `xpg_integrated.npz` | integrated reflections before scaling (NumPy archive) |

## Files from other programs

XPG also reads XDS.INP and XSCALE.INP. A run started from an XDS.INP keeps the converted `xpg.toml`, and
`xpg convert XDS.INP` writes it without running anything. Keywords without a meaning in XPG (number of jobs, profile
grid sizes, ...) are dropped. Our files hold different things:

* `xpg.toml` describes the experiment and only the settings you changed. Things XPG works out itself (spot widths,
  error model, resolution limit, lattices) aren't settings.
* Its sections and keys say what they are (`[detector] beam_x` instead of `ORGX=`), every key has one meaning, and
  unknown keys are refused instead of ignored.
* The outputs are MTZ and JSON instead of fixed-column text files. The unmerged MTZ comes with its geometry, so it is
  self-contained, and the decisions XPG took are in `xpg_summary.json`.
* The legacy text files are still written with `--export-legacy` if another program needs them.

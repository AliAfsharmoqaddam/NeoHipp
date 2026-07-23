# NeoHipp

Surface-based analysis of neocortical grey–white matter contrast ("blurring") in patients with hippocampal sclerosis (HS), and its relationship to epilepsy history, hippocampal volume asymmetry, and developmental/connectional maps of the cortex.

The pipeline builds on the [MELD classifier](https://github.com/MELDProject/meld_classifier) framework: T1 intensities are sampled across cortical depth onto a symmetric surface template, harmonised across sites, normalised, and reduced to a grey–white contrast asymmetry feature. This feature is then used to characterise the temporopolar cortex in HS and to test what shapes its spatial pattern.

> ⚠️ **Paths and data.** Raw patient MRI and the MELD-format HDF5 feature matrices are **not** included in this repository, and all scripts still contain absolute paths from the development environment (e.g. `/home/meldstudent/...`). The `data/` directory ships the **de-identified, group-level and subject-level derivatives** needed to regenerate the statistics and most figure panels without access to the imaging. See [Shipped data](#shipped-data) and [Reproducing figures from shipped data](#reproducing-figures-from-shipped-data).

---

## Repository structure

```
NeoHipp/
├── data/
│   ├── patient_derived/         # de-identified derivatives of the HS cohort
│   │   ├── subject_measures.csv
│   │   ├── depth_profiles.csv
│   │   ├── group_wg_contrast_ttest.lh.func.gii
│   │   └── patient_avg_wm_blurring.lh.func.gii
│   └── reference_maps/          # external atlases, templates and normative maps
│       ├── lh.HCP-MMP1_sym.annot
│       ├── lh.white / lh.sphere / lh.sphere.reg
│       ├── myelin_dev_months.lh.func.gii
│       ├── geo_dist_hippo.lh.func.gii
│       ├── gene_expression_hippo.lh.func.gii
│       └── hippo_fmri_connectivity_L.func.gii
├── preprocessing/               # MELD-style surface feature pipeline
│   ├── T1_sampling.py               # sample T1 onto surfaces at multiple depths
│   ├── xhemi_register.py            # register hemispheres to fsaverage_sym
│   ├── hdf5.py                      # write sampled features to HDF5 feature matrices
│   ├── clipping.py                  # compute cohort clipping (outlier) parameters
│   ├── smoothing.py                 # surface smoothing (FWHM 5 mm)
│   ├── combat_harmonise.py          # ComBat harmonisation across sites
│   └── normalisation.py             # intra/inter-subject z-scoring + asymmetry
└── figure_generation/
    ├── FIG1/                    # group-level cortical contrast maps & distributions (Manuscript Fig. 2)
    │   ├── fig_1a.py
    │   ├── fig_1b.py
    │   ├── fig_1c.py
    │   └── fig_1d+supp.py
    ├── FIG2/
    │   └── fig_2.py             # onset/duration, correlation matrix, group comparison (Manuscript Fig. 3)
    ├── FIG3/
    │   └── fig_3.py             # spatial contextualisation (dominance + spin test) (Manuscript Fig. 4)
    └── SUPP/
        └── supp_fig_2.py        # supplementary surface panels
```

---

## Shipped data

Everything under `data/` is either an external reference map or a de-identified derivative. No raw imaging, no subject identifiers, and no site labels are included.

### Common conventions

| | |
|---|---|
| **Surface template** | `fsaverage_sym` (FreeSurfer symmetric template), left hemisphere, ico7 |
| **Vertices** | 163,842 per surface file; 146,902 cortical vertices + 16,940 medial-wall vertices |
| **Medial wall** | Filled with `0` in every map except the p-value array, where it is filled with `1` |
| **Hemisphere convention** | All maps are `on_lh`: each patient's **ipsilateral** hemisphere is mapped onto the left-hemisphere template, so "left" here means ipsilateral to the sclerotic hippocampus, not anatomically left |
| **Feature values** | Unless stated otherwise, values are **asymmetry z-scores** — ipsilateral minus contralateral, ComBat-harmonised, intra- then inter-subject normalised. Negative = reduced value ipsilaterally |
| **Temporopolar ROI** | Glasser/HCP-MMP1 labels **131** (`L_TGd_ROI`, 1,713 vertices) and **172** (`L_TGv_ROI`, 634 vertices); 2,347 vertices combined |

---

## Data dictionary

### `data/patient_derived/subject_measures.csv`

One row per participant; **272 rows** (154 HS patients, 118 controls) × 10 columns. Used by the FIG1c, FIG2 and blurred-patient-selection steps.

| Column | Type | Description |
|---|---|---|
| `id` | int | Anonymised participant index, 1–272. Not a MELD ID; there is no key back to the original identifiers. Matches `id` in `depth_profiles.csv` row-for-row |
| `group` | str | `patient` (histopathologically confirmed HS, n = 154) or `control` (n = 118) |
| `sex` | int | **0 = male, 1 = female** |
| `age` | float | Age in years at the preoperative scan. Range 3.0–64.7 |
| `tpblur_score` | float | **Temporopolar blurring score** — mean grey–white contrast asymmetry z-score across the temporopolar ROI (TGd + TGv). More negative = greater blurring. Range −3.68 to +2.09 |
| `hipp_vol_asym` | float | Ipsilateral hippocampal volume asymmetry, ICV-corrected, ComBat-harmonised and inter-subject z-scored (controls: mean ≈ 0, SD ≈ 1). More negative = greater ipsilateral atrophy |
| `hemi` | str | Hemisphere ipsilateral to the HS: `lh` or `rh`. **Empty for controls** (see below) |
| `onset` | float | Age of epilepsy onset in years. **Patients only** |
| `duration` | float | Epilepsy duration in years at the time of the scan. **Patients only** |
| `histo` | str | `HS only` (n = 80) or `FCD IIIa` (n = 74) — concurrent cortical dysplasia confirmed on histopathology. **Empty for controls** |

**Missing-value semantics.** Blank cells are not all the same thing:

- `hemi`, `onset`, `duration`, `histo` are blank for **all 118 controls** by design — controls have no ipsilateral side, no epilepsy history and no histopathology. In `fig_1a.py`/`fig_1c.py` the control hemisphere is instead assigned at random with `random.seed(1)`.
- `onset` and `duration` are additionally blank for **16 patients** whose clinical history was unavailable, leaving n = 138 for the onset and duration correlations.
- `hipp_vol_asym` is blank for **6 participants** (5 patients, 1 control) who failed hippocampal segmentation QC, leaving n = 266.

Rows are **not** dropped for missing values; select per analysis, e.g.:

```python
import pandas as pd
d = pd.read_csv("data/patient_derived/subject_measures.csv")

patients  = d[d.group == "patient"]
controls  = d[d.group == "control"]
onset_set = patients.dropna(subset=["onset", "hipp_vol_asym"])   # correlation matrix, n = 133
blurred   = patients[patients.tpblur_score < -1]                 # thresholded blurring subgroup
```

### `data/patient_derived/depth_profiles.csv`

One row per participant; **272 rows** × 49 columns (`id` + 48 depth features). Row order and `id` values are identical to `subject_measures.csv`, so the two can be merged on `id`.

Each cell is the **mean asymmetry z-score across the temporopolar ROI** at one sampling depth (5 mm FWHM smoothing, ComBat-harmonised, intra/inter-subject normalised). This is the source for the depth-profile panel (manuscript Fig. 2D) and the supplementary pial-referenced replication (Supplementary Fig. 1).

#### Depth nomenclature

Column names follow the grammar `<tissue>_<depth>_<reference surface>`, where the suffix identifies which surface the depth is measured **from**:

| Suffix | Reference surface | FreeSurfer sampling |
|---|---|---|
| `_ws` | **w**hite **s**urface (the grey–white junction) | `mri_vol2surf --surf white` |
| `_ps` | **p**ial **s**urface | `mri_vol2surf --surf pial` |

There are three blocks of columns:

| Block | Pattern | Depth units | n | Sampling |
|---|---|---|---|---|
| Intracortical, fractional | `gm_<f>_ws` | Unitless fraction of cortical thickness, 0 → 1 in steps of 0.1 | 11 | `--projfrac <f> --surf white` |
| Subcortical, absolute | `wm_-<d>mm_ws` | mm below the white surface, 0.5 → 4 mm in 0.5 mm steps | 8 | `--projdist -<d> --surf white` |
| Pial-referenced, absolute | `gm_<d>mm_ps` | mm below the pial surface, 0 → 7 mm in 0.25 mm steps | 29 | `--projdist -<d> --surf pial` |

Worked examples:

- `gm_1_ws` — 100 % of cortical thickness above the white surface, i.e. **at the pial surface**
- `gm_0.3_ws` — 30 % of cortical thickness above the white surface (the grey-matter sample used in the blurring score)
- `gm_0_ws` — 0 % above the white surface, i.e. **at the grey–white junction**
- `wm_-0.5mm_ws` — 0.5 mm **below** the white surface, in superficial white matter
- `wm_-1mm_ws` — 1 mm below the white surface (the depth at which the group difference peaks, and the white-matter sample used in the blurring score)
- `gm_0mm_ps` — **at the pial surface**
- `gm_0.25mm_ps` — 0.25 mm **below** the pial surface
- `gm_7mm_ps` — 7 mm below the pial surface (≈ 3 mm of cortex + 4 mm of white matter)

Three naming quirks worth knowing before you parse these programmatically:

1. **The minus sign is inconsistent.** The `_ws` white-matter columns carry an explicit minus (`wm_-1.5mm_ws`) because that is how the underlying MELD feature strings are written; the `_ps` columns do not (`gm_1.5mm_ps`). In both cases the number denotes **depth below** the reference surface. Only the fractional `gm_<f>_ws` columns count *upward*, away from the white surface toward the pial.
2. **The `gm_` prefix on the `_ps` block is not a tissue claim.** All 29 pial-referenced columns are prefixed `gm_` because they were generated by a single `--surf pial` sampling loop. Depths beyond roughly 3 mm below the pial surface lie in white matter in most cortex.
3. **`gm_0_ws` and `gm_0mm_ps` are different depths** — the former sits at the grey–white junction, the latter at the pial surface.

The two ladders answer different questions and deliberately overlap. The `_ws` ladder is thickness-normalised, so equivalent cortical depths are compared across regions and participants whose thickness differs. The `_ps` ladder is measured in absolute millimetres from the pial surface, which does not depend on white-surface placement — important because the white boundary is itself hard to position reliably where blurring is severe. The two give the same answer, which is the point of the supplementary panel.

Full column order:

```
id,
gm_1_ws, gm_0.9_ws, gm_0.8_ws, gm_0.7_ws, gm_0.6_ws, gm_0.5_ws,
gm_0.4_ws, gm_0.3_ws, gm_0.2_ws, gm_0.1_ws, gm_0_ws,
wm_-0.5mm_ws, wm_-1mm_ws, wm_-1.5mm_ws, wm_-2mm_ws,
wm_-2.5mm_ws, wm_-3mm_ws, wm_-3.5mm_ws, wm_-4mm_ws,
gm_0mm_ps, gm_0.25mm_ps, gm_0.5mm_ps, ... (0.25 mm steps) ..., gm_7mm_ps
```

Selecting a block:

```python
import pandas as pd
p = pd.read_csv("data/patient_derived/depth_profiles.csv")

frac_cols = [c for c in p.columns if c.endswith("_ws") and c.startswith("gm_")]
wm_cols   = [c for c in p.columns if c.startswith("wm_")]
pial_cols = [c for c in p.columns if c.endswith("_ps")]

# numeric depth from a column name (always returned as depth below the reference surface)
def depth(col):
    body = col.rsplit("_", 1)[0].split("_", 1)[1]      # e.g. '-1mm', '0.3', '2.25mm'
    return abs(float(body.replace("mm", "")))
```

### Surface maps (`.func.gii`)

All are single-hemisphere GIfTI files on `fsaverage_sym`, 163,842 vertices, `float32`. Load with `nibabel.load(path).darrays[i].data`.

| File | Arrays | Contents |
|---|---|---|
| `patient_derived/group_wg_contrast_ttest.lh.func.gii` | 2 | **[0]** Welch's *t* statistic for grey–white contrast asymmetry, controls vs HS patients, per vertex. Signed as `ttest_ind(controls, patients)`, so **positive *t* = reduced contrast in patients**; range −3.04 to 10.01. **[1]** Corresponding **Holm–Bonferroni-corrected** *p* values. Medial wall is `t = 0`, `p = 1`. Source for manuscript Fig. 2A |
| `patient_derived/patient_avg_wm_blurring.lh.func.gii` | 1 | Per-vertex mean across HS patients of the white-matter T1w intensity asymmetry sampled **1 mm below the white surface** (`wm_T1_-1mm`). This is the dependent variable in the dominance analysis. Range −0.74 to 0.41; manuscript Fig. 4A |
| `reference_maps/myelin_dev_months.lh.func.gii` | 1 | Age in **months** at which grey–white contrast (T1w/T2w) reaches 50 % of its maximum within the 0–24 month window, from the Infant Brain Atlas. Range 0–22.8 months; higher = later-myelinating. Manuscript Fig. 4B |
| `reference_maps/geo_dist_hippo.lh.func.gii` | 1 | Geodesic distance in **mm** along the white surface from the subiculum/parahippocampal junction, computed with the heat method. Range 0–157.6 mm. Manuscript Fig. 4C |
| `reference_maps/gene_expression_hippo.lh.func.gii` | 1 | Spearman ρ between each cortical location's AHBA microarray expression profile and the consensus hippocampal expression vector. Range −0.36 to 0.72. Manuscript Fig. 4D |
| `reference_maps/hippo_fmri_connectivity_L.func.gii` | 1 | Mean resting-state fMRI correlation between each cortical vertex and all hippocampal voxels (HCP 1200 group-average). Range −0.08 to 0.37. Manuscript Fig. 4E |

### Reference surfaces and atlas

| File | Contents |
|---|---|
| `lh.white` | `fsaverage_sym` white surface geometry — 163,842 vertices, 327,680 faces. Used for all surface rendering and for geodesic distance |
| `lh.sphere`, `lh.sphere.reg` | Spherical and registered-spherical `fsaverage_sym` surfaces, required for `mris_apply_reg` in `xhemi_register.py` and for the spin-permutation null in `fig_3.py` |
| `lh.HCP-MMP1_sym.annot` | Symmetric Glasser (HCP-MMP1) parcellation, 180 cortical regions + unknown. Read with `nibabel.freesurfer.read_annot`, which returns `(labels, ctab, names)`. Temporopolar ROI = labels **131** (`L_TGd_ROI`) and **172** (`L_TGv_ROI`) |

```python
import nibabel as nb
labels, ctab, names = nb.freesurfer.read_annot("data/reference_maps/lh.HCP-MMP1_sym.annot")
temp_pole = (labels == 131) | (labels == 172)      # 2,347 vertices
```

### Provenance and licensing of reference maps

These maps are derived from third-party resources; please cite the original sources if you reuse them.

| Map | Derived from |
|---|---|
| `lh.HCP-MMP1_sym.annot` | Glasser et al., *Nature* 2016 |
| `lh.white`, `lh.sphere`, `lh.sphere.reg` | FreeSurfer `fsaverage_sym` template |
| `myelin_dev_months.lh.func.gii` | Infant Brain Atlas (Ahmad et al., 2023) |
| `hippo_fmri_connectivity_L.func.gii` | HCP 1200 group-average connectivity (Van Essen et al., 2013; Glasser et al., 2016) |
| `gene_expression_hippo.lh.func.gii` | Allen Human Brain Atlas (Hawrylycz et al., 2015), processed with `abagen` (Markello et al., 2021) |

---

## Reproducing figures from shipped data

The figure scripts as committed still read the original HDF5 feature matrices and demographics CSV through `MeldCohort`/`MeldSubject`. The files in `data/` are the de-identified equivalents of those inputs; to run without the imaging, substitute as follows.

| Hardcoded path in scripts | Shipped equivalent |
|---|---|
| `/home/meldstudent/Downloads/lh.HCP-MMP1_sym.annot` | `data/reference_maps/lh.HCP-MMP1_sym.annot` |
| `.../fs_outputs/fsaverage_sym/surf/lh.white` | `data/reference_maps/lh.white` |
| `.../Files/myelin_dev_months_new.IBA.lh.func.gii` | `data/reference_maps/myelin_dev_months.lh.func.gii` |
| `.../Files/hippo_fmri_data_L.fsaverage_sym_correct.func.gii` | `data/reference_maps/hippo_fmri_connectivity_L.func.gii` |
| `.../Files/spearman_hippocamp_surface.lh.fsaverage_sym.func.gii` | `data/reference_maps/gene_expression_hippo.lh.func.gii` |
| `altered_info5_with_new_name.csv` + per-subject temporopolar means | `data/patient_derived/subject_measures.csv` |
| `matrix_norm_avg_240624.csv` (hippocampal volumes) | `hipp_vol_asym` column of `subject_measures.csv` |
| Per-subject depth feature loops via `MeldSubject.load_feature_values` | `data/patient_derived/depth_profiles.csv` |
| Vertex-wise `ttest_ind` over the cohort | `data/patient_derived/group_wg_contrast_ttest.lh.func.gii` |
| `patient_mean_full(wm_asym_string)` | `data/patient_derived/patient_avg_wm_blurring.lh.func.gii` |

What can and cannot be regenerated without the HDF5 matrices:

- **Fully reproducible** — the Fig. 2A surface *t*-map and its Holm-corrected mask; the Fig. 2C temporopolar distributions and Mann–Whitney test; the Fig. 2D and Supplementary Fig. 1 depth profiles; all of Fig. 3; the whole Fig. 4 dominance analysis and spin test.
- **Not reproducible** — anything requiring per-vertex, per-subject values (e.g. re-deriving the *t*-map from individual data, or recomputing the temporopolar ROI mean under a different parcellation). Requests for the extracted surface-based features can be made by contacting meld.study@gmail.com.

---

## Preprocessing pipeline

The scripts under `preprocessing/` are intended to be run in order. Each stage reads the HDF5 feature matrix written by the previous one (the `hdf5_file_root` string in each script), so the filenames must be kept consistent across stages.

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `T1_sampling.py` | Uses FreeSurfer `mri_vol2surf` to sample T1 intensity onto each subject's surface at 11 grey-matter fractions, 29 grey-matter distances (0 to −7 mm), and 9 white-matter distances (0 to −4 mm), for both hemispheres. |
| 2 | `xhemi_register.py` | Registers each subject's sampled features to the symmetric template `fsaverage_sym` with `mris_apply_reg`, so left and right hemispheres are comparable and asymmetry can be computed. |
| 3 | `hdf5.py` | Reads the per-subject `.mgh` feature maps and writes them into MELD-format HDF5 feature matrices, inferring group (patient/control) and scanner (1.5T/3T) from the subject ID. |
| 4 | `clipping.py` | Computes cohort-wide mean/SD per feature and saves clipping parameters used to limit outliers during smoothing. |
| 5 | `smoothing.py` | Applies surface smoothing with a 5 mm FWHM kernel (using the clipping parameters). |
| 6 | `combat_harmonise.py` | Runs ComBat harmonisation across acquisition sites using the saved ComBat parameters. |
| 7 | `normalisation.py` | Performs intra- and inter-subject normalisation and computes left-vs-right asymmetry for each feature. |

The analysis feature used downstream is the grey–white matter contrast (`w-g.pct`) after ComBat, intra/inter-subject normalisation, and asymmetry — this is the **"blurring score"** referred to in the figures. The depth columns in `depth_profiles.csv` are the temporopolar means of the step-1 sampling ladder carried through the same stages at 5 mm FWHM.

### Subject ID convention

Subject IDs in the pipeline follow the MELD pattern, e.g. `MELD_H16_3T_FCD_hs0001`:

- **Site** — `H1`, `H11`, `H16`, `H29`
- **Scanner** — `3T` or `15T`
- **Type** (last token) — `hs####` (HS patient), `fcd####` (FCD patient), `c####` (control)

Site `H16` retains its FCD cohort for harmonisation requirements; other sites are restricted to HS patients and controls.

These identifiers do **not** appear in the shipped CSVs, which use the anonymised integer `id` instead.

---

## Figure generation

Each figure script is standalone and loads the harmonised HDF5 feature matrices plus a demographics CSV and the symmetric Glasser (HCP-MMP1) atlas. Temporopolar regions are defined as Glasser labels TGd (131) and TGv (172). Figure numbering in the repository is offset by one from the manuscript: `FIG1` → manuscript Fig. 2, and so on.

**FIG1 — group-level cortical contrast**
- `fig_1a.py` — Vertex-wise Welch's *t*-test of grey–white contrast, HS patients vs controls, Holm–Bonferroni corrected, rendered on the cortical surface. Shipped output: `group_wg_contrast_ttest.lh.func.gii`.
- `fig_1b.py` — Reference rendering of the symmetric Glasser parcellation (grey-shaded regions).
- `fig_1c.py` — Distribution of temporopolar blurring with a Mann–Whitney U comparison and kernel-density overlay. Shipped input: `tpblur_score` in `subject_measures.csv`.
- `fig_1d+supp.py` — Depth-resolved Mann–Whitney comparisons with multiple-comparison correction (main + supplementary panels). Shipped input: `depth_profiles.csv`. *Runtime ≈ 8–10 min from HDF5; near-instant from the CSV.*

**FIG2 — clinical relationships** (`fig_2.py`)
A four-panel figure: (A) epilepsy onset vs blurring with a linear GAM fit, (B) epilepsy duration vs blurring, (C) a staircase Spearman correlation matrix (onset, duration, blurring, hippocampal volume asymmetry; Holm–Bonferroni-corrected), and (D) a violin/box comparison of FCD IIIa vs HS-only with a Mann–Whitney U test. All four panels draw on `subject_measures.csv`.

**FIG3 — spatial contextualisation** (`fig_3.py`)
Relates the patient-averaged white-matter blurring map to four cortical maps — resting-state fMRI hippocampal connectivity, myelination timing, gene expression, and geodesic distance from the hippocampus (heat-method distance via `potpourri3d`). It runs a dominance analysis (relative importance of each predictor) and a spin-permutation test on the full-model R² to account for spatial autocorrelation (`brainspace` SpinPermutations, 1000 spins). Outputs surface plots, `spin_test_results.csv`, `null_r2_distribution.csv`, and a null-distribution histogram. Every input map for this figure is shipped in `data/`.

**SUPP** (`supp_fig_2.py`)
Supplementary surface panels using the same atlas, template surface and cohort conventions.

---

## Requirements

**External neuroimaging tools**
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) (v7.2.0 used here) — required for `T1_sampling.py` and `xhemi_register.py`, and provides the `fsaverage_sym` template. Not needed if you work only from the shipped `data/` files.
- [MELD classifier](https://github.com/MELDProject/meld_classifier) — provides `meld_classifier` (cohort/subject objects, preprocessing helpers, paths). Not needed if you work only from the shipped `data/` files.
- [matplotlib_surface_plotting](https://github.com/kwagstyl/matplotlib_surface_plotting) — surface rendering.

**Python packages**

```
numpy
pandas
scipy
scikit-learn
statsmodels
matplotlib
seaborn
nibabel
h5py
pygam
brainspace
potpourri3d
dominance-analysis
```

A Python 3.8 environment is assumed (matching the MELD classifier environment). Install the pip-available packages with:

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn \
            nibabel h5py pygam brainspace potpourri3d dominance-analysis
```

`meld_classifier` and `matplotlib_surface_plotting` should be installed from their respective repositories, and FreeSurfer configured separately.

> **Note on NumPy versions.** `h5py` in this environment requires `numpy<1.24`; `numpy==1.23.5` is known to work. If you hit a binary-incompatibility error on import, pin NumPy and restart the interpreter.

---

## Usage

### Working from the shipped data (no imaging required)

```python
import pandas as pd, nibabel as nb

subj  = pd.read_csv("data/patient_derived/subject_measures.csv")
depth = pd.read_csv("data/patient_derived/depth_profiles.csv")
df    = subj.merge(depth, on="id")

tmap, pmap = [d.data for d in
              nb.load("data/patient_derived/group_wg_contrast_ttest.lh.func.gii").darrays]
sig = pmap < 0.05                      # Holm-corrected
```

### Running the full pipeline

1. Install FreeSurfer, the MELD classifier, and the Python dependencies above.
2. Edit the hardcoded paths at the top of each script to point at your own data, atlas, surface template, and HDF5 feature matrices — or at the shipped equivalents in the substitution table above.
3. Run the preprocessing scripts in order (steps 1–7). Keep the `hdf5_file_root` filenames consistent so each stage reads the previous stage's output.
4. Run any figure script directly, e.g.:

```bash
python figure_generation/FIG2/fig_2.py
```

Figures are written to the working directory (e.g. `fig_2.py` saves `relationshipspanel.png`).

---

## Notes

- The pipeline scripts assume MELD's directory layout (`FS_SUBJECTS_PATH`, `BASE_PATH`, etc., defined in `meld_classifier.paths`). The shipped `data/` files do not.
- Filtering to HS patients is applied both by passing a dataset CSV to `MeldCohort` and, in several scripts, by keeping IDs containing `hs` — the latter is redundant when the dataset is passed but is left in as a safeguard.
- Control hemispheres are assigned at random with `random.seed(1)`; keep the seed to reproduce published values exactly.
- Results outputs (surface PNGs, CSVs) are written to the current working directory unless a full path is given.

---

## License

Code is released under the MIT License. See [`LICENSE`](LICENSE). Reference maps under `data/reference_maps/` remain subject to the licence terms of their original sources (see [Provenance](#provenance-and-licensing-of-reference-maps)).

## Contact

For questions about the analysis, please open an issue on this repository. Requests for access to the extracted surface-based features can be made by contacting meld.study@gmail.com.

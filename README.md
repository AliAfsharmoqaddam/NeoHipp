# NeoHipp

Surface-based analysis of neocortical grey–white matter contrast ("blurring") in patients with hippocampal sclerosis (HS), and its relationship to epilepsy history, hippocampal volume asymmetry, and developmental/connectional maps of the cortex.

The pipeline builds on the [MELD classifier](https://github.com/MELDProject/meld_classifier) framework: T1 intensities are sampled across cortical depth onto a symmetric surface template, harmonised across sites, normalised, and reduced to a grey–white contrast asymmetry feature. This feature is then used to characterise the temporopolar cortex in HS and to test what shapes its spatial pattern.

> ⚠️ **Paths and data.** All scripts contain absolute paths from the development environment (e.g. `/home/meldstudent/...`) and expect MELD-format HDF5 feature matrices plus a demographics CSV. Patient data is **not** included in this repository. You will need to edit the paths and provide your own inputs to reproduce the pipeline.

Extended notes: https://app.notion.com/p/G-W-contrast-395d07049f58805ea32ff7763bea886a

---

## Repository structure

```
NeoHipp/
├── preprocessing/          # MELD-style surface feature pipeline
│   ├── T1_sampling.py          # sample T1 onto surfaces at multiple depths
│   ├── xhemi_register.py       # register hemispheres to fsaverage_sym
│   ├── hdf5.py                 # write sampled features to HDF5 feature matrices
│   ├── clipping.py             # compute cohort clipping (outlier) parameters
│   ├── smoothing.py            # surface smoothing (FWHM 5 mm)
│   ├── combat_harmonise.py     # ComBat harmonisation across sites
│   └── normalisation.py        # intra/inter-subject z-scoring + asymmetry
└── figure_generation/
    ├── FIG1/               # group-level cortical contrast maps & distributions
    │   ├── fig_1a.py
    │   ├── fig_1b.py
    │   ├── fig_1c.py
    │   └── fig_1d+supp.py
    ├── FIG2/
    │   └── fig_2.py        # onset/duration, correlation matrix, group comparison
    └── FIG3/
        └── fig_3.py        # spatial contextualisation (dominance + spin test)
```

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

The analysis feature used downstream is the grey–white matter contrast (`w-g.pct`) after ComBat, intra/inter-subject normalisation, and asymmetry — this is the **"blurring score"** referred to in the figures.

### Subject ID convention

Subject IDs follow the MELD pattern, e.g. `MELD_H16_3T_FCD_hs0001`:

- **Site** — `H1`, `H11`, `H16`, `H29`
- **Scanner** — `3T` or `15T`
- **Type** (last token) — `hs####` (HS patient), `fcd####` (FCD patient), `c####` (control)

Site `H16` retains its FCD cohort for harmonisation requirements; other sites are restricted to HS patients and controls.

---

## Figure generation

Each figure script is standalone and loads the harmonised HDF5 feature matrices plus a demographics CSV and the symmetric Glasser (HCP-MMP1) atlas. Temporopolar regions are defined as Glasser labels TGd (131) and TGv (172).

**FIG1 — group-level cortical contrast**
- `fig_1a.py` — Vertex-wise Welch's t-test of grey–white contrast, HS patients vs controls, Holm–Bonferroni corrected, rendered on the cortical surface.
- `fig_1b.py` — Reference rendering of the symmetric Glasser parcellation (grey-shaded regions).
- `fig_1c.py` — Distribution of temporopolar blurring with a Mann–Whitney U comparison and kernel-density overlay.
- `fig_1d+supp.py` — Regional Mann–Whitney comparisons with multiple-comparison correction (main + supplementary panels). *Runtime ≈ 8–10 min.*

**FIG2 — clinical relationships** (`fig_2.py`)
A four-panel figure: (A) epilepsy onset vs blurring with a linear GAM fit, (B) epilepsy duration vs blurring, (C) a staircase Spearman correlation matrix (onset, duration, blurring, hippocampal volume asymmetry; Bonferroni-corrected), and (D) a violin/box comparison of FCD IIIa vs HS-only with a Mann–Whitney U test.

**FIG3 — spatial contextualisation** (`fig_3.py`)
Relates the patient-averaged white-matter blurring map to four cortical maps — resting-state fMRI hippocampal connectivity, myelination timing, gene expression, and geodesic distance from the hippocampus (heat-method distance via `potpourri3d`). It runs a dominance analysis (relative importance of each predictor) and a spin-permutation test on the full-model R² to account for spatial autocorrelation (`brainspace` SpinPermutations, 1000 spins). Outputs surface plots, `spin_test_results.csv`, `null_r2_distribution.csv`, and a null-distribution histogram.

---

## Requirements

**External neuroimaging tools**
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) (v7.2.0 used here) — required for `T1_sampling.py` and `xhemi_register.py`, and provides the `fsaverage_sym` template.
- [MELD classifier](https://github.com/MELDProject/meld_classifier) — provides `meld_classifier` (cohort/subject objects, preprocessing helpers, paths).
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

---

## Usage

1. Install FreeSurfer, the MELD classifier, and the Python dependencies above.
2. Edit the hardcoded paths at the top of each script to point at your own data, atlas, surface template, and HDF5 feature matrices.
3. Run the preprocessing scripts in order (steps 1–7). Keep the `hdf5_file_root` filenames consistent so each stage reads the previous stage's output.
4. Run any figure script directly, e.g.:

```bash
python figure_generation/FIG2/fig_2.py
```

Figures are written to the working directory (e.g. `fig_2.py` saves `relationshipspanel.png`).

---

## Notes

- The scripts assume MELD's directory layout (`FS_SUBJECTS_PATH`, `BASE_PATH`, etc., defined in `meld_classifier.paths`).
- Filtering to HS patients is applied both by passing a dataset CSV to `MeldCohort` and, in several scripts, by keeping IDs containing `hs` — the latter is redundant when the dataset is passed but is left in as a safeguard.
- Results outputs (surface PNGs, CSVs) are written to the current working directory unless a full path is given.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

## Contact

For questions about the analysis, please open an issue on this repository.

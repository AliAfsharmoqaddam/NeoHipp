#!/usr/bin/env python
# coding: utf-8
"""
Temporopolar blurring in HS: ipsilateral / contralateral surface t-maps + raincloud plot.

Produces three figures:
    1. ipsi_supp.png                     - ipsilateral HS vs controls, vertex-wise t-map
    2. supp_contra.png                   - contralateral HS vs controls, vertex-wise t-map
    3. ipsi_contra_control_raincloud.png - mean temporopolar score, all three groups

Feature values are loaded once per subject and reused for all three figures.
"""

import random

import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as multi
from scipy.stats import ttest_ind, mannwhitneyu, gaussian_kde
from matplotlib_surface_plotting import plot_surf

from meld_classifier.meld_cohort import MeldCohort, MeldSubject

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
INFO_CSV = '/home/meldstudent/Documents/RDS_NeoHipp/altered_info5_with_new_name.csv'
SURF_PATH = ('/home/meldstudent/Documents/RDS_NeoHipp/meld_data/output/'
             'fs_outputs/fsaverage_sym/surf/lh.white')
ANNOT_PATH = '/home/meldstudent/Downloads/lh.HCP-MMP1_sym.annot'
HDF5_ROOT = '{site_code}_{group}_featurematrix_combat_2.hdf5'

FEATURE = '.inter_z.intra_z.combat.on_lh.w-g.pct.sm10.mgh'
SITES = ['H1', 'H11', 'H16', 'H29']
TEMPORAL_POLE_LABELS = (131, 172)      # HCP-MMP1 TGd, TGv
SEED = 1                               # controls' hemisphere draw
ALPHA = 0.05

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
df_vis = pd.read_csv(INFO_CSV)
surf = nb.freesurfer.io.read_geometry(SURF_PATH)
aparc = nb.freesurfer.read_annot(ANNOT_PATH)

cohort = MeldCohort(hdf5_file_root=HDF5_ROOT)
cortex_idx = np.where(cohort.cortex_mask)[0]
temp_pole = np.isin(aparc[0], TEMPORAL_POLE_LABELS)

all_patients = cohort.get_subject_ids(site_codes=SITES, group='patient', lesional_only=False)
controls = cohort.get_subject_ids(site_codes=SITES, group='control', lesional_only=False)
patients = [s for s in all_patients if 'hs' in s]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def ipsi_hemi(subject):
    """Return 'lh'/'rh' for the subject's lesional (ipsilateral) hemisphere."""
    hemi = df_vis.loc[df_vis.classifier_new_id == subject, 'hemi']
    if hemi.empty:
        raise KeyError(f'No hemisphere recorded for {subject}')
    return 'lh' if str(hemi.iloc[0]).strip().lower() in ('left', 'lh') else 'rh'


def other(hemi):
    return 'rh' if hemi == 'lh' else 'lh'


def load_subject_values():
    """Full-surface feature values: (n_patients, n_vertices) x2, (n_controls, n_vertices)."""
    ipsi, contra = [], []
    for subject in patients:
        subj = MeldSubject(subject, cohort=cohort)
        hemi = ipsi_hemi(subject)
        ipsi.append(subj.load_feature_values(FEATURE, hemi))
        contra.append(subj.load_feature_values(FEATURE, other(hemi)))

    random.seed(SEED)
    ctrl = []
    for subject in controls:
        subj = MeldSubject(subject, cohort=cohort)
        ctrl.append(subj.load_feature_values(FEATURE, random.choice(['lh', 'rh'])))

    return np.array(ipsi), np.array(contra), np.array(ctrl)


patient_ipsi, patient_contra, control = load_subject_values()

# Quick sanity check on missing data within the cortex mask.
for name, arr in [('ipsi', patient_ipsi), ('contra', patient_contra), ('control', control)]:
    n_nan = np.isnan(arr[:, cortex_idx]).any(axis=1).sum()
    if n_nan:
        print(f'Warning: {n_nan} {name} subject(s) have NaNs inside the cortex mask')


# --------------------------------------------------------------------------- #
# Figures 1 & 2: vertex-wise t-maps
# --------------------------------------------------------------------------- #
def surface_ttest(patient_vals, control_vals):
    """Vertex-wise Welch t-test inside the cortex mask, Holm-corrected.

    Sign convention matches the original scripts: controls first, so positive
    t means controls > patients.
    """
    t_stats, p_vals = ttest_ind(control_vals[:, cortex_idx],
                                patient_vals[:, cortex_idx],
                                axis=0, equal_var=False)
    p_corrected = multi.multipletests(p_vals, method='holm', alpha=ALPHA)[1]

    t_map = np.zeros(len(cohort.cortex_mask))
    p_map = np.ones(len(cohort.cortex_mask))
    t_map[cortex_idx] = t_stats
    p_map[cortex_idx] = p_corrected
    return t_map, p_map < ALPHA


def plot_tmap(t_map, sig_mask, filename):
    plot_surf(surf[0], surf[1],
              overlay=t_map,
              mask=~cohort.cortex_mask,
              pvals=sig_mask,
              parcel=sig_mask,
              parcel_cmap={0: [0, 0, 0, 1], 1: [0, 0, 0, 1]},
              cmap='Reds',
              rotate=[90, 270],
              vmin=0, vmax=10,
              cmap_label=r"$t$-test",
              filename=filename)


plot_tmap(*surface_ttest(patient_ipsi, control), filename='ipsi_supp')
plot_tmap(*surface_ttest(patient_contra, control), filename='supp_contra')


# --------------------------------------------------------------------------- #
# Figure 3: raincloud of mean temporopolar values
# --------------------------------------------------------------------------- #
ipsi_tp = patient_ipsi[:, temp_pole].mean(axis=1)
contra_tp = patient_contra[:, temp_pole].mean(axis=1)
control_tp = control[:, temp_pole].mean(axis=1)

groups = [
    dict(name='Ipsilateral HS', values=ipsi_tp, colour='#E8735A', pos=3),
    dict(name='Contralateral HS', values=contra_tp, colour='#F5A623', pos=2),
    dict(name='Controls', values=control_tp, colour='#5A9BE8', pos=1),
]

fig, ax = plt.subplots(figsize=(9, 3.6))
rng = np.random.default_rng(seed=42)

for g in groups:
    values, colour, pos = g['values'], g['colour'], g['pos']
    ax.boxplot(values, positions=[pos], widths=0.15, patch_artist=True, notch=False,
               vert=False, manage_ticks=False,
               boxprops=dict(facecolor=colour, alpha=0.6, linewidth=1),
               medianprops=dict(color='black', linewidth=1),
               whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
               flierprops=dict(marker='o', markersize=3, markerfacecolor=colour, alpha=0.5))

    kde_x = np.linspace(values.min(), values.max(), 300)
    kde_y = gaussian_kde(values, bw_method=0.3)(kde_x)
    kde_y = kde_y / kde_y.max() * 0.35
    ax.fill_between(kde_x, pos, pos + kde_y, alpha=0.6, color=colour)
    ax.plot(kde_x, pos + kde_y, color=colour, linewidth=1)

    jitter = rng.uniform(-0.08, 0.0, size=len(values))
    ax.scatter(values, pos - 0.18 + jitter, alpha=0.4, s=6, color=colour, linewidths=0)

# --- Three Mann-Whitney U tests, Bonferroni-corrected ---
# Each entry: (bracket span y1-y2, label y, x offset from x0, test result).
# The x offsets are hand-tuned - adjust these to nudge bracket levels.
N_COMP = 3
comparisons = [
    ((2, 3), 2.5, 0.0, mannwhitneyu(ipsi_tp, contra_tp, alternative='two-sided')),
    ((1, 2), 1.5, 0.6, mannwhitneyu(contra_tp, control_tp, alternative='two-sided')),
    ((1, 3), 2.0, 3.5, mannwhitneyu(ipsi_tp, control_tp, alternative='two-sided')),
]


def fmt(u, p):
    pc = min(p * N_COMP, 1.0)
    ps = r'$p$ < 0.001' if pc < 0.001 else rf'$p$ = {pc:.3f}'
    return rf'$U$ = {u:.0f}, {ps}'


labels = [fmt(u, p) for _, _, _, (u, p) in comparisons]

# --- Bracket geometry: measure label width so spacing is tight but clip-free ---
all_vals = np.concatenate([ipsi_tp, contra_tp, control_tp])
x_span = all_vals.max() - all_vals.min()
FS = 8.5

fig.canvas.draw()  # renderer must exist before measuring text
renderer = fig.canvas.get_renderer()


def label_width_data(txt):
    t = ax.text(0, 0, txt, fontsize=FS)
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    inv = ax.transData.inverted()
    (xa, _), (xb, _) = inv.transform((0, 0)), inv.transform((bb.width, 0))
    return abs(xb - xa)


w_max = max(label_width_data(l) for l in labels)
text_pad = 0.015 * x_span            # bracket line -> its text
dx = text_pad + w_max + 0.05 * x_span  # canvas allowance per bracket level (see set_xlim)
x0 = all_vals.max() + 0.10 * x_span
prong = 0.03 * x_span

for ((y1, y2), ytext, x_offset, _), label in zip(comparisons, labels):
    x = x0 + x_offset
    ax.plot([x, x], [y1, y2], color='#555555', lw=1, clip_on=False)
    for y in (y1, y2):
        ax.plot([x - prong, x], [y, y], color='#555555', lw=1, clip_on=False)
    ax.text(x + text_pad, ytext, label, ha='left', va='center',
            fontsize=FS, color='#333333', clip_on=False)

# --- Axes cosmetics ---
X_AXIS_MAX = 2                       # visible x-axis stops here; canvas extends for brackets
x_left = all_vals.min() - 0.12 * x_span

ax.axvline(x=0, color='grey', linewidth=0.8, zorder=0)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Controls', 'Contralateral HS', 'Ipsilateral HS'], fontsize=11)
ax.set_xlabel('Mean temporopolar blurring score', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0.55, 3.5)
ax.set_xlim(x_left, x0 + 2 * dx + text_pad + w_max + 0.02 * x_span)
ax.set_xticks([-4, -2, 0, X_AXIS_MAX])
ax.spines['bottom'].set_bounds(x_left, X_AXIS_MAX)
ax.tick_params(axis='y', length=0, labelsize=11)
ax.tick_params(axis='x', labelsize=11)

plt.tight_layout()
plt.savefig('ipsi_contra_control_raincloud.png', dpi=300, bbox_inches='tight')
plt.show()
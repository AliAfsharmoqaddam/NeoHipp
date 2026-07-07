import re
import random

import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from meld_classifier.meld_cohort import MeldCohort, MeldSubject

#AVERAGE SCRIPT RUNTIME = 8-10 MINS 

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/altered_info5_with_new_name.csv')
aparc = nb.freesurfer.read_annot('/home/meldstudent/Downloads/lh.HCP-MMP1_sym.annot')

DATASET_FILE = '/home/meldstudent/Documents/RDS_NeoHipp/final_dataset_no_fcd.csv'
c = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_final_no_fcd.hdf5', dataset=DATASET_FILE)
c_blur = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_2.hdf5')

total_patients = c.get_subject_ids(site_codes=['H1', 'H11', 'H16', 'H29'], group='patient', lesional_only=False)
controls = c.get_subject_ids(site_codes=['H1', 'H11', 'H16', 'H29'], group='control', lesional_only=False)
patients = [x for x in total_patients if 'hs' in x]

# Temporopolar ROI (TGd + TGv)
temp_pole = (aparc[0] == 131) | (aparc[0] == 172)

BLUR_FEATURE = '.inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def subject_hemi(subject):
    """Hemisphere recorded for this subject in the demographics table."""
    hemi = df[df.classifier_new_id == subject].hemi.to_string(index=False)
    if hemi in ('left', 'lh'):
        return 'lh'
    if hemi in ('right', 'rh'):
        return 'rh'
    raise ValueError(f"Unrecognised hemisphere '{hemi}' for {subject}")


def mean_temporopolar(subjects, feature, cohort):
    """Mean temporopolar value of `feature` for each subject in `subjects`."""
    vals = []
    for subject in subjects:
        subj = MeldSubject(subject, cohort=cohort)
        v = subj.load_feature_values(feature, subject_hemi(subject))
        vals.append(v[temp_pole].mean())
    return np.array(vals)


def control_mean(feature):
    """Mean temporopolar value of `feature` per control.

    Controls have no ipsilateral side, so a hemisphere is chosen at random;
    the seed is reset each call so the assignment is identical across features.
    """
    random.seed(1)
    vals = []
    for subject in controls:
        subj = MeldSubject(subject, cohort=c)
        hemi = 'lh' if random.choice(['left', 'right']) == 'left' else 'rh'
        v = subj.load_feature_values(feature, hemi)
        vals.append(v[temp_pole].mean())
    return np.array(vals)


# ---------------------------------------------------------------------------
# Identify "blurred" HS patients (temporopolar blurring score < -1)
# ---------------------------------------------------------------------------
blur_scores = mean_temporopolar(patients, BLUR_FEATURE, cohort=c_blur)
blur_patients = [subj for subj, score in zip(patients, blur_scores) if score < -1]


# ---------------------------------------------------------------------------
# Depth-profile features
# ---------------------------------------------------------------------------
gm_dist_features = [
    f'.inter_z.asym.intra_z.combat.on_lh.gm_T1_{"-" if d else ""}{d}mm.sm5.mgh'
    for d in [round(0.25 * i, 2) for i in range(29)]
]
distances = [round(0.25 * i, 2) for i in range(29)]  # 0.0 .. 7.0 mm

gm_frac_features = [
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_1.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.9.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.8.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.7.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.6.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.5.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.4.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.3.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.2.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.1.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.gm_T1_0.sm5.mgh',
]
fractions = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

wm_dist_features = [
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_0mm.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-0.5mm.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-1mm.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-1.5mm.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-2mm.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-2.5mm.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-3mm.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-3.5mm.sm5.mgh',
    '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-4mm.sm5.mgh',
]
wm_distances = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]


# ---------------------------------------------------------------------------
# Build depth matrices for the blurred HS group
# ---------------------------------------------------------------------------
patient_blur_gm_dist_asym_feature_matrix = {
    dist: mean_temporopolar(blur_patients, feat, c)
    for dist, feat in zip(distances, gm_dist_features)
}
patient_blur_gm_frac_asym_feature_matrix = {
    frac: mean_temporopolar(blur_patients, feat, c)
    for frac, feat in zip(fractions, gm_frac_features)
}
patient_blur_wm_dist_asym_feature_matrix = {
    dist: mean_temporopolar(blur_patients, feat, c)
    for dist, feat in zip(wm_distances, wm_dist_features)
}


# ---------------------------------------------------------------------------
# Same depth matrices for the control group
# ---------------------------------------------------------------------------
control_gm_dist_asym_feature_matrix = {
    dist: control_mean(feat) for dist, feat in zip(distances, gm_dist_features)
}
control_gm_frac_asym_feature_matrix = {
    frac: control_mean(feat) for frac, feat in zip(fractions, gm_frac_features)
}
control_wm_dist_asym_feature_matrix = {
    dist: control_mean(feat) for dist, feat in zip(wm_distances, wm_dist_features)
}


# ---------------------------------------------------------------------------
# Significance testing (blurred HS vs controls): Mann-Whitney U + Holm
# The Holm-corrected p-values printed here are what the hard-coded asterisks
# on each figure are based on.
# ---------------------------------------------------------------------------
# Main figure: GM fractions (Pial -> GM/WM) followed by WM distances
p_vals_main = []
for frac in fractions:
    _, p = mannwhitneyu(patient_blur_gm_frac_asym_feature_matrix[frac],
                        control_gm_frac_asym_feature_matrix[frac])
    p_vals_main.append(p)
for dist in wm_distances:
    _, p = mannwhitneyu(patient_blur_wm_dist_asym_feature_matrix[dist],
                        control_wm_dist_asym_feature_matrix[dist])
    p_vals_main.append(p)
p_vals_main_cor = multipletests(p_vals_main, method='holm', alpha=0.05)[1]

main_labels = [('Pial' if f == 1 else 'GM/WM' if f == 0 else f'{int(f * 100)}%') for f in fractions]
main_labels += [f'{d:g}mm WM' for d in wm_distances]
print("Main figure - Holm-corrected Mann-Whitney p-values (blurred HS vs controls):")
for lab, p in zip(main_labels, p_vals_main_cor):
    print(f"  {lab:>9}: {p:.2e}{'  *' if p < 0.05 else ''}")

# Supplementary figure: GM distance from the pial surface
p_vals_supp = []
for dist in distances:
    _, p = mannwhitneyu(patient_blur_gm_dist_asym_feature_matrix[dist],
                        control_gm_dist_asym_feature_matrix[dist])
    p_vals_supp.append(p)
p_vals_supp_cor = multipletests(p_vals_supp, method='holm', alpha=0.05)[1]

print("\nSupp figure - Holm-corrected Mann-Whitney p-values (blurred HS vs controls):")
for dist, p in zip(distances, p_vals_supp_cor):
    label = 'Pial' if dist == 0 else f'{dist:.2f}mm'
    print(f"  {label:>8}: {p:.2e}{'  *' if p < 0.05 else ''}")


# ---------------------------------------------------------------------------
# Figure 1 (main): GM fractions + WM distances  ->  depth_plot_transparent3.png
# ---------------------------------------------------------------------------
frac_dict = patient_blur_gm_frac_asym_feature_matrix
mm_dict = patient_blur_wm_dist_asym_feature_matrix

frac_keys = sorted(frac_dict.keys(), reverse=True)        # 1 -> 0
mm_keys = sorted(k for k in mm_dict.keys() if k != 0)     # exclude 0 (duplicates GM/WM)

data_1 = {}
for depth in frac_keys:
    if depth == 1:
        col_name = "Pial"
    elif depth == 0:
        col_name = "GM/WM"
    else:
        col_name = f"{int(depth * 100)}%"
    data_1[col_name] = frac_dict[depth]
for depth in mm_keys:
    data_1[f"{depth:.1f}mm"] = mm_dict[depth]

df_main = pd.DataFrame(data_1)

medians = df_main.median(axis=0)
norm = plt.Normalize(medians.min(), medians.max())
cmap = plt.cm.viridis_r
colors = [cmap(norm(medians[col])) for col in df_main.columns]

# Prepend '-' to mm-depth labels (they sit on the WM side of 0)
display_labels = [
    f'-{col}' if re.match(r'^\d+\.?\d*mm$', col) else col
    for col in df_main.columns
]

fig, ax = plt.subplots(figsize=(6, 7))
box = ax.boxplot(
    df_main.values,
    patch_artist=True,
    vert=False,
    labels=display_labels,
    flierprops=dict(marker='.', markersize=0, markerfacecolor='black', markeredgecolor='black', alpha=0.6),
    medianprops=dict(color='white', linewidth=1, linestyle='-'),
)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

ax.set_xlabel("Mean temporopolar T1w intensity asymmetries", fontsize=11)
ax.invert_yaxis()
ax.axvline(x=0, lw=1, color='grey', alpha=0.7)
ax.set_xlim(-3, 2)
ax.tick_params(axis='y', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Colorbar
cax = inset_axes(ax, width="60%", height="2%", loc='lower center',
                 bbox_to_anchor=(0, -0.12, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_ticks([-0.6, -0.3, 0])
cbar.set_ticklabels(["-0.6", "-0.3", "0"])
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=10, length=0, pad=2)

# Significance asterisks, aligned to boxes, from the "10%" row onwards.
# Thresholds are hard-coded from p_vals_main_cor (printed above).
labels = list(df_main.columns)
start_idx = labels.index("10%")
for i in range(start_idx, len(labels)):
    y_pos = i + 1  # boxplot positions are 1-based
    right_whisker = box['whiskers'][2 * i + 1].get_xdata().max()
    x_pos = right_whisker + 0.1
    if i == start_idx:
        stars = "*"
    elif i >= len(labels) - 2:
        stars = "**"
    else:
        stars = "***"
    ax.text(x_pos, y_pos + 0.13, stars, color="red", fontsize=12, va="center", ha="left")

ax.tick_params(axis='y', length=0, labelsize=11)
ax.tick_params(axis='x', labelsize=11)
plt.savefig("depth_plot_main.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# ---------------------------------------------------------------------------
# Figure 2 (supplementary): GM distance from pial  ->  depth_plot_pial_supp2.png
# ---------------------------------------------------------------------------
df_supp = pd.DataFrame(patient_blur_gm_dist_asym_feature_matrix)

medians2 = df_supp.median(axis=0)
norm2 = plt.Normalize(medians2.min(), medians2.max())
cmap2 = plt.cm.viridis_r
colors2 = [cmap2(norm2(medians2[col])) for col in df_supp.columns]

display_labels = [
    'Pial' if float(col) == 0.0 else f'{float(col):.2f}'
    for col in df_supp.columns
]

fig, ax = plt.subplots(figsize=(6, 7))
box = ax.boxplot(
    df_supp.values,
    patch_artist=True,
    vert=False,
    labels=display_labels,
    flierprops=dict(marker='.', markersize=0, markerfacecolor='black', markeredgecolor='black', alpha=0.6),
    medianprops=dict(color='white', linewidth=1, linestyle='-'),
)
for patch, color in zip(box['boxes'], colors2):
    patch.set_facecolor(color)

ax.set_xlabel("Mean temporopolar T1w intensity asymmetries", fontsize=11)
ax.set_ylabel('Distance from pial surface (mm)', fontsize=11)
ax.invert_yaxis()
ax.axvline(x=0, lw=1, color='grey', alpha=0.7)
ax.set_xlim(-2, 2)
ax.set_xticks(range(-2, 3, 1))
ax.tick_params(axis='y', length=0, labelsize=11)
ax.tick_params(axis='x', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Significance asterisks by depth value (hard-coded from p_vals_supp_cor, printed above)
for i, col in enumerate(df_supp.columns):
    val = float(col)
    if val == 3.75:
        stars = '*'
    elif val == 4.00:
        stars = '**'
    elif val >= 4.25:
        stars = '***'
    else:
        continue
    y_pos = i + 1
    right_whisker = box['whiskers'][2 * i + 1].get_xdata().max()
    x_pos = right_whisker + 0.1
    ax.text(x_pos, y_pos + 0.11, stars, color='red', fontsize=10, va='center', ha='left')

# Colorbar
cax = inset_axes(ax, width="60%", height="2%", loc='lower center',
                 bbox_to_anchor=(0, -0.12, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, cax=cax, orientation='horizontal')
cbar2.set_ticks([-0.6, -0.3, 0])
cbar2.set_ticklabels(["-0.6", "-0.3", "0"])
cbar2.outline.set_visible(False)
cbar2.ax.tick_params(labelsize=10, length=0, pad=2)

plt.savefig("depth_plot_supp.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()
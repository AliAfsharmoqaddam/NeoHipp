import warnings
import numpy as np
import pandas as pd
import nibabel as nb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import spearmanr
from pygam import LinearGAM, s
from meld_classifier.meld_cohort import MeldCohort, MeldSubject

warnings.filterwarnings('ignore')

# ===========================================================================
# Load data
# ===========================================================================
df = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/altered_info5_with_new_name.csv')
df_hipp = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/aidhs/matrix_norm_avg_240624.csv')
aparc = nb.freesurfer.read_annot('/home/meldstudent/Downloads/lh.HCP-MMP1_sym.annot')
DATASET_FILE = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/final_dataset_h16_fcd.csv')

#Instantiate cohort
c = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_2.hdf5', dataset=DATASET_FILE)
total_patients = c.get_subject_ids(site_codes=['H1', 'H11', 'H16', 'H29'], group='patient', lesional_only=False)

#Redundant if dataset is passed to MeldCohort 
patients = [x for x in total_patients if 'hs' in x]

#Define w-g contrast feature globally 
feature = '.inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh'

#Define temporopolar ROI 
temp_pole = (aparc[0] == 131) | (aparc[0] == 172)   # TGd + TGv

# ===========================================================================
# Hippocampal volume asymmetry table (ipsilateral only)
# ===========================================================================
df_hipp = df_hipp[['ID', '.inter_z.asym.combat.label-avg.FS_volume_icvcorr', 'lesional']]
df_hipp_renamed = df_hipp.rename(columns={'.inter_z.asym.combat.label-avg.FS_volume_icvcorr': 'fs_vol_asym'})
df_hipp_renamed = df_hipp_renamed.set_index('ID')
df_hipp_renamed = df_hipp_renamed[~df_hipp_renamed['lesional'].str.contains('contra', case=False, na=False)]

# ===========================================================================
# Build the patient dataframe (blurring z-score, duration, histology, hipp asym)
# ===========================================================================
df_patients = df[['classifier_new_id', 'sex (M=0, F=1)', 'age_at_scans_years', 'epilepsy_onset_years', 'histo_diagnosis', 'hemi']].copy()
df_patients = df_patients[df_patients.classifier_new_id.isin(patients)]
df_patients = df_patients.set_index('classifier_new_id')
df_patients['z_score'] = np.nan

#Iterate through df rows and populate with blurring score, histology, epilepsy duration 
for i, row in df_patients.iterrows():
    subj = MeldSubject(i, cohort=c)
    hemi = df[df.classifier_new_id == i].hemi.to_string(index=False)
    histo = df[df.classifier_new_id == i].histo_diagnosis.to_string(index=False)
    if hemi in ('lh', 'left'):
        vals = subj.load_feature_values(feature, 'lh')
    elif hemi in ('rh', 'right'):
        vals = subj.load_feature_values(feature, 'rh')
    df_patients.at[i, 'histo_diagnosis'] = 1 if 'FCD 3' in histo else 0
    df_patients.at[i, 'z_score'] = vals[temp_pole].mean()

#Clean up df 
df_patients['histo_diagnosis'] = df_patients.histo_diagnosis.astype(int)
df_patients['epilepsy_onset_years'] = df_patients.epilepsy_onset_years.astype(float)
df_patients['Duration'] = df_patients.age_at_scans_years - df_patients.epilepsy_onset_years
df_patients['histo_string'] = df_patients.histo_diagnosis.apply(lambda x: 'FCD IIIa' if x == 1 else 'HS only')
df_patients = df_patients.rename(columns={'sex (M=0, F=1)': 'sex',
                                          'age_at_scans_years': 'age_at_scan',
                                          'epilepsy_onset_years': 'epilepsy_onset'})

#Populate hippocampal volume asymmetry 
df_patients['hipp_asym'] = np.nan
for i, val in df_hipp_renamed.iterrows():
    vol = df_hipp_renamed.at[i, 'fs_vol_asym']
    new_id = df[df['study ID'] == i].classifier_new_id.to_string(index=False)
    if new_id in patients:
        df_patients.at[new_id, 'hipp_asym'] = vol

# Rows with both epilepsy onset and hippocampal asymmetry available
df_matrix = df_patients.dropna(subset=['epilepsy_onset', 'hipp_asym'])

# ===========================================================================
# Figure: relationships panel (A onset, B duration, C correlation matrix, D group)
# ===========================================================================
df1 = df_matrix
df1['histo_string'] = df1['histo_string'].astype(str)

# ── Publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'axes.linewidth':   0.8,
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'figure.dpi':       150,
})

SCATTER_COLOR = '#2C6BAC'
VIOLIN_PALETTE = {'FCD IIIa': '#E07B39', 'HS only': '#2C6BAC'}
ALPHA = 0.7
DOT_SIZE = 28
Y_LABEL = 'Mean temporopolar blurring score'

#Helper function for determining ylim 
def pad_ylim(series_list, pad=0.12):
    combined = np.concatenate([s.dropna().values for s in series_list])
    lo, hi = combined.min(), combined.max()
    margin = (hi - lo) * pad
    return lo - margin, hi + margin

#Helper function for p val colouring 
def stat_color(p_corrected):
    return '#C0392B' if p_corrected < 0.05 else '#999999'

#Helper function for p val dp format
def fmt_p(p_corrected):
    if p_corrected < 0.0001:
        return 'p < 0.0001'
    return f'p = {p_corrected:.4f}'

ylim_top = pad_ylim([df1['z_score']])
ylim_bottom = ylim_top   # df1 and df2 are the same frame here

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 7), facecolor='white')

# Top row: 2 equal columns; bottom row split [2, 1] so C is wide and D is narrow
outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.10,
                             left=0.11, right=0.97, top=0.94, bottom=0.09)
ax_tl = fig.add_subplot(outer_gs[0, 0])
ax_tr = fig.add_subplot(outer_gs[0, 1])

bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1, :],
                                             width_ratios=[2, 1], wspace=0.28)
ax_bl = fig.add_subplot(bottom_gs[0])
ax_br = fig.add_subplot(bottom_gs[1])

# ═══════════════════════════════════════════════════════════════════════════
# A  Epilepsy onset vs blurring score (linear GAM with 95% CI)
# ═══════════════════════════════════════════════════════════════════════════
x_on = df1['epilepsy_onset'].dropna().values
y_on = df1.loc[df1['epilepsy_onset'].notna(), 'z_score'].values

#Scatter plot 
ax_tl.scatter(x_on, y_on, color=SCATTER_COLOR, alpha=ALPHA, s=DOT_SIZE, linewidths=0, zorder=3)

#GAM line with 95% CI 
gam = LinearGAM(s(0, n_splines=4, spline_order=2)).fit(x_on, y_on)
x_fit = np.linspace(x_on.min(), x_on.max(), 300)
y_fit = gam.predict(x_fit)
ci = gam.confidence_intervals(x_fit, width=0.95)
ax_tl.plot(x_fit, y_fit, color='#C0392B', lw=2, zorder=4)
ax_tl.fill_between(x_fit, ci[:, 0], ci[:, 1], color='#C0392B', alpha=0.15, zorder=2, linewidth=0)

# Spearman (Bonferroni x6). Not annotated on the figure; also shown in panel C.
rho_a, p_sp_a_raw = spearmanr(x_on, y_on)
p_sp_a_corr = p_sp_a_raw * 6
print(f"Panel A  onset vs blurring:    rs = {rho_a:.3f}, {fmt_p(p_sp_a_corr)} (Bonferroni x6)")

#Adjustments 
ax_tl.set_xlabel('Epilepsy onset (years)', fontsize=11)
ax_tl.set_ylabel(Y_LABEL, fontsize=11)
ax_tl.set_ylim(*ylim_top)
ax_tl.tick_params(axis='x', labelsize=11)
ax_tl.tick_params(axis='y', labelsize=11)
ax_tl.set_xticks(range(0, 41, 10))

# ═══════════════════════════════════════════════════════════════════════════
# B  Epilepsy duration vs blurring score
# ═══════════════════════════════════════════════════════════════════════════
mask_b = df1['Duration'].notna() & df1['z_score'].notna()
x_dur = df1.loc[mask_b, 'Duration'].values
y_dur = df1.loc[mask_b, 'z_score'].values

#Scatter plot 
ax_tr.scatter(x_dur, y_dur, color=SCATTER_COLOR, alpha=ALPHA, s=DOT_SIZE, linewidths=0, zorder=3)

# Spearman (Bonferroni x6). Not annotated on the figure; also shown in panel C.
rho_b, p_sp_b_raw = spearmanr(x_dur, y_dur)
p_sp_b_corr = p_sp_b_raw * 6
print(f"Panel B  duration vs blurring: rs = {rho_b:.3f}, {fmt_p(p_sp_b_corr)} (Bonferroni x6)")

#Adjustments
ax_tr.set_xlabel('Epilepsy duration (years)', fontsize=11)
ax_tr.set_ylabel('')
ax_tr.set_ylim(*ylim_top)
ax_tr.set_yticklabels([])
ax_tr.tick_params(axis='x', labelsize=11)

# ═══════════════════════════════════════════════════════════════════════════
# C  Staircase Spearman correlation matrix (Bonferroni x6)
# ═══════════════════════════════════════════════════════════════════════════
cols_mat = ['epilepsy_onset', 'Duration', 'z_score', 'hipp_asym']
labels_mat = ['Onset', 'Duration', 'Blurring\nscore', 'Hippocampal\nvolume\nasymmetry']
n_mat = len(cols_mat)

#Generate matrix data  
r_mat = np.full((n_mat, n_mat), np.nan)
p_mat = np.full((n_mat, n_mat), np.nan)
for i in range(n_mat):
    for j in range(i):
        r, p = stats.spearmanr(df_matrix[cols_mat[i]], df_matrix[cols_mat[j]], nan_policy='omit')
        r_mat[i, j] = r
        p_mat[i, j] = min(p * 6, 1.0)   # Bonferroni correction for 6 comparisons

#Adjustments 
cmap_mat = plt.cm.Blues
ax_bl.set_xlim(1, n_mat - 1)
ax_bl.set_ylim(-0.3, n_mat - 1)
ax_bl.set_aspect('equal', adjustable='datalim')
ax_bl.axis('off')

#Render staircase matrix with custom text position  
for i in range(1, n_mat):
    for j in range(i):
        r = r_mat[i, j]
        p = p_mat[i, j]
        abs_r = abs(r)
        color = cmap_mat(abs_r)
        x = j
        y = (n_mat - 1) - i

        ax_bl.add_patch(plt.Rectangle([x + 0.05, y + 0.05], 0.9, 0.9, color=color, zorder=1))

        text_color = 'white' if abs_r >= 0.5 else '#042C53'
        stars = '***' if p < 0.0001 else ('**' if p < 0.001 else ('*' if p < 0.01 else ''))
        r_sign = '−' if r < 0 else ''
        r_str = f"{r_sign}{abs_r:.3f}"
        p_str = r"$p_{\mathrm{corr}}$<.001" if p < 0.001 else rf"$p_{{\mathrm{{corr}}}}$={p:.3f}"

        ax_bl.text(x + 0.5, y + 0.62, r_str, ha='center', va='center',
                   fontsize=9, fontweight='bold', color=text_color, zorder=2)
        ax_bl.text(x + 0.5, y + 0.40, p_str, ha='center', va='center',
                   fontsize=7.5, color=text_color, alpha=0.85, zorder=2)
        ax_bl.text(x + 0.5, y + 0.22, stars, ha='center', va='center',
                   fontsize=8, color='r', zorder=2)

# Row labels
for i, label in enumerate(labels_mat[1:], 1):
    y_pos = (n_mat - 1) - i + 0.5
    ax_bl.text(-0.06, y_pos, label, ha='right', va='center',
               fontsize=9, color='gray', transform=ax_bl.transData)

# Column labels (placed BELOW the bottom row of the staircase)
for j, label in enumerate(labels_mat[:-1]):
    ax_bl.text(j + 0.5, -0.10, label, ha='center', va='top',
               fontsize=9, color='gray', transform=ax_bl.transData)

# Colourbar, centred vertically alongside panel C
fig.canvas.draw()   # flush layout so get_position() is accurate
pos = ax_bl.get_position()
cbar_width = 0.012
cbar_height = pos.height * 0.40
cbar_left = pos.x1 - cbar_width - 0.1
cbar_bottom = pos.y0 + (pos.height - cbar_height) / 2
cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
sm = plt.cm.ScalarMappable(cmap=cmap_mat, norm=mcolors.Normalize(0, 1))
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label(r'|$r$|', fontsize=9)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.set_ticklabels(["0", '0.5', "1"])
cbar.ax.tick_params(labelsize=9, length=0, pad=2)

# ═══════════════════════════════════════════════════════════════════════════
# D  Violin + boxplot by histology, with Mann-Whitney U
# ═══════════════════════════════════════════════════════════════════════════
order = ['FCD IIIa', 'HS only']
violin_colors = [VIOLIN_PALETTE[k] for k in order]

#Violin plot 
sns.violinplot(data=df1, x='histo_string', y='z_score', order=order, palette=violin_colors, inner=None, cut=0, linewidth=0.8, ax=ax_br, alpha=0.55, saturation=0.85)
for poly in ax_br.collections:
    poly.set_alpha(0.50)

#Box plot 
sns.boxplot(data=df1, 
            x='histo_string', 
            y='z_score', 
            order=order,
            width=0.15, 
            fliersize=3,
            boxprops=dict(facecolor='white', edgecolor='#333', linewidth=1.2),
            medianprops=dict(color='#C0392B', linewidth=2),
            whiskerprops=dict(color='#333', linewidth=1),
            capprops=dict(color='#333', linewidth=1),
            flierprops=dict(marker='o', markerfacecolor='#666', markeredgewidth=0, markersize=3, alpha=0.6),
            ax=ax_br
           )

#Mann Whitney U 
g1 = df1.loc[df1['histo_string'] == 'FCD IIIa', 'z_score'].dropna()
g2 = df1.loc[df1['histo_string'] == 'HS only', 'z_score'].dropna()
U_stat, p_vio = stats.mannwhitneyu(g1, g2, alternative='two-sided')
p_vio_str = 'p < 0.0001' if p_vio < 0.0001 else f'p = {p_vio:.3f}'
col_vio = stat_color(p_vio)
print(f"Panel D  FCD IIIa vs HS only:  U = {U_stat:.0f}, {p_vio_str}")

#Adjustments 
y_max = ylim_bottom[1] * 0.82
ax_br.plot([0, 0, 1, 1], [y_max - 0.15, y_max, y_max, y_max - 0.15], lw=1, color='#333')
ax_br.text(0.5, y_max + 0.5, f'U = {U_stat:.0f}', ha='center', va='bottom',
           fontsize=8.5, fontstyle='italic', color=col_vio)
ax_br.text(0.5, y_max + 0.15, p_vio_str, ha='center', va='bottom',
           fontsize=8.5, fontstyle='italic', color=col_vio)

ax_br.set_xlabel('')
ax_br.set_ylabel('Mean temporopolar blurring score', fontsize=11)
ax_br.set_ylim(*ylim_bottom)
ax_br.set_xticklabels(order, fontsize=11)
ax_br.tick_params(axis='x', length=0, labelsize=11)
ax_br.tick_params(axis='y', labelsize=11)

#Save and render 
fig.savefig('relationshipspanel.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

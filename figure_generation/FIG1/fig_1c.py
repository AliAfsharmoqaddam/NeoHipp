import random
import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, gaussian_kde
from meld_classifier.meld_cohort import MeldCohort, MeldSubject

# Load demographics
df = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/altered_info5_with_new_name.csv')
DATASET_FILE = '/home/meldstudent/Documents/RDS_NeoHipp/final_dataset_no_fcd.csv'

# Load Glasser atlas (symmetric)
aparc = nb.freesurfer.read_annot('/home/meldstudent/Downloads/lh.HCP-MMP1_sym.annot')

# Load cohort with corresponding hdf5
c = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_2.hdf5', dataset=DATASET_FILE)

# Load participants
total_patients = c.get_subject_ids(site_codes=['H1', 'H11', 'H16', 'H29'], group='patient', lesional_only=False)
controls = c.get_subject_ids(site_codes=['H1', 'H11', 'H16', 'H29'], group='control', lesional_only=False)

# Keep only HS patients (redundant if dataset is passed to MeldCohort instantiation) 
patients = [x for x in total_patients if 'hs' in x]

# Define w-g contrast feature globally 
FEATURE = '.inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh'

# Define temporopolar ROI (TGd + TGv)
temp_pole = (aparc[0] == 131) | (aparc[0] == 172)


def load_polar_values(group):
    """Mean temporopolar blurring score for each subject in the group.

    Patients use the hemisphere recorded in df_vis; controls are assigned a
    random hemisphere (seeded for reproducibility).
    """
    if group == 'patient':
        subjects = patients
    elif group == 'control':
        subjects = controls
        random.seed(1)
    else:
        raise ValueError(f"Unknown group: {group}")

    polar_vals = []
    for subject in subjects:
        if group == 'patient':
            hemi = df[df.classifier_new_id == subject].hemi.to_string(index=False)
        else:
            hemi = random.choice(['left', 'right'])

        if hemi in ('left', 'lh'):
            hemi = 'lh'
        elif hemi in ('right', 'rh'):
            hemi = 'rh'

        subj = MeldSubject(subject, cohort=c)
        vals = subj.load_feature_values(FEATURE, hemi)
        polar_vals.append(vals[temp_pole].mean())

    return np.array(polar_vals)


# Mean temporopolar blurring scores per group
patient_polar_vals = load_polar_values('patient')
control_polar_vals = load_polar_values('control')

#Figure setup 
fig, ax = plt.subplots(figsize=(6, 3))
data = [patient_polar_vals, control_polar_vals]
labels = ['HS', 'Controls']
colours = ['#E8735A', '#5A9BE8']
positions = [2, 1]

# Mann-Whitney U test
stat, p_val = mannwhitneyu(patient_polar_vals, control_polar_vals, alternative='two-sided')

for pos, values, colour in zip(positions, data, colours):
    # Horizontal box plot
    bp = ax.boxplot(values, 
                    positions=[pos], 
                    widths=0.15,
                    patch_artist=True, 
                    notch=False,
                    vert=False,
                    manage_ticks=False,
                    boxprops=dict(facecolor=colour, alpha=0.6, linewidth=1),
                    medianprops=dict(color='black', linewidth=1),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                    flierprops=dict(marker='o', markersize=3, markerfacecolor=colour, alpha=0.5)
                   )
    # Half violin (above each box plot)
    kde = gaussian_kde(values, bw_method=0.3)
    kde_range = np.linspace(values.min(), values.max(), 300)
    kde_vals = kde(kde_range)
    kde_vals = kde_vals / kde_vals.max() * 0.35  # scale height
    ax.fill_between(kde_range, pos, pos + kde_vals, alpha=0.6, color=colour)
    ax.plot(kde_range, pos + kde_vals, color=colour, linewidth=1)

    # Rain (jittered data points below box plot)
    rng = np.random.default_rng(seed=42)
    jitter = rng.uniform(-0.08, 0.0, size=len(values))
    rain_y = pos - 0.18 + jitter
    ax.scatter(values, rain_y, alpha=0.4, s=6, color=colour, linewidths=0)

# Vertical reference line at x=0
ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='-', zorder=0)

# Annotate with Mann-Whitney results — italic U and p via math text
p_text = r'$p$ < 0.001' if p_val < 0.001 else rf'$p$ = {p_val:.3f}'
ax.text(0.98, 
        0.05,
        rf'$U$ = {stat:.0f}, \n{p_text}',
        transform=ax.transAxes,
        fontsize=10,
        ha='right', va='bottom',
        color='#444444'
       )

#Adjustments   
ax.set_yticks(positions)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Mean temporopolar blurring score', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Clip y-axis so it doesn't extend beyond the top KDE curve
ax.set_ylim(0.3, 2 + 0.35 + 0.05)

#Adjust ticks 
ax.tick_params(axis='y', length=0, labelsize=11)
ax.tick_params(axis='x', labelsize=11)

#Render
plt.tight_layout()
plt.show()

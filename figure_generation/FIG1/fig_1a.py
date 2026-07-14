import random
import numpy as np
import pandas as pd
import nibabel as nb
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as multi
from matplotlib_surface_plotting import plot_surf
from meld_classifier.meld_cohort import MeldCohort, MeldSubject

# Load demographics
df = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/altered_info5_with_new_name.csv')
DATASET_FILE = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/final_dataset_no_fcd.csv') 

# Load white surface
surf = nb.freesurfer.io.read_geometry('/home/meldstudent/Documents/RDS_NeoHipp/meld_data/output/fs_outputs/fsaverage_sym/surf/lh.white')

# Load cohort with corresponding hdf5
c = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_2.hdf5', dataset=DATASET_FILE)

# Index location of cortex (used to mask out the medial wall)
cortex_mask_index = np.where(c.cortex_mask == True)[0]

# Load patients and controls
total_patients = c.get_subject_ids(site_codes=['H1', 'H11', 'H16', 'H29'], group='patient', lesional_only=False)
controls = c.get_subject_ids(site_codes=['H1', 'H11', 'H16', 'H29'], group='control', lesional_only=False)

# Keep only HS patients (redundant if dataset passed to MeldCohort instantiation) 
patients = [x for x in total_patients if 'hs' in x]

# Define w-g contrast feature globally
FEATURE = '.inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh'

def load_group_values(group, feature):
    """Return cortex-masked feature values for every subject in a group.

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

    group_vals = []
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
        vals = subj.load_feature_values(feature, hemi)
        group_vals.append(vals[cortex_mask_index])

    return np.array(group_vals)


# Load group-level contrast feature
patient_vals_p = load_group_values('patient', FEATURE)
control_vals_p = load_group_values('control', FEATURE)

# Welch's t-test across every cortical vertex at once
t_test_stats, p_vals = ttest_ind(control_vals_p, patient_vals_p, equal_var=False, axis=0)

# p-value correction with Holm-Bonferroni correction 
p_vals_cor = multi.multipletests(p_vals, method='holm', alpha=0.05)

# Map results back onto the full surface (masking out the medial wall)
t_stats_final = np.zeros(len(c.cortex_mask))
t_stats_final[cortex_mask_index] = t_test_stats

p_vals_final = np.ones(len(c.cortex_mask))
p_vals_final[cortex_mask_index] = p_vals_cor[1]

# Final p-value mask
p_vals_mask = p_vals_final < 0.05

# t-test plot with p-value overlay
plot_surf(
    surf[0],
    surf[1],
    overlay=t_stats_final,
    mask=~c.cortex_mask,
    pvals=p_vals_mask,
    cmap='viridis',
    vmin=0,
    vmax=10,
    cmap_label=r"$t$-test",
    filename='ipsi_blur_asym',
)

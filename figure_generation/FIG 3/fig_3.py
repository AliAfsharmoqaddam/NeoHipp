import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
import matplotlib as mpl
import potpourri3d as pp3d
from matplotlib_surface_plotting import plot_surf
from brainspace.null_models import SpinPermutations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dominance_analysis import Dominance

from meld_classifier.meld_cohort import MeldCohort, MeldSubject

# ===========================================================================
# Load surface maps
# ===========================================================================
fmri_left_data = nb.load('/home/meldstudent/Documents/RDS_NeoHipp/scripts/Files/hippo_fmri_data_L.fsaverage_sym_correct.func.gii').darrays[0].data
myelin_left_data = nb.load('/home/meldstudent/Documents/RDS_NeoHipp/scripts/Files/myelin_dev_months_new.IBA.lh.func.gii').darrays[0].data
gene_left_data = nb.load('/home/meldstudent/Documents/RDS_NeoHipp/scripts/Files/spearman_hippocamp_surface.lh.fsaverage_sym.func.gii').darrays[0].data

df = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/altered_info5_with_new_name.csv')
aparc = nb.freesurfer.read_annot('/home/meldstudent/Downloads/lh.HCP-MMP1_sym.annot')
surf = nb.freesurfer.io.read_geometry('/home/meldstudent/Documents/RDS_NeoHipp/meld_data/output/fs_outputs/fsaverage_sym/surf/lh.white')

hipp = np.where(aparc[0] == 120)[0]

# ===========================================================================
# Cohort and geodesic distance from hippocampus
# ===========================================================================
c = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_final.hdf5')
total_patients = c.get_subject_ids(site_codes=['H1', 'H11', 'H16', 'H29'], group='patient', lesional_only=False)
patients = [x for x in total_patients if 'hs' in x]

mask = c.cortex_mask

solver = pp3d.MeshHeatMethodDistanceSolver(c.surf['coords'], c.surf['faces'])
geo_d = solver.compute_distance_multisource(hipp)
geo_d[geo_d < 0] = 0

wm_asym_string = '.inter_z.asym.intra_z.combat.on_lh.wm_T1_-1mm.sm5.mgh'


# ===========================================================================
# Patient-averaged WM map
# ===========================================================================
def subject_hemi(subject):
    hemi = df[df.classifier_new_id == subject].hemi.to_string(index=False)
    if hemi in ('lh', 'left'):
        return 'lh'
    if hemi in ('rh', 'right'):
        return 'rh'
    raise ValueError(f"Unrecognised hemisphere '{hemi}' for {subject}")


def patient_mean_full(feature):
    """Per-vertex mean across patients for `feature` (full surface)."""
    vals = [MeldSubject(s, cohort=c).load_feature_values(feature, subject_hemi(s)) for s in patients]
    return np.array(vals).mean(axis=0)


wm_full = patient_mean_full(wm_asym_string)   # full surface (for surface plot)
wm = wm_full[mask]                            # cortex only (for the models)

# Model design matrix (one row per cortical vertex)
df_main = pd.DataFrame({
    'fmri': fmri_left_data[mask],
    'myelin': myelin_left_data[mask],
    'gene': gene_left_data[mask],
    'distance': geo_d[mask],
    'wm': wm,
})


# ===========================================================================
# Brain surface plots
# ===========================================================================
plot_surf(surf[0], surf[1], overlay=wm_full, mask=~c.cortex_mask,
          cmap='seismic_r', cmap_label='WM intensity', vmin=-0.75, vmax=0.75,
          filename='wm3')

plot_surf(surf[0], surf[1], overlay=geo_d, mask=~c.cortex_mask,
          cmap='YlOrRd_r', cmap_label='Distance', vmin=0, vmax=160,
          filename='distance4.png')

plot_surf(surf[0], surf[1], overlay=myelin_left_data, mask=~c.cortex_mask,
          cmap='magma', cmap_label='Myelination', vmin=0, vmax=18,
          filename='myelination5.png')

plot_surf(surf[0], surf[1], overlay=gene_left_data, mask=~c.cortex_mask,
          cmap='seismic', cmap_label='Gene', vmin=-0.7, vmax=0.7,
          filename='gene4.png')

plot_surf(surf[0], surf[1], overlay=fmri_left_data, mask=~c.cortex_mask,
          cmap='seismic', cmap_label='fMRI', vmin=-0.3, vmax=0.3,
          filename='fmri4.png')

# Standalone myelin colorbar (positions 0-100 relabelled 0-18 months)
norm = mpl.colors.Normalize(vmin=0, vmax=100)
cmap = mpl.cm.get_cmap('magma')
fig, ax = plt.subplots(figsize=(1, 0.2))
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
cb.outline.set_linewidth(1)
cb.set_ticks([0, 100])
cb.set_ticklabels([0, 18])
cb.ax.tick_params(length=0, labelsize=12, pad=5)
cb.set_label('Age at 50% myelination\n(months)', fontsize=12)
plt.savefig('colorbar_myelin2.svg', bbox_inches='tight', transparent=True)
plt.show()


# ===========================================================================
# Dominance analysis (relative importance of each predictor for wm)
# ===========================================================================
da = Dominance(data=df_main, target='wm', objective=1)  # objective=1 -> regression (R2)

print(da.incremental_rsquare())
da.plot_incremental_rsquare()
plt.show()
print(da.dominance_stats())
print(da.dominance_level())


# ===========================================================================
# Spin test on the full-model R2
# ---------------------------------------------------------------------------
# Fit OLS with all predictors -> observed R2, then rebuild the null R2
# distribution from spin-permuted versions of the wm map. One-sided p-value
# is the fraction of null R2 >= observed R2.
# ===========================================================================
PREDICTORS = ['myelin', 'distance', 'gene', 'fmri']
DEPENDENT = 'wm'
N_SPINS = 1000
RANDOM_SEED = 42

sphere = nb.freesurfer.read_geometry('/home/meldstudent/Programmes/FreeSurfer7.2.0/freesurfer/subjects/fsaverage_sym/surf/lh.sphere')
coords = sphere[0][mask]

y = df_main[DEPENDENT].values
X = df_main[PREDICTORS].values


def full_model_r2(y_vec, X_mat):
    model = LinearRegression().fit(X_mat, y_vec)
    return r2_score(y_vec, model.predict(X_mat))


observed_r2 = full_model_r2(y, X)

sp = SpinPermutations(n_rep=N_SPINS, random_state=RANDOM_SEED)
sp.fit(coords)
wm_spun = sp.randomize(y)   # (N_SPINS, n_vertices)

null_r2 = np.array([full_model_r2(wm_spun[i], X) for i in range(N_SPINS)])

p_value = max(np.mean(null_r2 >= observed_r2), 1.0 / N_SPINS)

print("\nSpin-test results")
print(f"  Observed R2  : {observed_r2:.6f}")
print(f"  Null mean R2 : {null_r2.mean():.6f}")
print(f"  Null std R2  : {null_r2.std():.6f}")
print(f"  p-value      : {p_value:.4f}  (one-sided, {N_SPINS} spins)")

# Save outputs
pd.Series({
    'observed_r2': observed_r2,
    'null_mean_r2': null_r2.mean(),
    'null_std_r2': null_r2.std(),
    'p_value': p_value,
    'n_spins': N_SPINS,
}).to_csv('spin_test_results.csv', header=['value'])
pd.Series(null_r2, name='null_r2').to_csv('null_r2_distribution.csv', index=False)

# Null distribution plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(null_r2, bins=40, color='#4C72B0', alpha=0.75, edgecolor='white', label='Null R² (spins)')
ax.axvline(observed_r2, color='#DD4949', linewidth=2, label=f'Observed R² = {observed_r2:.4f}')
ax.set_title(f'Spin-test: full-model R²\np = {p_value:.4f} (one-sided, n={N_SPINS})',
             fontsize=12, fontweight='bold')
ax.set_xlabel('R²')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('spin_test_plot.png', dpi=150, bbox_inches='tight')
plt.show()

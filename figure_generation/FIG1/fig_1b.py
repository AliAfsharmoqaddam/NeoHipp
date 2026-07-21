import numpy as np
import nibabel as nb
from matplotlib_surface_plotting import plot_surf
from meld_classifier.meld_cohort import MeldCohort

# Load Glasser atlas (symmetric)
glasser_atlas = '/home/meldstudent/Downloads/lh.HCP-MMP1_sym.annot'
aparc = nb.freesurfer.read_annot(glasser_atlas)

# Load white surface
white_surf = '/home/meldstudent/Documents/RDS_NeoHipp/meld_data/output/fs_outputs/fsaverage_sym/surf/lh.white'
surf = nb.freesurfer.io.read_geometry(white_surf)

# Load cohort (only for the cortex mask)
DATASET_FILE = '/home/meldstudent/Documents/RDS_NeoHipp/final_dataset_no_fcd.csv'
c = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_h16_final.hdf5', dataset=DATASET_FILE)

# Parcellation and its unique regions
atlas = aparc[0]
rois = sorted(set(atlas))

# Assign a unique grey shade to each region
grey_shades = np.linspace(0.4, 0.95, len(rois))
label_atlas = {roi: (shade, shade, shade, 1) for roi, shade in zip(rois, grey_shades)}

# Highlight the temporopolar ROI
label_atlas[131] = (0.6, 0.0, 0.0, 1)  # TGd - dark red
label_atlas[172] = (1.0, 0.4, 0.4, 1)  # TGv - light red

# Visualisation
plot_surf(
    surf[0],
    surf[1],
    overlay=np.ones(len(c.cortex_mask)),
    mask=~c.cortex_mask,
    parcel=atlas,
    parcel_cmap=label_atlas,
    filled_parcels=True,
    colorbar=False,
    filename='glasser'
)

import os
import numpy as np
import json
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.data_preprocessing import Preprocess, Feature
from meld_classifier.paths import BASE_PATH, NORM_CONTROLS_PARAMS_FILE, FINAL_SCALING_PARAMS, CLIPPING_PARAMS_FILE
import time

site_codes = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H9",
    "H10",
    "H11",
    "H12",
    "H14",
    "H15",
    "H16",
    "H17",
    "H18",
    "H19",
    "H21",
    "H23",
    "H24",
    "H26",
    "H29"
]

#Generate feature lists
ds_gmfrac_features_to_generate = []
ds_gmdist_features_to_generate = []
ds_wm_features_to_generate = []
hemispheres = ['lh', 'rh']
    
for h in hemispheres:
#GM fractions
    for gmfrac in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        ds_gmfrac_features_to_generate.append((h, gmfrac))
#GM distance
    for gmdist in np.linspace(0,-7,29):
        ds_gmdist_features_to_generate.append((h, gmdist))
#WM distance
    for dwm in [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4]:
        ds_wm_features_to_generate.append((h, dwm))

features = []
for dsf in ds_gmfrac_features_to_generate:
    d = dsf[1]
    feat = f'.on_lh.gm_T1_{d}.mgh'
    if feat not in features:
        features.append(feat)

for dsf in ds_gmdist_features_to_generate:
    d = dsf[1]
    feat = f'.on_lh.gm_T1_{d}mm.mgh'
    if feat not in features:
        features.append(feat)

for dsf in ds_wm_features_to_generate:
    d = dsf[1]
    feat = f'.on_lh.wm_T1_{d}mm.mgh'
    if feat not in features:
        features.append(feat)

# create cohort to smooth
cohort= MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_new.hdf5")

"""get mean and std of all brain for the given cohort and save parameters"""
cohort_ids = cohort.get_subject_ids(group="both")

for feature in features:
    print(feature)
    # Give warning if list of controls empty
    if len(cohort_ids) == 0:
        print("WARNING: there is no subject in this cohort")
    vals_array = []
    included_subj = []
    for id_sub in cohort_ids:
        # create subject object
        subj = MeldSubject(id_sub, cohort=cohort)
        # append data to compute mean and std if feature exist and for FLAIR=0
        if subj.has_features(feature):
            # load feature's value for this subject
            vals_lh = subj.load_feature_values(feature, hemi="lh")
            vals_rh = subj.load_feature_values(feature, hemi="rh")
            vals = np.array(np.hstack([vals_lh[cohort.cortex_mask], vals_rh[cohort.cortex_mask]]))
            if (feature == ".on_lh.sulc.mgh") & (np.mean(vals, axis=0) > 0.2):
                vals = vals / 10
            vals_array.append(vals)
            included_subj.append(id_sub)                
    print("Compute mean and std from {} subject".format(len(included_subj)))
    # get min and max percentile
    vals_array = np.matrix(vals_array)
    min_p = np.percentile(vals_array.flatten(),0.1)
    print(f'min percentile {min_p}')
    print(f'vertices below min : {(vals_array.flatten()<min_p).sum()}')
    max_p = np.percentile(vals_array.flatten(),99.9)
    print(f'max percentile {max_p}')
    print(f'vertices above min : {(vals_array.flatten()>max_p).sum()}')
    # save in json
    data = {}
    data["{}".format(feature)] = {
        "min_percentile": str(min_p),
        "max_percentile": str(max_p),
    }
    # create or re-write json file
    file = CLIPPING_PARAMS_FILE
    if os.path.isfile(file):
        # open json file and get dictionary
        with open(file, "r") as f:
            x = json.loads(f.read())
        # update dictionary with new dataset version
        x.update(data)
    else:
        x = data
    # save dictionary in json file
    with open(file, "w") as outfile:
        json.dump(x, outfile, indent=4)
    print(f"parameters saved in {file}")

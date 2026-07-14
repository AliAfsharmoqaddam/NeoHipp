import numpy as np
import os
from meld_classifier.paths import BASE_PATH, COMBAT_PARAMS_FILE
from meld_classifier.meld_cohort import MeldCohort
from meld_classifier.data_preprocessing import Preprocess

#Define combat file path
combat_file = os.path.join(BASE_PATH, COMBAT_PARAMS_FILE)

#Generate feature names
features = []

for d in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    features.append(f".on_lh.gm_T1_{d}.sm5.mgh")

for d in np.linspace(0,-7,29):
    features.append(f".on_lh.gm_T1_{d}mm.sm5.mgh")

for d in [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4]:
    features.append(f".on_lh.wm_T1_{d}mm.sm5.mgh")

#Main function for ComBat with corresponding .hdf5  
def main():
    c_smooth = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_h16_new.hdf5")
    combat = Preprocess(c_smooth, write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_h16_new.hdf5", data_dir=BASE_PATH)
    for feature in features:
        print(feature)
        combat.combat_whole_cohort(feature, outliers_file=None, combat_params_file=combat_file)

if __name__ == "__main__":
    main()

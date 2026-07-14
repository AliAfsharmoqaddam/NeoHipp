import numpy as np
from meld_classifier.paths import BASE_PATH
from meld_classifier.meld_cohort import MeldCohort
from meld_classifier.data_preprocessing import Preprocess

#Generate feature names 
features = []

for d in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    features.append(f".combat.on_lh.gm_T1_{d}.sm5.mgh")

for d in np.linspace(0,-7,29):
    features.append(f".combat.on_lh.gm_T1_{d}mm.sm5.mgh")

for d in [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4]:
    features.append(f".combat.on_lh.wm_T1_{d}mm.sm5.mgh")

#Main function for normalisation with corresponding .hdf5
def main():
    c_combat = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_combat_h16_new.hdf5")
    norm = Preprocess(c_combat, write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_h16_final.hdf5", data_dir=BASE_PATH)
    for feature in features:
        print(feature)
        norm.intra_inter_subject(feature)
        norm.asymmetry_subject(feature)

if __name__ == "__main__":
    main()

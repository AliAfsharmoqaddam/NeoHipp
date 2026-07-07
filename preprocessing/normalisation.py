import numpy as np
from meld_classifier.paths import BASE_PATH
from meld_classifier.meld_cohort import MeldCohort
from meld_classifier.data_preprocessing import Preprocess

DATASET_FILE = '/home/meldstudent/Documents/RDS_NeoHipp/final_dataset_no_fcd.csv'

#Generate feature names 
feature_list = []

for d in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    feature_list.append(f".combat.on_lh.gm_T1_{d}.sm5.mgh")

for d in np.linspace(0,-7,29):
    feature_list.append(f".combat.on_lh.gm_T1_{d}mm.sm5.mgh")

for d in [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4]:
    feature_list.append(f".combat.on_lh.wm_T1_{d}mm.sm5.mgh")

#Main function for normalisation 
def main():
    c_combat = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_combat_new_no_fcd.hdf5", dataset=DATASET_FILE)
    norm = Preprocess(
        c_combat,
        write_hdf5_file_root="{site_code}_{group}_featurematrix_combat_final_no_fcd.hdf5",
        data_dir=BASE_PATH,
    )
    for feature in feature_list:
        print(feature)
        norm.intra_inter_subject(feature)
        norm.asymmetry_subject(feature)

if __name__ == "__main__":
    main()

import os
import numpy as np
import nibabel as nb
import h5py
import pandas as pd
from os.path import join as opj
from meld_classifier.paths import BASE_PATH, MELD_DATA_PATH, FS_SUBJECTS_PATH, CLIPPING_PARAMS_FILE
from meld_classifier.meld_cohort import MeldCohort
from meld_classifier.data_preprocessing import Preprocess
from meld_classifier.tools_commands_prints import get_m

DATASET_FILE = '/home/meldstudent/Documents/RDS_NeoHipp/final_dataset_no_fcd.csv'

#Generate feature names 
feature_list = []
features = {}

for d in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    feature_list.append(f".on_lh.gm_T1_{d}.mgh")

for d in np.linspace(0,-7,29):
    feature_list.append(f".on_lh.gm_T1_{d}mm.mgh")

for d in [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4]:
    feature_list.append(f".on_lh.wm_T1_{d}mm.mgh")

#Set size of FWHM kernel 5mm
FWHM = 5

def main():
    c_raw = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_new.hdf5", dataset=DATASET_FILE)
    smoothing = Preprocess(c_raw, write_hdf5_file_root="{site_code}_{group}_featurematrix_smoothed_new_3.hdf5",
                           data_dir=BASE_PATH)

    for feature in np.sort(list(set(features))):
        print(feature)
        smoothing.smooth_data(feature, FWHM, clipping_params=CLIPPING_PARAMS_FILE,
                              outliers_file=None)


if __name__ == "__main__":
    main()



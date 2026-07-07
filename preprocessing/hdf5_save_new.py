import os
import numpy as np
import nibabel as nb
import h5py
import traceback
import pandas as pd
from meld_classifier.tools_commands_prints import get_m

#Script for saving features to hdf5 format

#Paths 
MELD_DATA_PATH = '/home/meldstudent/Documents/RDS_NeoHipp/meld_data'
BASE_PATH = os.path.join(MELD_DATA_PATH, 'output/preprocessed_surf_data')
FS_SUBJECTS_PATH = os.path.join(MELD_DATA_PATH, 'output/fs_outputs')

#List subjects 
all_dirs = os.listdir(FS_SUBJECTS_PATH)
all_subjects = [d for d in all_dirs if d.startswith('MELD')]

#Define depths for each feature 
feature_list = []

for d in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    feature_list.append(f".on_lh.gm_T1_{d}.mgh")

for d in np.linspace(0, -7, 29):
    feature_list.append(f".on_lh.gm_T1_{d}mm.mgh")

for d in [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4]:
    feature_list.append(f".on_lh.wm_T1_{d}mm.mgh")

#Functions borrowed from MELD classifier 
def import_mgh(filename):
    """ import mgh file using nibabel. returns flattened data array"""
    mgh_file=nb.load(filename)
    mmap_data=mgh_file.get_fdata()
    array_data=np.ndarray.flatten(mmap_data)
    return array_data;

def get_cp(fs_id):
    cp=fs_id.split('_')[3]
    if cp in ("FCD" , "fcd"):
        c_p='patient'
    elif cp in ("C" , "c"):
        c_p='control'
    else:
        print('subject '+ fs_id + ' cannot be identified as either patient or control...')
        print('Please double check the IDs in the list of subjects')
        c_p='false'
    return c_p

def get_scanner(fs_id):
    sc=fs_id.split('_')[2]
    if sc in ("15T" , "1.5T" , "15t" , "1.5t" ):
        scanner="15T"
    elif sc in ("3T" , "3t" ):
        scanner="3T"
    else:
        print('scanner for subject '+ fs_id + ' cannot be identified as either 1.5T or 3T...')
        print('Please double check the IDs in the list of subjects')
        scanner='false'
    return scanner

def get_sitecode(fs_id):
    site_code=fs_id.split('_')[1]
    if site_code[0] != 'H':
        print('site code from subject id does not fit format "H<num>". please double check')
        site_code='false'
    return site_code
        
def create_dataset_file(subjects, output_path):
    df = pd.DataFrame()
    subjects_id = [subject for subject in subjects]
    df['subject_id'] = subjects_id
    df['split'] = ['test' for subject in subjects]
    df.to_csv(output_path)

def create_training_data_hdf5(subject, subject_dir, output_dir):
    # list features
    features = np.array(feature_list)
    n_vert = 163842
    cortex_label = nb.freesurfer.io.read_label(os.path.join(subject_dir, 'fsaverage_sym/label/lh.cortex.label'))
    medial_wall = np.delete(np.arange(n_vert), cortex_label)
    failed = save_subject(subject, features, medial_wall, subject_dir, output_dir)
    if failed == True:
        print(get_m(f'Features not saved. Something went wrong', subject, 'ERROR'))
        return False
    else:
        print(get_m(f'All features have been extracted and saved in {output_dir}', subject, 'INFO'))


def save_subject(fs_id, features, medial_wall, subject_dir, output_dir=None):
    failed = False
    n_vert = 163842
    # get subject info from id
    c_p = get_cp(fs_id)
    scanner = get_scanner(fs_id)
    site_code = get_sitecode(fs_id)
    print(f" Saving: {fs_id}")
    print(f" Metadata - site: {site_code}, cp: {c_p}, scanner: {scanner}")

    # skip subject if info not available
    if 'false' in (c_p, scanner, site_code):
        print(f"Skipping subject {fs_id}: Invalid metadata (cp={c_p}, scanner={scanner}, site_code={site_code})")
        return True  # skip safely

    hemis = ['lh', 'rh']
    # save feature in hdf5 file
    if output_dir is None:
        output_dir = subject_dir

    #Save into seperate hdf5 - NB: '_featurematrix_new.hdf5'
    hdf5_file = os.path.join(output_dir, site_code + "_" + c_p + "_featurematrix_new.hdf5")
    if hdf5_file is not None:
        with h5py.File(hdf5_file, "a") as f:
            for h in hemis:
                group_path = os.path.join(site_code, scanner, c_p, fs_id, h)
                try:
                    group = f.require_group(group_path)
                except Exception as e:
                    print(f" Failed to create/access group: {group_path}")
                    traceback.print_exc()
                    failed = True
                    continue  # skip writing this feature entirely
                
                for f_name in features:
                    try:
                        feature = import_mgh(os.path.join(subject_dir, fs_id, 'xhemi/surf_meld', h + f_name))
                        feature[medial_wall] = 0
                        dset = group.require_dataset(f_name, shape=(n_vert,), dtype='float32', compression="gzip",
                                                     compression_opts=9)
                        if feature.shape[0] != n_vert:
                            print(f"WARNING: Feature {f_name} has shape {feature.shape}, expected {n_vert}")
                            failed = True
                            continue
                        dset[:] = feature
                    except Exception as e:
                        print(f"ERROR: Failed to write feature {f_name} for subject {fs_id} ({h} hemisphere)")
                        print(f"Path: {os.path.join(subject_dir, fs_id, 'xhemi/surf_meld', h + f_name)}")
                        traceback.print_exc()
                        failed = True

        lesion_name = os.path.join(subject_dir, fs_id, 'xhemi/surf_meld', h + '.on_lh.lesion.mgh')
        if os.path.isfile(lesion_name):
            lesion = import_mgh(lesion_name)
            dset = group.require_dataset('.on_lh.lesion.mgh', shape=(n_vert,), dtype='float32', compression="gzip",
                                         compression_opts=9)
            dset[:] = lesion
    f.close()
    return failed

#Main function to run
def main():
    for subj in all_subjects:
        site_code = get_sitecode(subj)
        c_p = get_cp(subj)
        print(f"Processing subject: {subj}, site: {site_code}, cp: {c_p}")
        create_training_data_hdf5(subj, FS_SUBJECTS_PATH, BASE_PATH)


if __name__ == "__main__":
    main()

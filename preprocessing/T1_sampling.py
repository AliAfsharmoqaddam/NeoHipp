import os
import numpy as np
import subprocess
from subprocess import Popen
import pandas as pd
from meld_classifier.tools_commands_prints import get_m
from meld_classifier.paths import FS_SUBJECTS_PATH

#Function for sampling per patient, at various intracortical/subcortical depth 
def sample_T1_features(subject_id, verbose=False):

    subjects_dir = FS_SUBJECTS_PATH

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

    #Sampling vol2surf for gmfrac
    for dsf in ds_gmfrac_features_to_generate:
        hemi = dsf[0]
        d = dsf[1]
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{hemi}.gm_T1_{d}.mgh"):
            src_path = os.path.join(subjects_dir, subject_id, 'mri/orig.mgz')
            srcreg_path = os.path.join(subjects_dir, subject_id, 'mri/transforms/Identity.dat')
            out_path = os.path.join(subjects_dir, subject_id, f'surf_meld/{hemi}.gm_T1_{d}.mgh')
            command = f"env SUBJECTS_DIR={subjects_dir} mri_vol2surf --mov {src_path} --out {out_path} --hemi {hemi} --projfrac {d} --regheader {subject_id} --surf white"
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False

    #Sampling vol2surf for gmdist
    for dsf in ds_gmdist_features_to_generate:
        hemi = dsf[0]
        d = dsf[1]
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{hemi}.gm_T1_{d}mm.mgh"):
            src_path = os.path.join(subjects_dir, subject_id, 'mri/orig.mgz')
            srcreg_path = os.path.join(subjects_dir, subject_id, 'mri/transforms/Identity.dat')
            out_path = os.path.join(subjects_dir, subject_id, f'surf_meld/{hemi}.gm_T1_{d}mm.mgh')
            command = f"env SUBJECTS_DIR={subjects_dir} mri_vol2surf --mov {src_path} --out {out_path} --hemi {hemi} --projdist {d} --regheader {subject_id} --surf pial"
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False

    #Sampling vol2surf for wm
    for dsf in ds_wm_features_to_generate:
        hemi = dsf[0]
        d = dsf[1]
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{hemi}.wm_T1_{d}mm.mgh"):
            src_path = os.path.join(subjects_dir, subject_id, 'mri/orig.mgz')
            srcreg_path = os.path.join(subjects_dir, subject_id, 'mri/transforms/Identity.dat')
            out_path = os.path.join(subjects_dir, subject_id, f'surf_meld/{hemi}.wm_T1_{d}mm.mgh')
            command = f"env SUBJECTS_DIR={subjects_dir} mri_vol2surf --mov {src_path} --out {out_path} --hemi {hemi} --projdist {d} --regheader {subject_id} --surf white"
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False

#List all subjects (including H16 FCD) and run function 
participants = pd.read_csv('/home/meldstudent/Documents/RDS_NeoHipp/final_dataset_h16_fcd.csv').subject_id
for subj in participants:
    sample_T1_features(subj, verbose=True)

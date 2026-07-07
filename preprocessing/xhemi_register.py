import os 
import numpy as np
import subprocess
from subprocess import Popen
from meld_classifier.tools_commands_prints import get_m

#Function to register individual cortical surface to fsaverage_sym 
def xhemi_register(subject_id, verbose=False):
    
    subjects_dir = '/home/meldstudent/Documents/RDS_NeoHipp/meld_data/output/fs_outputs/'

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

    #xhemi reg for gmfrac
    for dsf in ds_gmfrac_features_to_generate:
        hemi = dsf[0]
        d = dsf[1]
    #Set file paths
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/{hemi}.on_lh.gm_T1_{d}.mgh"):
            src_path = os.path.join(subjects_dir, subject_id, f'surf_meld/{hemi}.gm_T1_{d}.mgh')
            trgreg_path = '/home/meldstudent/Programmes/FreeSurfer7.2.0/freesurfer/subjects/fsaverage_sym/surf/lh.sphere.reg'
            trg_path = os.path.join(subjects_dir, subject_id, f'xhemi/surf_meld/{hemi}.on_lh.gm_T1_{d}.mgh')
        #Find corresponding reg file
            if hemi =='lh':
                srcreg_path = os.path.join(subjects_dir, subject_id, 'surf/lh.fsaverage_sym.sphere.reg')
            else:
                srcreg_path = os.path.join(subjects_dir, subject_id, 'xhemi/surf/lh.fsaverage_sym.sphere.reg') 
        #Apply reg
            command = f'mris_apply_reg --src {src_path} --trg {trg_path} --streg {srcreg_path} {trgreg_path}'
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False

    #xhemi reg for gmdist
    for dsf in ds_gmdist_features_to_generate:
        hemi = dsf[0]
        d = dsf[1]
    #Set file paths
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/{hemi}.on_lh.gm_T1_{d}mm.mgh"):
            src_path = os.path.join(subjects_dir, subject_id, f'surf_meld/{hemi}.gm_T1_{d}mm.mgh')
            trgreg_path = '/home/meldstudent/Programmes/FreeSurfer7.2.0/freesurfer/subjects/fsaverage_sym/surf/lh.sphere.reg'
            trg_path = os.path.join(subjects_dir, subject_id, f'xhemi/surf_meld/{hemi}.on_lh.gm_T1_{d}mm.mgh')
        #Find corresponding reg file
            if hemi == 'lh':
                srcreg_path = os.path.join(subjects_dir, subject_id, 'surf/lh.fsaverage_sym.sphere.reg')
            else:
                srcreg_path = os.path.join(subjects_dir, subject_id, 'xhemi/surf/lh.fsaverage_sym.sphere.reg') 
        #Apply reg
            command = f'mris_apply_reg --src {src_path} --trg {trg_path} --streg {srcreg_path} {trgreg_path}'
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False

    #xhemi reg for wm
    for dsf in ds_wm_features_to_generate:
        hemi = dsf[0]
        d = dsf[1]
    #Set file paths
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/{hemi}.on_lh.wm_T1_{d}mm.mgh"):
            src_path = os.path.join(subjects_dir, subject_id, f'surf_meld/{hemi}.wm_T1_{d}mm.mgh')
            trgreg_path = '/home/meldstudent/Programmes/FreeSurfer7.2.0/freesurfer/subjects/fsaverage_sym/surf/lh.sphere.reg'
            trg_path = os.path.join(subjects_dir, subject_id, f'xhemi/surf_meld/{hemi}.on_lh.wm_T1_{d}mm.mgh')
        #Find corresponding reg file
            if hemi == 'lh':
                srcreg_path = os.path.join(subjects_dir, subject_id, 'surf/lh.fsaverage_sym.sphere.reg')
            else:
                srcreg_path = os.path.join(subjects_dir, subject_id, 'xhemi/surf/lh.fsaverage_sym.sphere.reg') 
        #Apply reg
            command = f'mris_apply_reg --src {src_path} --trg {trg_path} --streg {srcreg_path} {trgreg_path}'
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False

subjects_dir = '/home/meldstudent/Documents/RDS_NeoHipp/meld_data/output/fs_outputs/'

#List all subjects and run function 
dirs = os.listdir(subjects_dir)
all_subjects = []
for subj in dirs:
    if subj.startswith('MELD'):
        xhemi_register(subj, verbose=True)
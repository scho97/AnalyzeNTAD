"""Source reconstruct the NTAD data

"""

# Install dependencies
import os
import glob
import mne
import numpy as np
import pandas as pd
import nibabel as nib
from sys import argv
from scipy import spatial
from osl import utils, source_recon
from dask.distributed import Client

# Define modality 
if len(argv) != 2:
    raise ValueError("need to pass one argument: modality (e.g., python script.py eeg)")
modality = argv[1]
print(f"[INFO] Modality: {modality.upper()}")

# Set directories
BASE_DIR = "/ohba/pi/mwoolrich/scho/NTAD"
PREPROC_DIR = os.path.join(BASE_DIR, f"preproc/{modality}")
SRC_DIR  = os.path.join(BASE_DIR, f"src/{modality}")

# Define file formats
preproc_file_format = os.path.join(PREPROC_DIR, "{0}_resting_close_bl_raw_tsss/{0}_resting_close_bl_tsss_preproc_raw.fif")
smri_file_format = "/ohba/pi/mwoolrich/datasets/NTAD/{0}/{1}/{1}_Baseline_MR/analysis/{1}_Baseline.nii"

# Define local functions
def get_site_from_subject(subject_id):
    """Determine the site based on the subject ID."""
    if int(subject_id[1]) in [1, 3]:
        return "cambridge"
    return "oxford"

def fix_smri_sform_code(subject_id, smri_file_path):
    """Fix sMRI files with different sform codes."""
    FIXED_SUFFIX = "_fixed.nii"
    if subject_id in ["C1019", "C1022", "P1023", "P1027", "P1053", "P1066", "P1070"]:
        fixed_path = smri_file_path.replace(".nii", FIXED_SUFFIX)
        if not os.path.exists(fixed_path):
            smri = nib.load(smri_file_path)
            smri.header["sform_code"] = 1
            smri.to_filename(fixed_path)
        return fixed_path
    return smri_file_path

def headsize_from_fids(fids):
    """Get rough estimate of the subject head size."""
    # Input is a [3 x 3] array of [observations x dimensions].
    # Reference: https://en.wikipedia.org/wiki/Heron%27s_formula
    dists = spatial.distance.pdist(fids)
    semi_perimeter = np.sum(dists) / 2
    area = np.sqrt(semi_perimeter * np.prod(semi_perimeter - dists))
    return area

def get_device_fids(raw):
    """Get fiducials in the M/EEG device space."""

    # Put fiducials in device space
    head_fids = mne.viz._3d._fiducial_coords(raw.info['dig'])
    head_fids = np.vstack(([0, 0, 0], head_fids))
    fid_space = raw.info['dig'][0]['coord_frame']
    assert(fid_space == 4)  # Ensure we have FIFFV_COORD_HEAD coords

    # Get device to head transform and inverse
    dev2head = raw.info['dev_head_t']
    head2dev = mne.transforms.invert_transform(dev2head)
    assert(head2dev['from'] == 4)
    assert(head2dev['to'] == 1)

    # Apply transformation to get fids in device space
    device_fids = mne.transforms.apply_trans(head2dev, head_fids)

    return device_fids

def get_matched_vect(df, subject_ids, key):
    """Match subject IDs to DataFrame."""
    out = []
    for ii in range(len(subject_ids)):
        sid = subject_ids[ii]
        inds = np.where(np.array([row_id.find(sid) for row_id in df['ID'].values]) > -1)[0]
        if len(inds) > 0:
            out.append(df[key].iloc[inds[0]])
        else:
            out.append(np.nan)
    return out

def find_replacement_anats(df, subject_ids_no_mri, meeg_files_no_mri, subject_ids, meeg_files, smri_files):
    """
    Find an sMRI file of another subject for a subject 
    without it based on sex and head size
    """
    # Validation
    if (len(subject_ids) != len(meeg_files)) or (len(subject_ids) != len(smri_files)):
        raise ValueError("subject IDs, preprocessed M/EEG file paths, " + 
                         "and sMRI file paths should have the same length.")

    # Get subject headsize
    headsize = [] # for subjects with sMRI data
    for file in meeg_files:
        raw = mne.io.read_raw(file)
        headsize.append(headsize_from_fids(get_device_fids(raw)))
    
    headsize_no_mri = [] # for subjects withotu sMRI data
    for file in meeg_files_no_mri:
        raw = mne.io.read_raw(file)
        headsize_no_mri.append(headsize_from_fids(get_device_fids(raw)))

    # Get subject sex
    sex = get_matched_vect(df, subject_ids, "Sex (1=female, 2=male)")
    sex_no_mri = get_matched_vect(df, subject_ids_no_mri, "Sex (1=female, 2=male)")
    replaced_subjects = []
    replaced_smri_files = []
    for subject, hs, sx in zip(subject_ids_no_mri, headsize_no_mri, sex_no_mri):
        difference = np.min(np.abs(hs - np.array(headsize)[sx == sex]))
        replacement_idx = np.argmin(np.abs(np.abs(hs - headsize) - difference))
        replaced_smri_files.append(smri_files[replacement_idx])
        replaced_subject = subject_ids[replacement_idx]
        print(f"{subject} sMRI file: replaced to use data of {replaced_subject}")
        replaced_subjects.append(replaced_subject)

    return replaced_subjects, replaced_smri_files

# Get file paths
preproc_files = sorted(glob.glob(os.path.join(PREPROC_DIR, "*/*_preproc_raw.fif")))
subjects = [file.split('/')[-1].split('_')[0] for file in preproc_files]
print(f"Number of preprocessed subjects: {len(subjects)}")
smri_files = []
for subject in subjects:
    site = get_site_from_subject(subject)
    smri_files.append(smri_file_format.format(site, subject))

# Identify subjects whose sMRI files are problematic
subject_ids_problem = []
for file_path in smri_files:
    subject = file_path.split("/")[7]
    if os.path.exists(file_path) == 0:
        subject_ids_problem.append(subject) # subjects without sMRI data
    if modality == "eeg":
        if subject in ["C2006", "C2019", "P1004", "P1005", "P1060", "P2002", "P2034"]:
            subject_ids_problem.append(subject) # subjects with unstable sMRI data (only for eeg)
print(f"Subjects with sMRI problems (n={len(subject_ids_problem)}): {subject_ids_problem}")

# Get file paths of valid subjects
valid_subject_ids = [id for id in subjects if id not in set(subject_ids_problem)]
valid_preproc_files = []
valid_smri_files = []
for subject in valid_subject_ids:
    valid_preproc_files.append(preproc_file_format.format(subject))
    site = get_site_from_subject(subject)
    smri_file_path = smri_file_format.format(site, subject)
    valid_smri_file_path = fix_smri_sform_code(subject, smri_file_path) # fix sMRI file paths if necessary
    valid_smri_files.append(valid_smri_file_path)

# Validate the order of subjects
if valid_subject_ids != [file.split('/')[7] for file in valid_smri_files]:
    raise ValueError("subject IDs and sMRI files not correctly aligned.")

# Get meta data
META_DIR = "/ohba/pi/mwoolrich/datasets/NTAD/all_data_info.csv"
df_meta = pd.read_csv(META_DIR, sep=",")

# Assign new sMRI files to subjects with problematic sMRI data
initial_subject_ids = valid_subject_ids.copy()
initial_preproc_files = valid_preproc_files.copy()
initial_smri_files = valid_smri_files.copy()
for subject in subject_ids_problem:
    valid_preproc_files.append(preproc_file_format.format(subject))
    similar_subject, _ = find_replacement_anats(
        df_meta,
        [subject],
        [preproc_file_format.format(subject)],
        initial_subject_ids,
        initial_preproc_files,
        initial_smri_files,
    )
    if len(similar_subject) == 1:
        similar_subject = similar_subject[0]
    else:
        print(similar_subject)
        raise ValueError("there should be one subject available to replace the current subject.")
    site = get_site_from_subject(similar_subject)
    smri_file_path = smri_file_format.format(site, similar_subject)
    valid_smri_file_path = fix_smri_sform_code(similar_subject, smri_file_path)
    valid_smri_files.append(valid_smri_file_path)
valid_subject_ids += subject_ids_problem

# Validate the size of data files
if len(valid_subject_ids) != len(valid_preproc_files):
    raise ValueError("subject IDs and preprocessed files must have same length.")
if len(valid_subject_ids) != len(valid_smri_files):
    raise ValueError("subject IDs and sMRI files must have same length.")

# Print available subjects
print(f"Number of available subjects: {len(valid_subject_ids)}")

# Define custom functions
def fix_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Get criteria dividing nose and neck points
    criteria = np.mean([np.mean([lpa[1], rpa[1]]), nas[1]]) * 0.90

    # Remove headshape points more than 15 cm away
    distances = np.sqrt((nas[2] - hs[2]) ** 2)
    keep = distances < 150
    hs = hs[:, keep]

    # Remove anything below rpa (except for nose points)
    keep = ~np.logical_and(hs[2] < rpa[2], hs[1] < criteria)
    hs = hs[:, keep]

    # Remove anything below lpa (except for nose points)
    keep = ~np.logical_and(hs[2] < lpa [2], hs[1] < criteria)
    hs = hs[:, keep]

    # Remove anything outside of rpa
    distances = (rpa[0] - hs[0])
    keep = distances > -12
    hs = hs[:, keep]

    # Remove anything outside of lpa
    distances = lpa[0] - hs[0]
    keep = distances < 12
    hs = hs[:, keep]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

# Configure pipeline
if modality == "meg":
    config = """
        source_recon:
            - extract_fiducials_from_fif:
                include_eeg_as_headshape: false
            - fix_headshape_points: {}
            - compute_surfaces_coregister_and_forward_model:
                include_nose: true
                use_nose: true
                use_headshape: true
                model: Single Layer
            - beamform_and_parcellate:
                freq_range: [1, 45]
                chantypes: [mag, grad]
                rank: {meg: 60}
                parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
                method: spatial_basis
                orthogonalisation: symmetric
    """
elif modality == "eeg":
    config = """
        source_recon:
            - extract_fiducials_from_fif:
                include_eeg_as_headshape: true
            - fix_headshape_points: {}
            - compute_surfaces_coregister_and_forward_model:
                include_nose: true
                use_nose: true
                use_headshape: true
                model: Triple Layer
                eeg: true
            - beamform_and_parcellate:
                freq_range: [1, 45]
                chantypes: eeg
                rank: {eeg: 45}
                parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
                method: spatial_basis
                orthogonalisation: symmetric
    """

# SOURCE RECONSTRUCT DATA
if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Set up FSL
    source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")

    # Set up parallel processing
    client = Client(n_workers=3, threads_per_worker=1)
    
    # Initiate source reconstruction
    print("Running from {} to {} ...".format(valid_subject_ids[0], valid_subject_ids[-1]))
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=valid_subject_ids,
        preproc_files=valid_preproc_files,
        smri_files=valid_smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )

print("Source reconstruction complete.")
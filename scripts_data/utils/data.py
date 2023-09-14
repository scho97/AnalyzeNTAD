"""Functions to handle and inspect data

"""

import os
from glob import glob

def get_subject_ids(data_dir, modality):
    """Extract subject IDs from files in the given directory.

    Parameters
    ----------
    data_dir : str
        Directory path that contains all the subject data.
    modality : str
        Type of modality. Should be either "eeg" or "meg".

    Returns
    -------
    subject_ids : list of str
        List of subject IDs.
    n_subjects : int
        Number of subjects.   
    """

    # Validation
    data_type = os.path.basename(data_dir)
    if data_type not in ["preproc", "src"]:
        raise ValueError("data_type should be either 'preproc' or 'src'.")

    # Get subject IDs
    if data_type == "preproc":
        file_name = os.path.join(data_dir, f"{modality}/*_tsss/*_preproc_raw.fif")
        files = sorted(glob(file_name))
        subject_ids = [file.split("/")[-1][:5] for file in files]
    elif data_type == "src":
        file_name = os.path.join(data_dir, f"{modality}/*/sflip_parc-raw.fif")
        files = sorted(glob(file_name))
        subject_ids = [file.split("/")[-2] for file in files]

    # Get the number of subjects
    n_subjects = len(subject_ids)

    return subject_ids, n_subjects
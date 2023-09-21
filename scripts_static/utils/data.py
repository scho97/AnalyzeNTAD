"""Functions to handle and inspect data

"""

import os
import numpy as np
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

def load_group_information(subject_ids):
    """Get subject indices in each group (amyloid negative controls
       vs. amyloid positive patients with MCI or AD).

    Parameters
    ----------
    subject_ids : list of str
        List of subject IDs.

    Returns
    -------
    an_idx : list of int
        Subject indices for the amyloid negative group.
    ap_idx : list of int
        Subject indices for the amyloid positive group.
    """

    group_labels = [id[0] for id in subject_ids]
    an_idx, ap_idx = [], [] # subject indices in each group
    for i, lbl in enumerate(group_labels):
        if lbl == "C":
            an_idx.append(i)
        elif lbl == "P":
            ap_idx.append(i)

    return an_idx, ap_idx

def load_site_information(subject_ids):
    """Get recording sites for each subject. The site should be 
       either Oxford or Cambridge.

    Parameters
    ----------
    subject_ids : list of str
        List of subject IDs.

    Returns
    -------
    sites : np.ndarray
        Array marking recording sites for each subject.
        Cambridge is marked by 0, and Oxford is marked by 1.
    """

    sites = np.zeros(len(subject_ids),)
    for n, id in enumerate(subject_ids):
        if int(id[1]) in [1, 3]: # site: Cambridge
            sites[n] = 0
        if int(id[1]) == 2: # site: Oxford
            sites[n] = 1
    
    return sites

def load_scanner_information(subject_ids, df, modality="meg"):
    """Get measurement devices for each subject.
       Available MEG scanners:
            - Electra VectorView system (0)
            - MEGIN TRIUX Neo system (1)
       Available EEG scanners:
            - 70-channel EasyCap (0)
            - 64-channel EasyCap (1)
            - 60-channel EasyCap (2)

    Parameters
    ----------
    subject_ids : list of str
        List of subject IDs.
    df : pandas.DataFrame
        Dataframe containing scanner information.
    modality : str
        Type of a neuroimaging modality.

    Returns
    -------
    scanners : np.ndarray
        Array of scanner devices for each subject.
    """

    # Validation
    if modality not in ["meg", "eeg"]:
        raise ValueError("modality should be either MEG or EEG.")

    # Get scanner information
    scanners = np.zeros(len(subject_ids),)
    for n, id in enumerate(subject_ids):
        device_name = df[f"{modality}_device"][df["ID"] == id].values[0]
        if modality == "meg":
            if device_name == "VectorView":
                scanners[n] = 0
            elif device_name == "TRIUX":
                scanners[n] = 1
            else:
                raise ValueError(f"unexpected scanner device reported for {id}.")
        if modality == "eeg":
            if device_name == "EasyCap70":
                scanners[n] = 0
            elif device_name == "EasyCap64":
                scanners[n] = 1
            elif device_name == "EasyCap60":
                scanners[n] = 2
            else:
                raise ValueError(f"unexpected scanner device reported for {id}.")
    
    return scanners
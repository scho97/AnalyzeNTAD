"""Functions to handle and inspect data

"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

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
       vs. amyloid positive patients with MCI or AD)

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

def load_order(run_dir, modality):
    """Extract a state/mode order of a given run written on the
       excel sheet. This order can be used to match the states/
       modes of a run to those of the reference run.

    Parameters
    ----------
    run_dir : str
        Name of the directory containing the model run (e.g., "run6_hmm").
    modality : str
        Type of the modality. Should be either "eeg" or "meg".

    Returns
    -------
    order : list of int
        Order of the states/modes matched to the reference run.
        Shape is (n_states,). If there is no change in order, None is
        returned.
    """

    # Define model type and run ID
    model_type = run_dir.split("_")[-1]
    run_id = int(run_dir.split("_")[0][3:])
    
    # Get list of orders
    BASE_DIR = "/home/scho/AnalyzeNTAD"
    df = pd.read_excel(os.path.join(BASE_DIR, "scripts_dynamic/run_orders.xlsx"))

    # Extract the order of a given run
    index = np.logical_and.reduce((
        df.Modality == modality,
        df.Model == model_type,
        df.Run == run_id,
    ))
    order = df.Order[index].values[0]
    convert_to_list = lambda x: [int(n) for n in x[1:-1].split(',')]
    order = convert_to_list(order)
    if order == list(np.arange(8)):
        order = None
    
    return order

def load_outlier(run_dir, modality):
    """Extract indices of subject outliers for a given run written 
       on the excel sheet. These indices can be used to exclude the 
       subjects from the analyses. Note that outliers are only relevant 
       for EEG DyNeMo model runs.

    Parameters
    ----------
    run_dir : str
        Name of the directory containing the model run (e.g., "run6_hmm").
    modality : str
        Type of the modality. Should be either "eeg" or "meg".

    Returns
    -------
    outlier : list of int
        Subject outliers in the given model run. Shape is (n_outliers,).
    """

    # Define model type and run ID
    model_type = run_dir.split("_")[-1]
    run_id = int(run_dir.split("_")[0][3:])

    # Validation
    if (model_type != "dynemo") or (modality != "eeg"):
        raise ValueError("outlier detection is relevant only for EEG DyNeMo runs.")
    
    # Get list of subject outlier index
    BASE_DIR = "/home/scho/AnalyzeNTAD"
    df = pd.read_excel(os.path.join(BASE_DIR, "scripts_dynamic/run_outliers.xlsx"))

    # Extract the outliers of a given run
    index = np.logical_and.reduce((
        df.Modality == modality,
        df.Model == model_type,
        df.Run == run_id,
    ))
    outlier = df.Outlier[index].values[0]
    convert_to_list = lambda x: [int(n) for n in x[1:-1].split(',')]
    outlier = convert_to_list(outlier)

    return outlier

def get_dynemo_mtc(alpha, Fs, data_dir, plot_mtc=False):
    """Load or compute GMM-fitted DyNeMo mode time courses.

    Parameters
    ----------
    alpha : np.ndarray or list of np.ndarray
        Inferred mode mixing coefficients. Shape must be (n_samples, n_modes)
        or (n_subjects, n_samples, n_modes).
    Fs : int
        Sampling frequency of the training data.
    data_dir : str
        Data directory where a model run is stored.
    plot_mtc : bool
        Whether to plot example segments of mode time courses.
        Defaults to False.

    Returns
    -------
    mtc : np.ndarray or list of np.ndarray
        GMM time courses with binary entries.
    """

    # Number of modes
    if isinstance(alpha, list):
        n_modes = alpha[0].shape[1]
    else: n_modes = alpha.shape[1]

    # Binarize DyNeMo mixing coefficients
    mtc_path = os.path.join(data_dir, "model/results/dynemo_mtc.pkl")
    if os.path.exists(mtc_path):
        print("DyNeMo mode time courses already exist. The saved file will be loaded.")
        with open(mtc_path, "rb") as input_path:
            mtc = pickle.load(input_path)
    else:
        mtc = modes.gmm_time_courses(
            alpha,
            logit_transform=True,
            standardize=True,
            filename=os.path.join(data_dir, "analysis", "gmm_time_courses_.png"),
            plot_kwargs={
                "x_label": "Standardised logit",
                "y_label": "Probability",
            },
        )
        with open(mtc_path, "wb") as output_path:
            pickle.dump(mtc, output_path)
        
        # Plot mode activation time courses
        if plot_mtc:
            for i in range(n_modes):
                # Get the first 5s of each mode activation time course of the first subject
                if isinstance(mtc, list):
                    mode_activation = mtc[0][:5 * Fs, i][..., np.newaxis]
                else:
                    mode_activation = mtc[:5 * Fs, i][..., np.newaxis]
                fig, ax = plotting.plot_alpha(
                    mode_activation,
                    sampling_frequency=Fs,
                )
                for axis in ax:
                    axis.tick_params(
                        axis='y',
                        which='both',
                        left=False,
                        labelleft=False,
                    ) # remove y-axis ticks
                fig.axes[-1].remove() # remove colorbar
                save_path = os.path.join(data_dir, "analysis", f"gmm_mode_activation_{i}.png")
                plotting.save(fig, filename=save_path, tight_layout=False)
                plt.close()

    return mtc

def divide_psd_by_group(psd, ts, group_idx):
    """Separate PSD arrays into groups.

    Parameters
    ----------
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape is (n_subjects,
        n_states, n_channels, n_freqs).
    ts : list of np.ndarray
        Time series data for each subject. Shape must be (n_subjects, n_samples,
        n_channels).
    group_idx : list of lists
        List containing indices of subjects in each group.

    Returns
    -------
    psd_group1 : np.ndarray
        Power spectra of the first group. Shape is (n_subjects, n_states,
        n_channels, n_freqs).
    psd_group2 : np.ndarray
        Power spectra of the second group. Shape is (n_subjects, n_states,
        n_channels, n_freqs).
    w_group1 : np.ndarray
        Weight for each subject-specific PSD in the first group.
        Shape is (n_subjects,).
    w_group2 : np.ndarray
        Weight for each subject-specific PSD in the second group.
        Shape is (n_subjects,).
    """

    # Get index of participants in each group
    group1_idx, group2_idx = group_idx[0], group_idx[1]

    # Get PSD data of each group
    psd_group1 = np.array([psd[idx] for idx in group1_idx])
    psd_group2 = np.array([psd[idx] for idx in group2_idx])

    # Get time series data of each group
    ts_group1 = [ts[idx] for idx in group1_idx]
    ts_group2 = [ts[idx] for idx in group2_idx]

    # Get time series sample numbers subject-wise
    n_samples_group1 = [ts.shape[0] for ts in ts_group1]
    n_samples_group2 = [ts.shape[0] for ts in ts_group2]

    # Recalculate weights for each age group
    w_group1 = np.array(n_samples_group1) / np.sum(n_samples_group1)
    w_group2 = np.array(n_samples_group2) / np.sum(n_samples_group2)
    
    return psd_group1, psd_group2, w_group1, w_group2
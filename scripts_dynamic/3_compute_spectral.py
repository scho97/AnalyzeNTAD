"""Perform post-hoc spectral analysis using power spectra and coherences

"""

# Set up dependencies
import os
import pickle
import warnings
import numpy as np
from sys import argv
from osl_dynamics import analysis
from osl_dynamics.inference import modes
from utils import visualize
from utils.analysis import get_psd_coh
from utils.data import (load_order,
                        load_group_information,
                        load_outlier,
                        get_dynemo_mtc)


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("Step 1 - Setting up ...")

    # Set hyperparameters
    if len(argv) != 4:
        print("Need to pass three arguments: modality, model type, and run ID (e.g., python script.py eeg hmm 0)")
    modality = argv[1]
    model_type = argv[2]
    run_id = argv[3]
    print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()} | Run: run{run_id}_{model_type}")

    # Get state/mode orders for the specified run
    run_dir = f"run{run_id}_{model_type}"
    order = load_order(run_dir, modality)

    # Define training hyperparameters
    Fs = 250 # sampling frequency
    n_channels = 80 # number of channels
    if model_type == "hmm":
        n_class = 8 # number of states
        seq_len = 800 # sequence length for HMM training
    if model_type == "dynemo":
        n_class = 8 # number of modes
        seq_len = 200 # sequence length for DyNeMo training
    if modality == "eeg":
        data_name = "ntad_eeg"
    elif modality == "meg":
        data_name = "ntad_meg"

    # Set parcellation file paths
    mask_file = "MNI152_T1_8mm_brain.nii.gz"
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Set up directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results"
    DATA_DIR = os.path.join(BASE_DIR, f"dynamic/{data_name}/{model_type}/{run_dir}")

    # Load data
    with open(os.path.join(DATA_DIR, f"model/results/{data_name}_{model_type}.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    alpha = data["alpha"]
    ts = data["training_time_series"]

    # Load group information
    subject_ids = data["subject_ids"]
    n_subjects = len(subject_ids)
    an_idx, ap_idx = load_group_information(subject_ids)
    print("Total {} subjects | AN: {} | AP: {}".format(
        n_subjects, len(an_idx), len(ap_idx))
    )

    # Define group assignment
    group_assignments = np.zeros((n_subjects,))
    group_assignments[ap_idx] = 1 # amyloid positive (w/ MCI, AD)
    group_assignments[an_idx] = 2 # amyloid negative (controls)

    # Validation
    if len(alpha) != n_subjects:
        warnings.warn(f"The length of alphas does not match the number of subjects. n_subjects reset to {len(alpha)}.")
        n_subjects = len(alpha)

    # ----------------- [2] ------------------- #
    #      Preprocess inferred parameters       #
    # ----------------------------------------- #
    print("Step 2 - Preparing state/mode time courses ...")

    # Reorder states or modes if necessary
    if order is not None:
        print(f"Reordering {modality.upper()} state/mode time courses ...")
        print(f"\tOrder: {order}")
        alpha = [a[:, order] for a in alpha] # dim: n_subjects x n_samples x n_modes

    # Get HMM state time courses
    if model_type == "hmm":
        btc = modes.argmax_time_courses(alpha)
    
    # Get DyNeMo mode activation time courses
    if model_type == "dynemo":
        btc = get_dynemo_mtc(alpha, Fs, data_dir=DATA_DIR)

    # --------- [3] --------- #
    #      Load Spectra       #
    # ----------------------- #
    print("Step 3 - Loading spectral information ...")
    
    # Set the number of CPUs to use for parallel processing
    n_jobs = 16

    # Calculate subject-specific PSDs and coherences
    if model_type == "hmm":
        print("Computing HMM multitaper spectra ...")
        f, psd, coh, w = get_psd_coh(
            ts, btc, Fs,
            calc_type="mtp",
            save_dir=DATA_DIR,
            n_jobs=n_jobs,
        )
    if model_type == "dynemo":
        print("Computing DyNeMo glm spectra ...")
        f, psd, coh, w = get_psd_coh(
            ts, alpha, Fs,
            calc_type="glm",
            save_dir=DATA_DIR,
            n_jobs=n_jobs,
        )

    # Exclude specified outliers
    if (modality == "eeg") and (model_type == "dynemo"):
        outlier_idx = load_outlier(run_dir, modality)
        print("Excluding subject outliers ...\n"
              "\tOutlier indices: ", outlier_idx)
        not_olr_idx = np.setdiff1d(np.arange(n_subjects), outlier_idx)
        ts = [ts[idx] for idx in not_olr_idx]
        btc = [btc[idx] for idx in not_olr_idx]
        alpha = [alpha[idx] for idx in not_olr_idx]
        psd = psd[not_olr_idx]
        coh = coh[not_olr_idx]
        print(f"\tPSD shape: {psd.shape} | Coherence shape: {coh.shape}")
        # Reassign group indices
        group_assignments = group_assignments[not_olr_idx]
        subject_ids = [subject_ids[idx] for idx in not_olr_idx]
        n_subjects = len(subject_ids)
        an_idx, ap_idx = load_group_information(subject_ids)
        print("\tTotal {} subjects (after excluding outliers) | AN: {} | AP: {}".format(
            n_subjects, len(an_idx), len(ap_idx),
        ))
        if (np.count_nonzero(group_assignments == 2) != len(an_idx)) or (np.count_nonzero(group_assignments == 1) != len(ap_idx)):
            raise ValueError("group assignments do not match with group information.")

    # Rescale regression coefficients of DyNeMo mode-specific PSDs
    rescale_psd = True
    if rescale_psd and (model_type == "dynemo"):
        print("Rescaling DyNeMo regression coefficients ...")
        psd_rescaled = analysis.spectral.rescale_regression_coefs(
            psd,
            alpha,
            window_length=1000,
            step_size=20,
            n_sub_windows=8,
        )
        print("Complete.")

    # Get fractional occupancies to be used as weights
    fo = modes.fractional_occupancies(btc) # dim: (n_subjects, n_states)
    gfo = np.mean(fo, axis=0)

    # ----------- [4] ------------ #
    #      Spectral analysis       #
    # ---------------------------- #
    print("Step 4 - Analyzing spectral information ...")

    n_test = n_class + 1 # tests repeated over states/modes and static mean

    # Cluster permutation test on PSDs (mean-subtracted)
    if model_type == "hmm":
        input_psd = psd - np.average(psd, axis=1, weights=gfo, keepdims=True)
    if model_type == "dynemo":
        input_psd = psd[:, 0, :, :, :] # use regression coefficients
    # dim: (n_subjects, n_states, n_channels, n_freqs)
    # NOTE: The mean across states/modes is subtracted from the PSDs subject-wise.

    visualize.plot_mode_spectra_group_diff(
        f,
        input_psd,
        subject_ids,
        group_assignments,
        method=model_type,
        modality=modality,
        bonferroni_ntest=n_test,
        filename=os.path.join(DATA_DIR, "analysis/psd_cluster_dynamic.png"),
    )

    # Cluster permutation test on PSDs (mean-only)
    if model_type == "hmm":
        input_psd = np.average(psd, axis=1, weights=gfo, keepdims=True)
    if model_type == "dynemo":
        input_psd = psd[:, 1, :, :, :] # use regression intercepts
        input_psd = np.expand_dims(input_psd[:, 0, :, :], axis=1)
        # all modes have same regression intercepts
    # dim: (n_subjects, n_states, n_channels, n_freqs)
    # NOTE: The mean across states/modes is extracted from the PSDs subject-wise.

    visualize.plot_mode_spectra_group_diff(
        f,
        input_psd,
        subject_ids,
        group_assignments,
        method=model_type,
        modality=modality,
        bonferroni_ntest=n_test,
        filename=os.path.join(DATA_DIR, "analysis/psd_cluster_static.png"),
    )

    # Plot PSD vs. Coherence (mean-subtracted)
    if model_type == "hmm":
        input_psd = psd - np.average(psd, axis=1, weights=gfo, keepdims=True)
    if model_type == "dynemo":
        input_psd = psd_rescaled[:, 0, :, :, :] # use rescaled regression coefficients

    coh_static_mean = np.average(coh, axis=1, weights=gfo, keepdims=True)
    input_coh = coh - coh_static_mean

    visualize.plot_pow_vs_coh(
        f,
        input_psd,
        input_coh,
        subject_ids,
        group_assignments,
        method=model_type,
        modality=modality,
        filenames=[os.path.join(DATA_DIR, f"analysis/pow_coh_dynamic_{lbl}.png") for lbl in ["an", "ap"]],
    )

    print("Analysis complete.")
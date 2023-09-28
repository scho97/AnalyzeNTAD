"""Compute wide-band power and functional connectivity maps

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
from utils.statistics import fit_glm


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

    # Set directory paths
    BASE_DIR = "/home/scho/AnalyzeNTAD/results"
    DATA_DIR = os.path.join(BASE_DIR, f"dynamic/{data_name}/{model_type}/{run_dir}")

    # Load data
    with open(os.path.join(DATA_DIR, f"model/results/{data_name}_{model_type}.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    alpha = data["alpha"]
    cov = data["covariance"]
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
        cov = cov[order] # dim: n_modes x n_channels x n_channels

    # Get HMM state time courses
    if model_type == "hmm":
        btc = modes.argmax_time_courses(alpha)
    
    # Get DyNeMo mode activation time courses
    if model_type == "dynemo":
        btc = get_dynemo_mtc(alpha, Fs, data_dir=DATA_DIR)

    # ----------- [3] ------------ #
    #      Spectral analysis       #
    # ---------------------------- #
    print("Step 3 - Computing spectral information ...")

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

    # ---------------- [4] ----------------- #
    #      Power and connectivity maps       #
    # -------------------------------------- #
    print("Step 4 - Computing power and connectivity maps (wide-band: 1-45 Hz) ...")

    # Calculate power maps
    if model_type == "hmm":
        power_map = analysis.power.variance_from_spectra(f, psd)
    elif model_type == "dynemo":
        power_map = analysis.power.variance_from_spectra(f, psd[:, 0, :, :])
        # NOTE: Here, relative (regression coefficients only) power maps are computed.
    # dim: (n_subjects, n_modes, n_channels)
    print("Shape of power maps: ", power_map.shape)

    # Caculate connectivity maps
    conn_map = analysis.connectivity.mean_coherence_from_spectra(f, coh)
    # dim: (n_subjects, n_modes, n_channels, n_channels)
    print("Shape of connectivity maps: ", conn_map.shape)

    # Get fractional occupancies to be used as weights
    fo = modes.fractional_occupancies(btc) # dim: (n_subjects, n_states)
    gfo = np.mean(fo, axis=0)

    # Fit GLM model to power maps
    power_model, power_design, power_data = fit_glm(
        power_map,
        subject_ids,
        group_assignments,
        dimension_labels=["Subjects", "States/Modes", "Channels"],
    )

    # Fit GLM model to connectivity maps
    conn_model, conn_design, conn_data = fit_glm(
        conn_map,
        subject_ids,
        group_assignments,
        dimension_labels=["Subjects", "States/Modes", "Channels", "Channels"],
    )

    # Get COPEs for overall mean (after confound regression)
    power_map = power_model.copes[2]
    conn_map = conn_model.copes[2]

    # Plot power maps
    visualize.plot_power_map(
        power_map,
        mask_file,
        parcellation_file,
        subtract_mean=(True if model_type == "hmm" else False),
        mean_weights=gfo,
        filename=os.path.join(DATA_DIR, "maps/power_map_whole.png"),
    )
    # NOTE: For DyNeMo, as we only used regression coefficients of PSDs to compute the power maps, 
    # the mean power has already been subtracted (i.e., it's contained in the intercept (constant 
    # regressor) of PSD). Hence, it is correct not to subtract the mean power across the modes here.

    # Threshold connectivity maps
    edges = analysis.connectivity.threshold(
        conn_map,
        percentile=97,
        subtract_mean=True,
        mean_weights=gfo,
        return_edges=True,
    )
    conn_map[~edges] = 0 # apply thresholding
    
    # Plot connectivity maps
    visualize.plot_connectivity_map(
        conn_map,
        parcellation_file,
        filename=os.path.join(DATA_DIR, "maps/conn_map_whole.png"),
    )

    # Plot mean-subtracted channel-averaged PSDs for each state/mode
    if model_type == "hmm":
        psd = psd - np.average(psd, axis=1, weights=gfo, keepdims=True)
    if model_type == "dynemo":
        psd = psd[:, 0, :, :, :] # use regression coefficients only
    
    psd_model, psd_design, psd_data = fit_glm(
        psd,
        subject_ids,
        group_assignments,
        dimension_labels=["Subjects", "States/Modes", "Channels", "Frequency"],
    )
    psd_mean = psd_model.copes[2]
    psd_se = np.sqrt(psd_model.varcopes[2])
    # dim: (n_modes, n_channels, n_freqs)

    visualize.plot_rsn_psd(
        f, psd_mean, psd_se, 
        filename=os.path.join(DATA_DIR, "maps/conn_psd_whole.png"),
    )

    print("Analysis complete.")
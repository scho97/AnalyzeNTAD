"""Compute static PSD using Welch's method

"""

# Set up dependencies
import os, pickle
import numpy as np
from sys import argv
from osl_dynamics import data
from osl_dynamics.analysis.static import power_spectra
from utils.data import get_subject_ids, load_group_information


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 3:
        print("Need to pass two arguments: modality & data space (e.g., python script.py eeg sensor)")
        exit()
    modality = argv[1]
    data_space = argv[2]
    Fs = 250 # sampling frequency
    print(f"[INFO] Modality: {modality.upper()}, Data Space: {data_space}")

    # Set directory paths
    BASE_DIR = "/home/scho/AnalyzeNTAD/results"
    DATA_DIR = "/ohba/pi/mwoolrich/scho/NTAD"
    SRC_DIR = os.path.join(DATA_DIR, "src")
    SAVE_DIR = os.path.join(BASE_DIR, f"static/{modality}/{data_space}_psd")
    TMP_DIR = os.path.join(SAVE_DIR, "tmp")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Load group information
    subject_ids, n_subjects = get_subject_ids(SRC_DIR, modality)
    an_idx, ap_idx = load_group_information(subject_ids)
    print(f"Number of available subjects: {n_subjects} | AN={len(an_idx)} | AP={len(ap_idx)}")
    if n_subjects != (len(an_idx) + len(ap_idx)):
        raise ValueError("one or more groups lacking subjects.")

    # Load data
    print("Loading data ...")
    file_names = []    
    for id in subject_ids:
        if data_space == "source":
            pick_name = "misc"
            file_path = os.path.join(DATA_DIR, f"src/{modality}/{id}/sflip_parc-raw.fif")
        elif data_space == "sensor":
            pick_name = modality
            file_path = os.path.join(DATA_DIR, f"preproc/{modality}/{id}_resting_close_bl_raw_tsss" 
                                     + f"/{id}_resting_close_bl_tsss_preproc_raw.fif")
        file_names.append(file_path)

    # Build training data
    print(f"Picking {pick_name.upper()} channels ...")
    training_data = data.Data(file_names, picks=pick_name, reject_by_annotation="omit", store_dir=TMP_DIR)

    # Separate data into groups
    input_data = [x for x in training_data.arrays]
    if input_data[0].shape[0] < input_data[0].shape[1]:
        print("Reverting dimension to (samples x channels) ...")
        input_data = [x.T for x in input_data]
    input_an = [input_data[idx] for idx in an_idx]
    input_ap = [input_data[idx] for idx in ap_idx]
    print("Total # of channels/parcels: ", input_data[0].shape[1])
    print("Shape of the single subject input data: ", np.shape(input_an[0]))

    # Clean up
    training_data.delete_dir()

    # Calculate subject-specific static power spectra
    print("Computing PSDs ...")

    f_an, psd_an, w_an = power_spectra(
        data=input_an,
        window_length=int(Fs * 2),
        sampling_frequency=Fs,
        frequency_range=[1, 45],
        step_size=int(Fs),
        return_weights=True,
        standardize=True,
    ) # for amyloid negative participants

    f_ap, psd_ap, w_ap = power_spectra(
        data=input_ap,
        window_length=int(Fs * 2),
        sampling_frequency=Fs,
        frequency_range=[1, 45],
        step_size=int(Fs),
        return_weights=True,
        standardize=True,
    ) # for amyloid positive participants

    if (f_an != f_ap).any():
        raise ValueError("Frequency vectors of each age group do not match.")
    freqs = f_an

    # Get PSDs and weights of the entire dataset
    psd = np.concatenate((psd_an, psd_ap), axis=0)
    n_samples = [d.shape[0] for d in input_an + input_ap]
    w = np.array(n_samples) / np.sum(n_samples)

    # Save results
    print("Saving results ... ")
    output = {"freqs": freqs,              
              "psd_an": psd_an,
              "psd_ap": psd_ap,
              "psd": psd,
              "weights_an": w_an,
              "weights_ap": w_ap,
              "weights": w}
    with open(SAVE_DIR + "/psd.pkl", "wb") as output_path:
        pickle.dump(output, output_path)
    output_path.close()

    print("Computation completed.")
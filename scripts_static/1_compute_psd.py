"""Compute static PSD using Welch's method

"""

# Set up dependencies
import os
import pickle
import mne
import numpy as np
from sys import argv
from osl_dynamics import data
from osl_dynamics.analysis.static import power_spectra
from utils.data import get_subject_ids


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

    # Load subject information
    subject_ids, n_subjects = get_subject_ids(SRC_DIR, modality)
    print(f"Number of available subjects: {n_subjects}")

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

    # Get subject-wise signal recordings
    print(f"Picking {pick_name.upper()} channels ...")
    if (modality == "eeg") and (data_space == "sensor"):
        input_data = []
        for file_path in file_names:
            # Get subject-wise data arrays
            raw = mne.io.read_raw_fif(file_path, verbose=False)
            data_array = raw.get_data(picks=pick_name, reject_by_annotation="omit", verbose=False).T
            data_array = data_array.astype(np.float32)
            raw.close()

            # Load common EEG sensor indices
            with open(BASE_DIR + "/data/common_eeg_sensor.pkl", "rb") as input_path:
                common_eeg_idx = pickle.load(input_path)
            input_path.close()

            # Retain common EEG sensors
            n_sensors = data_array.shape[1]
            if n_sensors == 66: n_sensors = 70
            idx = common_eeg_idx[f"EasyCap{n_sensors}"]
            input_data.append(data_array[:, idx])
    else:
        # Build training data
        training_data = data.Data(file_names, picks=pick_name, reject_by_annotation="omit", store_dir=TMP_DIR)

        # Get subject-wise data arrays
        input_data = [x for x in training_data.arrays]
        if input_data[0].shape[0] < input_data[0].shape[1]:
            print("Reverting dimension to (samples x channels) ...")
            input_data = [x.T for x in input_data]

        # Clean up
        training_data.delete_dir()
    
    print("Total # of channels/parcels: ", input_data[0].shape[1])
    print("Shape of the single subject input data: ", np.shape(input_data[0]))
    
    # Get sample sizes of data arrays
    n_samples_input = [d.shape[0] for d in input_data]
    # NOTE: This can be used later to calculate weights for each group.

    # Calculate subject-specific static power spectra
    print("Computing PSDs ...")

    freqs, psd, w = power_spectra(
        data=input_data,
        window_length=int(Fs * 2),
        sampling_frequency=Fs,
        frequency_range=[1, 45],
        step_size=int(Fs),
        return_weights=True,
        standardize=True,
    ) # for entire participants

    # Save results
    print("Saving results ... ")
    output = {"freqs": freqs,
              "psd": psd,
              "weights": w,
              "n_samples": n_samples_input}
    with open(SAVE_DIR + "/psd.pkl", "wb") as output_path:
        pickle.dump(output, output_path)
    output_path.close()

    print("Computation completed.")
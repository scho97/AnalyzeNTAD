"""Manually identify bad channels in the NTAD EEG data

NOTE: In NTAD EEG, we noticed that there can be a few bad channels that are difficult to 
detect for some subjects. For these subjects, bad channels were first identified and then 
manually dropped using the custom `detect_bad_channels_manual()` function during the 
preprocessing step.

"""

# Install dependencies
import os
import mne
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Define subject IDs of interest
    subject_ids = ["C1019", "P1010", "P1030", "P1054", "P1060"]

    subject_ids = ["C1019"]

    # Define modality
    modality = "eeg"
    eeg_flag, meg_flag = False, False
    if modality == "eeg": eeg_flag = True
    if modality == "meg": meg_flag = True

    # Set file path
    preproc_file_path = f"/ohba/pi/mwoolrich/scho/NTAD/preproc/{modality}"
    
    for id in subject_ids:
        # Load raw signal
        preproc_file_name = os.path.join(preproc_file_path, 
                                         f"{id}_resting_close_bl_raw_tsss/{id}_resting_close_bl_tsss_preproc_raw.fif")
        raw = mne.io.read_raw_fif(preproc_file_name)
        raw.pick_types(eeg=eeg_flag, meg=meg_flag)
        ch_names = raw.info["ch_names"]
        x = raw.get_data(picks=[modality], reject_by_annotation="omit")
        print(f"{id} - # of channels: {len(ch_names)}")
        print(f"{id} - Data shape: {x.shape}")
        
        # Plot and detect bad channel outliers
        fig, ax = plt.subplots(nrows=1, ncols=1)
        Pxx = []
        for ch_idx in np.arange(x.shape[0]):
            pxx, freqs = ax.psd(x[ch_idx, :], Fs=raw.info["sfreq"])
            Pxx.append(pxx)
        Pxx = np.array(Pxx) # dim: (n_channels, n_freqs)
        Pxx_dB = 10*np.log10(Pxx) # convert to log scale
        Pxx_dB_sum = np.sum(Pxx_dB, axis=1) # sum over frequencies
        zscores = (Pxx_dB_sum - np.mean(Pxx_dB_sum)) / np.std(Pxx_dB_sum)
        if np.any(np.abs(zscores) > 3):
            outlier_idx = np.where(np.abs(zscores) > 3)[0]
            outlier_names = []
            for i, idx in enumerate(outlier_idx):
                outlier_names.append(ch_names[idx])
                ax.text(freqs[len(freqs)//2] + (i * 20), np.mean(Pxx_dB[idx, :]), ch_names[idx], color="red", fontsize=8)
            print(f"{id} - Outlier channels: {outlier_names}")

        plt.savefig(f"bad_channels_{id}.png")

    print("Analysis complete.")
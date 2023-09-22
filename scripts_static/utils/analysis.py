"""Functions for static post-hoc analysis

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osl_dynamics.analysis import power, static
from osl_dynamics.data import Data
from utils.data import get_subject_ids, load_group_information

##################
##     PSDs     ##
##################

def get_peak_frequency(freqs, psd, freq_range):
    """Extract frequency at which peak happens in a PSD.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array of PSD. Shape must be (n_freqs,).
    psd : np.ndarray
        Power spectral densities. Shape must be (n_freqs,) or (n_subjects, n_freqs).
    freq_range : list
        List containing the lower and upper bounds of frequencies of interest.

    Returns
    -------
    peak_freq : np.ndarray
        Frequencies at which peaks occur.
    """
    # Validation
    if psd.ndim > 2:
        raise ValueError("psd need to be an array with 1 or 2 dimensions.")

    # Frequencies to search for the peak
    peak_freq_range = np.where(np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1]))
    
    # Detect a frequency in which a peak happens
    if psd.ndim == 1:
        bounded_psd = psd[peak_freq_range]
        peak_freq = freqs[psd == max(bounded_psd)]
    elif psd.ndim == 2:
        bounded_psd = np.squeeze(psd[:, peak_freq_range])
        peak_freq = np.empty((bounded_psd.shape[0]))
        for n in range(len(psd)):
            peak_freq[n] = freqs[psd[n] == max(bounded_psd[n])]

    return peak_freq

##################
##  Power Maps  ##
##################

class SubjectStaticPowerMap():
    """
    Class for computing the subject-level power maps.
    """
    def __init__(self, freqs, data):
        self.n_subjects = data.shape[0]
        self.freqs = freqs
        self.psds = data # dim: (n_subjects x n_channels x n_freqs)

    def plot_psd(self, filename):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for n in range(self.n_subjects):
            psd = np.mean(self.psds[n], axis=0) # mean across channels
            plt.plot(self.freqs, psd)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (a.u.)')
        ax.set_title('Subject-level PSDs')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig.savefig(filename)
        plt.close()
        return None

    def compute_power_map(self, freq_range, scale=False):
        power_maps = power.variance_from_spectra(
            self.freqs,
            self.psds,
            frequency_range=freq_range,
        )
        if scale:
            power_maps_full = power.variance_from_spectra(self.freqs, self.psds)
            power_maps = np.divide(power_maps, power_maps_full)
        print("Shape of power maps: ", power_maps.shape)
        
        return power_maps
    
####################
##  Connectivity  ##
####################

def compute_aec(dataset_dir,
                data_space,
                modality,
                sampling_frequency,
                freq_range,
                tmp_dir,
    ):
    """Compute subject-level AEC matrices.

    Parameters
    ----------
    dataset_dir : str
        Path to the directory containing subject data.
    data_space : str
        Data measurement space. Should be either "sensor" or "source".
    modality : str
        Type of data modality. Currently supports only "eeg" and "meg".
    sampling_frequency : int
        Sampling frequency of the measured data.
    freq_range : list of int
        Upper and lower frequency bounds for filtering signals to calculate
        amplitude envelope.
    tmp_dir : str
        Path to a temporary directory for building a traning dataset.
        For further information, see data.Data() in osl-dynamics package.

    Returns
    -------
    conn_map : np.ndarray
        AEC functional connectivity matrix. Shape is (n_subjects, n_channels, n_channels).
    """

    # Validation
    data_type = os.path.basename(dataset_dir)
    if data_type not in ["preproc", "src"]:
        raise ValueError("data_type should be either 'preproc' or 'src'.")
    if data_type == "preproc":
        assert data_space == "sensor", "data_space should be 'sensor'."
    if data_type == "src":
        assert data_space == "source", "data_space should be 'source'."
    
    # Load group information
    subject_ids, n_subjects = get_subject_ids(dataset_dir, modality)
    an_idx, ap_idx = load_group_information(subject_ids)
    print("Number of subjects: {} (AN={}, AP={})".format(n_subjects, len(an_idx), len(ap_idx)))

    # Load data
    file_names = []    
    for id in subject_ids:
        if data_space == "source":
            pick_name = "misc"
            file_path = os.path.join(dataset_dir, f"{modality}/{id}/sflip_parc-raw.fif")
        elif data_space == "sensor":
            pick_name = modality
            file_path = os.path.join(dataset_dir, f"{modality}/{id}_resting_close_bl_raw_tsss" 
                                     + f"/{id}_resting_close_bl_tsss_preproc_raw.fif")
        file_names.append(file_path)

    # Build training data
    training_data = Data(file_names, picks=pick_name, reject_by_annotation="omit", store_dir=tmp_dir)

    # Separate data into groups
    input_data = [x for x in training_data.arrays]
    if input_data[0].shape[0] < input_data[0].shape[1]:
        print("Reverting dimension to (samples x channels) ...")
        input_data = [x.T for x in input_data]
    print("Total # of channels/parcels: ", input_data[0].shape[1])
    print("Shape of the single subject input data: ", np.shape(input_data[0]))
    data = Data(input_data, store_dir=tmp_dir, sampling_frequency=sampling_frequency)

    # Prepare data to compute amplitude envelope
    data.prepare(
        methods = {
            "filter": {"low_freq": freq_range[0], "high_freq": freq_range[1]},
            "amplitude_envelope": {},
            "standardize": {},
        }
    )
    ts = data.time_series()

    # Calculate functional connectivity using AEC
    conn_map = static.functional_connectivity(ts, conn_type="corr")

    # Clean up
    training_data.delete_dir()
    data.delete_dir()

    return conn_map
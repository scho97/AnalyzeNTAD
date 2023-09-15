"""Functions for static post-hoc analysis

"""

import numpy as np
import matplotlib.pyplot as plt
from osl_dynamics.analysis import power

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

    def separate_by_group(self, power_maps, divide_idx):
        power1 = power_maps[:divide_idx, :]
        power2 = power_maps[divide_idx:, :]
        
        return power1, power2
"""Functions for static post-hoc analysis

"""

import numpy as np

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
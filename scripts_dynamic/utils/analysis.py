"""Functions for analyzing the data

"""

import os
import pickle
from osl_dynamics import analysis

def get_psd_coh(dataset, alpha, Fs, calc_type, save_dir, n_jobs=1):
    """Computes or loads state/mode-specific PSD and coherence data.

    Parameters
    ----------
    dataset : list of np.ndarray or np.ndarray
        Time series data to calculate a time-varying PSD for. Shape must be (n_subjects,
        n_samples, n_channels) or (n_samples, n_channels).
    alpha : list of np.ndarray or np.ndarray
        Inferred mode mixing factors. Shape must be (n_subjects, n_samples,
        n_modes) or (n_samples, n_modes).
    Fs : int
        Sampling frequency of the input time series.
    calc_type : str
        Method for PSD and coherence computation. Should be either "glm"
        (regression) or "mtp" (multitaper).
    save_dir : str
        Path to directory where the PSDs and coherences will be saved.
    n_jobs : int
        Number of parallel jobs. Default to 1.

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freqs,).
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape is (n_subjects,
        n_states, n_channels, n_freqs) or (n_subjects, 2, n_states, n_channels,
        n_freqs).
    coh : np.ndarray
        Coherences for each state/mode. Shape is (n_subjects, n_states, n_channels,
        n_channels, n_freqs).
    w : np.ndarray
        Weights for each subject-specific PSD. Shape is (n_subjects,).
    """

    # Set path for saving the outputs
    filename = os.path.join(save_dir, f"model/results/{calc_type}_psd_coh.pkl")

    # Load data if computed results already exist
    if os.path.exists(filename):
        print("Data already exist. Loading data ...")
        res = _load_psd_coh(filename)
        return res
    
    # Calculate subject-specific state/mode PSDs and coherences
    if calc_type == "glm":
        f, psd, coh, w = analysis.spectral.regression_spectra(
            data=dataset,
            alpha=alpha,
            window_length=int(4 * Fs),
            sampling_frequency=Fs,
            frequency_range=[1, 45],
            step_size=20,
            n_sub_windows=8,
            return_weights=True,
            return_coef_int=True,
            standardize=True,
            n_jobs=n_jobs,
        )
        # dim (psd): n_subjects x 2 x n_modes x n_channels x n_freqs
        # dim (coh): n_subjects, n_modes, n_channels, n_channels, n_freqs
    elif calc_type == "mtp":
        f, psd, coh, w = analysis.spectral.multitaper_spectra(
            data=dataset,
            alpha=alpha,
            sampling_frequency=Fs,
            time_half_bandwidth=4,
            n_tapers=7,
            frequency_range=[1, 45],
            return_weights=True,
            standardize=True,
            n_jobs=n_jobs,
        )
        # dim (psd): n_subjects x n_modes x n_channels x n_freqs
        # dim (coh): n_subjects x n_modes x n_channels x n_channels x n_freqs

    # Save outputs
    outputs = {
        "f": f,
        "psd": psd,
        "coh": coh,
        "w": w,
    }
    with open(filename, "wb") as output_path:
        pickle.dump(outputs, output_path)
    output_path.close()

    return (f, psd, coh, w)

def _load_psd_coh(filename):
    """Loads PSD and coherence data that were generated with `get_psd_coh()`.

    Parameters
    ----------
    filename : str
        File location where the data is stored.

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freqs,).
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape is (n_subjects,
        n_states, n_channels, n_freqs) or (n_subjects, 2, n_states, n_channels,
        n_freqs).
    coh : np.ndarray
        Coherences for each state/mode. Shape is (n_subjects, n_states, n_channels,
        n_channels, n_freqs).
    w : np.ndarray
        Weights for each subject-specific PSD. Shape is (n_subjects,).
    """

    # Load saved data
    with open(filename, "rb") as input_path:
        data = pickle.load(input_path)
    input_path.close()

    # Extract data from dictionary
    f = data["f"]
    psd = data["psd"]
    coh = data["coh"]
    w = data["w"]

    return (f, psd, coh, w)
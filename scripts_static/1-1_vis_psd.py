"""Visualize computed static PSDs

"""

# Set up dependencies
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sys import argv
from utils.analysis import get_peak_frequency
from utils.array_ops import get_mean_error
from utils.data import get_subject_ids, load_group_information
from utils.statistics import fit_glm, max_stat_perm_test, cluster_perm_test
from utils.visualize import GroupPSDDifference, categorize_pvalue


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #

    # Set hyperparameters
    if len(argv) != 3:
        print("Need to pass two arguments: modality & data space (e.g., python script.py eeg sensor)")
    modality = argv[1]
    data_space = argv[2]
    print(f"[INFO] Data Space: {data_space.upper()} | Modality: {modality.upper()}")

    # Set directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results/static"
    DATA_DIR = os.path.join(BASE_DIR, f"{modality}/{data_space}_psd")
    SAVE_DIR = DATA_DIR
    SRC_DIR = "/ohba/pi/mwoolrich/scho/NTAD/src"

    # Load data
    with open(DATA_DIR + "/psd.pkl", "rb") as input_path:
        data = pickle.load(input_path)
    freqs = data["freqs"]
    psd = data["psd"]
    weights = data["weights"]
    n_samples = data["n_samples"]

    # Average PSDs over channels/parcels
    cpsd = np.mean(psd, axis=1)
    # dim: (n_subjects, n_channels, n_freqs) -> (n_subjects, n_freqs)

    # Load meta data
    df_meta = pd.read_excel(
        "/home/scho/AnalyzeNTAD/scripts_data/all_data_info.xlsx"
    )

    # Load group information
    subject_ids, n_subjects = get_subject_ids(SRC_DIR, modality)
    an_idx, ap_idx = load_group_information(subject_ids)
    n_an, n_ap = len(an_idx), len(ap_idx)
    print(f"Number of available subjects: {n_subjects} | AN={n_an} | AP={n_ap}")

    # --------------- [2] -------------- #
    #      Group-level static PSDs       #
    # ---------------------------------- #

    # Fit GLM model to PSDs
    psd_model, psd_design, psd_data = fit_glm(
        psd,
        modality=modality,
        dimension_labels=["Subjects", "Channels", "Frequency"],
    )

    # Get group-level PSDs
    gpsd = psd_model.copes[1]
    gpsd_an = psd_model.betas[1]
    gpsd_ap = psd_model.betas[0]
    # dim: (n_channels, n_freqs)

    # Compute the mean and standard errors over channels
    avg_psd, err_psd = get_mean_error(gpsd)
    avg_psd_an, err_psd_an = get_mean_error(gpsd_an)
    avg_psd_ap, err_psd_ap = get_mean_error(gpsd_ap)

    # Report alpha peaks
    an_peak = get_peak_frequency(freqs, avg_psd_an, freq_range=[5, 15])
    ap_peak = get_peak_frequency(freqs, avg_psd_ap, freq_range=[5, 15])
    print("Alpha peak in amyloid negative population (Hz): ", an_peak)
    print("Alpha peak in amyloid positive population (Hz): ", ap_peak)

    # Set visualization parameters
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    cmap = sns.color_palette("deep")

    # Plot group-level (i.e., subject-averaged) PSDs
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    plt.plot(freqs, avg_psd, color=cmap[7], lw=2, label='All (n={})'.format(n_an + n_ap))
    plt.fill_between(freqs, avg_psd - err_psd, avg_psd + err_psd, color=cmap[7], alpha=0.4)
    plt.plot(freqs, avg_psd_an, color=cmap[0], lw=2, label='Amyloid Negative (n={})'.format(n_an))
    plt.fill_between(freqs, avg_psd_an - err_psd_an, avg_psd_an + err_psd_an, color=cmap[0], alpha=0.4)
    plt.plot(freqs, avg_psd_ap, color=cmap[3], lw=2, label='Amyloid Positive (n={})'.format(n_ap))
    plt.fill_between(freqs, avg_psd_ap - err_psd_ap, avg_psd_ap + err_psd_ap, color=cmap[3], alpha=0.4)
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('PSD (a.u.)', fontsize=14)
    ax.set_ylim(0, 0.1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(2)
    ax.tick_params(width=2)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'static_psd.png'))
    plt.close(fig)

    # --------------- [3] -------------- #
    #      Group-level static PSDs       #
    # ---------------------------------- #

    # Fit GLM model to PSDs
    cpsd_model, cpsd_design, cpsd_data = fit_glm(
        cpsd,
        modality=modality,
        dimension_labels=["Subjects", "Frequency"],
    )

    # Perform a cluster permutation test on parcel-averaged PSDs
    print("*** Running Cluster Permutation Test ***")
    _, clu = cluster_perm_test(
        cpsd_model,
        cpsd_data,
        cpsd_design,
        pooled_dims=(1,),
        contrast_idx=0,
        n_perm=5000,
        metric="tstats",
        bonferroni_ntest=2, # n_test = n_data_space
    )

    # Plot group difference PSDs
    PSD_DIFF = GroupPSDDifference(freqs, gpsd_an, gpsd_ap, data_space, modality)
    PSD_DIFF.prepare_data()
    PSD_DIFF.plot_psd_diff(
        clusters=clu,
        group_lbls=["AN", "AP"],
        save_dir=SAVE_DIR
    )
    
    # -------------- [4] -------------- #
    #      Alpha peak frequencies       #
    # --------------------------------- #

    # Compute subject-wise peak shifts of PSDs
    peaks = get_peak_frequency(freqs, cpsd, freq_range=[7, 14])
    peaks = peaks[:, np.newaxis]

    # Fit GLM model to alpha peak frequencies
    peak_model, peak_design, peak_data = fit_glm(
        peaks,
        modality=modality,
        dimension_labels=["Subjects", "Frequency"],
    )

    # Test between-group difference in peak shifts
    pval = max_stat_perm_test(
        peak_model,
        peak_data,
        peak_design,
        pooled_dims=1,
        contrast_idx=0,
        metric="tstats",
    )
    bonferroni_ntest = 2 # n_test = n_data_space
    pval *= bonferroni_ntest
    pval_lbl = categorize_pvalue(pval[0])

    # Summarize results
    print("Alpha peak shifts: {:.3e} +/ {:.3e}".format(
        peak_model.copes[0][0],
        peak_model.varcopes[0][0],
    ))
    print(f"Max-t permutation test p-value: {pval[0]:.3e} ({pval_lbl})")

    print("Visualization complete.")
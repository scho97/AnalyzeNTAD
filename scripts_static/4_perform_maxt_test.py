"""Perform maximum statistics non-parameteric permutation testing 
   on power and connectivity maps
"""

# Set up dependencies
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glmtools as glm
from osl_dynamics import analysis
from utils.array_ops import round_nonzero_decimal, round_up_half
from utils.statistics import fit_glm, max_stat_perm_test
from utils.visualize import create_transparent_cmap


if __name__ == "__main__":
    # Set up hyperparameters
    modality = "eeg"
    data_space = "source"
    data_type = "power"
    band_name = "wide"
    bonferroni_ntest = 1 # n_test = n_freq_bands
    print(f"[INFO] Modality: {modality.upper()}, Data Space: {data_space}, Data Type: {data_type}, Frequency Band: {band_name}")

    # Set parcellation file paths
    mask_file = "MNI152_T1_8mm_brain.nii.gz"
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Set directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results/static"
    DATA_DIR = os.path.join(BASE_DIR, f"{modality}/{data_type}_{data_space}_{band_name}")
    SAVE_DIR = DATA_DIR

    # Load data
    print("Loading data ...")
    with open(os.path.join(SAVE_DIR, f"{data_type}.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    input_path.close()

    if data_type == "power":
        input_data = data["power_maps"]
        # dim: (n_subjects, n_channels)
        n_channels = input_data.shape[-1]
        dimension_labels = ["Subjects", "Channels"]
    elif data_type == "aec":
        input_data = data["conn_maps"]
        n_channels = input_data.shape[-1]
        i, j = np.triu_indices(n_channels, 1) # excluding diagonals
        m, n = np.tril_indices(n_channels, -1) # excluding diagonals
        input_data = np.array([d[i, j] for d in input_data])
        # NOTE: We flatten the upper triangle of connectivity matrices.
        # dim: (n_subjects, n_connections)
        dimension_labels = ["Subjects", "Connections"]
    pooled_dims = 1

    # Fit GLM on specified data
    glm_model, glm_design, glm_data = fit_glm(
        input_data,
        modality=modality,
        dimension_labels=dimension_labels,
        plot_verbose=True,
        save_path=os.path.join(SAVE_DIR, "design.png"),
    )

    # Perform a max-t permutation test
    pval, perm = max_stat_perm_test(
        glm_model,
        glm_data,
        glm_design,
        pooled_dims,
        contrast_idx=0,
        metric="tstats",
        return_perm=True,
    )
    pval *= bonferroni_ntest # apply Bonferroni correction
    null_dist = perm.nulls # dim: (n_perm,)

    # Get critical threshold value
    p_alpha = 100 - (0.05 / bonferroni_ntest) * 100
    thresh = perm.get_thresh(p_alpha)
    # NOTE: We use 0.05 as our alpha threshold.
    print(f"Metric threshold: {thresh:.3f}")

    # Plot null distribution and threshold
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    ax.hist(null_dist, bins=50, histtype="step", density=True)
    ax.axvline(thresh, color='black', linestyle='--')
    ax.set_xlabel('Max t-statistics')
    ax.set_ylabel('Density')
    ax.set_title('Threshold: {:.3f}'.format(thresh))
    plt.savefig(os.path.join(SAVE_DIR, "null_dist.png"))
    plt.close(fig)

    # Plot the results
    if data_type == "aec":
        # Get t-map
        tmap = np.zeros((n_channels, n_channels))
        tmap[i, j] = glm_model.tstats[0]
        tmap[m, n] = tmap.T[m, n] # make matrix symmetrical
        # alternatively: tmap = tmap + tmap.T

        # Plot t-heatmap
        np.fill_diagonal(tmap, val=0)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
        vmin, vmax = np.min(tmap), np.max(tmap)
        ticks = np.arange(0, len(tmap), 12)
        if vmax <= 0:
            vmax = np.abs(vmin) # make symmetrical color scale
        tnorm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        img = ax.imshow(tmap, cmap='RdBu_r', norm=tnorm)
        ax.set(
            xticks=ticks,
            yticks=ticks,
            xticklabels=ticks + 1,
            yticklabels=ticks + 1,
        )
        ax.tick_params(labelsize=18)
        ax.set_xlabel('Regions', fontsize=18)
        ax.set_ylabel('Regions', fontsize=18)
        cbar = fig.colorbar(img, ax=ax, shrink=0.92)
        cbar.set_label("t-statistics", fontsize=18)
        cbar.ax.tick_params(labelsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "map_tscore.png"))
        plt.close(fig)

        # Plot t-map graph network
        t_network = analysis.connectivity.threshold(tmap, absolute_value=True, percentile=97)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        analysis.connectivity.save(
            connectivity_map=t_network,
            parcellation_file=parcellation_file,
            plot_kwargs={"edge_cmap": "RdBu_r", "figure": fig, "axes": ax},
        )
        cb_ax = fig.get_axes()[-1]
        cb_ax.tick_params(labelsize=20)
        fig.savefig(os.path.join(SAVE_DIR, "network_tscore.png"), transparent=True)
        plt.close(fig)

        # Plot thresholded graph network
        tmap_thr = tmap.copy()
        pval_map = np.ones((n_channels, n_channels))
        pval_map[i, j] = pval
        pval_map[m, n] = pval_map.T[m, n] # make matrix symmetrical
        thr_idx = pval_map < 0.05
        if np.sum(thr_idx) > 0:
            tmap_thr = np.multiply(tmap_thr, thr_idx)
            cmap = "RdBu_r"
            savename = os.path.join(SAVE_DIR, "network_tscore_thr.png")
        else:
            tmap_thr = np.ones((tmap_thr.shape))
            cmap = create_transparent_cmap("binary_r")
            savename = os.path.join(SAVE_DIR, "network_tscore_thr_ns.png")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        analysis.connectivity.save(
            connectivity_map=tmap_thr,
            parcellation_file=parcellation_file,
            plot_kwargs={"edge_cmap": cmap, "figure": fig, "axes": ax},
        )
        cb_ax = fig.get_axes()[-1]
        cb_ax.tick_params(labelsize=20)
        fig.savefig(savename, transparent=True)
        plt.close()
    
    if data_type == "power":
        # Get t-map
        tmap = glm_model.tstats[0]

        # Plot power map of t-statistics
        figures, axes = analysis.power.save(
            power_map=tmap,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            plot_kwargs={"cmap": "RdBu_r"},
        )
        fig = figures[0]
        fig.set_size_inches(5, 6)
        cb_ax = axes[0][-1]
        pos = cb_ax.get_position()
        new_pos = [pos.x0 * 0.90, pos.y0 + 0.02, pos.width * 1.20, pos.height * 1.10]
        cb_ax.set_position(new_pos)
        if np.any(np.abs(np.array(cb_ax.get_xlim())) < 1):
            hmin = round_nonzero_decimal(cb_ax.get_xlim()[0], method="ceil") # ceiling for negative values
            hmax = round_nonzero_decimal(cb_ax.get_xlim()[1], method="floor") # floor for positive values
            cb_ax.set_xticks(np.array([hmin, 0, hmax]))
        else:
            cb_ax.set_xticks(
                [round_up_half(val) for val in cb_ax.get_xticks()[1:-1]]
            )
        cb_ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 6))
        cb_ax.tick_params(labelsize=24)
        fig.savefig(os.path.join(SAVE_DIR, "map_tscore.png"), bbox_inches="tight")
        plt.close(fig)

        # Plot power map of thresholded t-statistics
        tmap_thr = tmap.copy()
        thr_idx = pval < 0.05
        if np.sum(thr_idx) > 0:
            tmap_thr = np.multiply(tmap_thr, thr_idx)
            cmap = "RdBu_r"
            savename = os.path.join(SAVE_DIR, "map_tscore_thr.png")
        else:
            tmap_thr = np.ones((tmap_thr.shape))
            cmap = create_transparent_cmap("RdBu_r")
            savename = os.path.join(SAVE_DIR, "map_tscore_thr_ns.png")
        figures, axes = analysis.power.save(
            power_map=tmap_thr,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            plot_kwargs={"cmap": cmap},
        )
        fig = figures[0]
        fig.set_size_inches(5, 6)
        cb_ax = axes[0][-1]
        pos = cb_ax.get_position()
        new_pos = [pos.x0 * 0.90, pos.y0 + 0.02, pos.width * 1.20, pos.height * 1.10]
        cb_ax.set_position(new_pos)
        if np.any(np.abs(np.array(cb_ax.get_xlim())) < 1):
            hmin = round_nonzero_decimal(cb_ax.get_xlim()[0], method="ceil") # ceiling for negative values
            hmax = round_nonzero_decimal(cb_ax.get_xlim()[1], method="floor") # floor for positive values
            cb_ax.set_xticks(np.array([hmin, 0, hmax]))
        else:
            cb_ax.set_xticks(
                [round_up_half(val) for val in cb_ax.get_xticks()[1:-1]]
            )
        cb_ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 6))
        cb_ax.tick_params(labelsize=24)
        fig.savefig(savename, bbox_inches="tight")
        plt.close(fig)

    print("Max t-statistics of the original t-map: ", np.max(np.abs(tmap))) # absolute values used for two-tailed t-test
    
    print("Analysis Complete.")
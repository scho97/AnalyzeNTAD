"""Visualise group-level AEC heatmaps and whole-brain graph networks

"""

# Set up dependencies
import os
import pickle
import numpy as np
from utils import visualize
from utils.statistics import fit_glm
from osl_dynamics.analysis import connectivity


if __name__ == "__main__":
    # Set up hyperparameters
    modality = "eeg"
    data_space = "source"
    band_name = "wide"
    print(f"[INFO] Data Space: {data_space.upper()} | Modality: {modality.upper()} | Frequency Band: {band_name.upper()}")

    # Set parcellation file paths
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Set directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results/static"
    SAVE_DIR = os.path.join(BASE_DIR, f"{modality}/aec_{data_space}_{band_name}")

    # Load data
    with open(os.path.join(SAVE_DIR, "aec.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    conn_maps = data["conn_maps"]
    n_channels = conn_maps.shape[-1]
    print("Shape of AEC maps: ", conn_maps.shape)
    # dim: (n_subjects x n_channels x n_channels)

    # Fit GLM on connectivity maps
    aec_model, aec_design, aec_data = fit_glm(
        conn_maps,
        modality=modality,
        dimension_labels=["Subjects", "Channels", "Channels"],
    )

    # Get group-level AEC maps
    gconn_map_an = aec_model.betas[1]
    gconn_map_ap = aec_model.betas[0]
    # dim: (n_channels, n_channels)

    print(gconn_map_an.shape)
    print(gconn_map_ap.shape)
    
    # Fill diagonal elements with NaNs for visualization
    np.fill_diagonal(gconn_map_an, np.nan)
    np.fill_diagonal(gconn_map_ap, np.nan)
    # Note: NaN is preferred over zeros, because a zero value will be included in the distribution, while NaNs won't.

    # Plot AEC maps
    print("Plotting AEC maps ...")

    heatmaps = [gconn_map_an, gconn_map_ap]
    vmin = np.nanmin(np.concatenate(heatmaps))
    vmax = np.nanmax(np.concatenate(heatmaps))
    labels = ["an", "ap"]
    for i, heatmap in enumerate(heatmaps):
        visualize.plot_aec_heatmap(
            heatmap=heatmap,
            filename=os.path.join(SAVE_DIR, f"aec_heatmap_{labels[i]}.png"),
            vmin=vmin, vmax=vmax,
        )

    gconn_map_diff = gconn_map_ap - gconn_map_an # amyloid positive vs. amyloid negative
    visualize.plot_aec_heatmap(
        heatmap=gconn_map_diff,
        filename=os.path.join(SAVE_DIR, "aec_heatmap_diff.png"),
    )

    # Threshold connectivity matrices
    gconn_map_an = connectivity.threshold(gconn_map_an, percentile=97)
    gconn_map_ap = connectivity.threshold(gconn_map_ap, percentile=97)
    gconn_map_diff = connectivity.threshold(gconn_map_diff, absolute_value=True, percentile=97)

    # Plot AEC graph networks
    print("Plotting AEC networks ...")

    visualize.plot_group_connectivity_map(
        conn_map=gconn_map_an,
        parcellation_file=parcellation_file,
        filename=os.path.join(SAVE_DIR, "aec_network_an.png"),
    )
    visualize.plot_group_connectivity_map(
        conn_map=gconn_map_ap,
        parcellation_file=parcellation_file,
        filename=os.path.join(SAVE_DIR, "aec_network_ap.png"),
    )
    visualize.plot_group_connectivity_map(
        conn_map=gconn_map_diff,
        parcellation_file=parcellation_file,
        filename=os.path.join(SAVE_DIR, "aec_network_diff.png"),
    )

    print("Visualization complete.")
"""Visualise group-level power maps

"""

# Set up dependencies
import os
import pickle
import numpy as np
from utils.statistics import fit_glm
from utils.visualize import plot_group_power_map


if __name__ == "__main__":
    # Set hyperparameters
    modality = "eeg"
    data_space = "source"
    freq_range = [1, 45]
    band_name = "wide"
    print(f"[INFO] Modality: {modality.upper()} | Data Space: {data_space} | " + 
          "Frequency Band: {band_name} ({freq_range[0]}-{freq_range[1]} Hz)")

    # Set parcellation file paths
    mask_file = "MNI152_T1_8mm_brain.nii.gz"
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Set directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results/static"
    DATA_DIR = os.path.join(BASE_DIR, f"{modality}/power_{data_space}_{band_name}")
    SAVE_DIR = DATA_DIR

    # Load data
    with open(os.path.join(DATA_DIR, "power.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    power_maps = data["power_maps"]
    print("Shape of power maps: ", power_maps.shape)

    # Fit GLM on power maps
    power_model, power_design, power_data = fit_glm(
        power_maps,
        modality=modality,
        dimension_labels=["Subjects", "Channels"],
    )

    # Get group-level power maps
    gpower_an = power_model.betas[1]
    gpower_ap = power_model.betas[0]
    gpower_diff = gpower_ap - gpower_an # amyloid positive - amyloid negative
    # dim: (n_channels,)
    print("Shape of group-level power maps :", gpower_diff.shape)

    # Set visualization parameters
    hmax = np.max([gpower_an, gpower_ap])
    hmax *= 1.5 # add some margin to prevent oversaturation

    # Plot group-level power maps
    plot_group_power_map(
        gpower_an,
        filename=os.path.join(SAVE_DIR, "power_map_an.png"),
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        data_space=data_space,
        modality=modality,
        fontsize=20,
        plot_kwargs={"vmax": hmax},
    )
    plot_group_power_map(
        gpower_ap,
        filename=os.path.join(SAVE_DIR, "power_map_ap.png"),
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        data_space=data_space,
        modality=modality,
        fontsize=20,
        plot_kwargs={"vmax": hmax}
    )
    plot_group_power_map(
        gpower_diff,
        filename=os.path.join(SAVE_DIR, "power_map_diff.png"),
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        data_space=data_space,
        modality=modality,
    )

    print("Visualization complete.")
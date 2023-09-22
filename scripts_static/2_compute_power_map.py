"""Static power map computation using static PSDs

"""

# Set up dependencies
import os
import pickle
from utils.analysis import SubjectStaticPowerMap


if __name__ == "__main__":
    # Set hyperparameters
    modality = "eeg"
    data_space = "source"
    freq_range = [1, 45]
    band_name = "wide"
    verbose = True
    print(f"[INFO] Modality: {modality.upper()} | Data Space: {data_space} | " + 
          "Frequency Band: {band_name} ({freq_range[0]}-{freq_range[1]} Hz)")

    # Set directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results/static"
    DATA_DIR = os.path.join(BASE_DIR, f"{modality}/{data_space}_psd")
    SAVE_DIR = os.path.join(BASE_DIR, f"{modality}/power_{data_space}_{band_name}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    with open(os.path.join(DATA_DIR, f"psd.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    
    freqs = data["freqs"]
    psds = data["psd"]
    n_subjects = psds.shape[0]

    print("PSD shape: ", psds.shape)
    print(f"PSD loaded. Total {n_subjects} subjects")

    # Initiate save object
    output = {"freqs": freqs}

    # Initiate class object
    PM = SubjectStaticPowerMap(freqs, psds)

    # Plot subject-level PSDs
    if verbose:
        PM.plot_psd(filename=os.path.join(SAVE_DIR, "subject_psds.png"))

    # Compute and save power maps
    print(f"Computing power maps ({band_name.upper()}: {freq_range[0]}-{freq_range[1]} Hz) ...")
    power_maps = PM.compute_power_map(freq_range=freq_range)
    output["power_maps"] = power_maps

    # Save results
    with open(os.path.join(SAVE_DIR, "power.pkl"), "wb") as output_path:
        pickle.dump(output, output_path)
    output_path.close()

    print("Power map computation complete.")
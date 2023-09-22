"""Static AEC computation using osl-dynamics

"""

# Set up dependencies
import os
import pickle
from utils.analysis import compute_aec


if __name__ == "__main__":
    # Set hyperparameters
    modality = "eeg"
    data_space = "source"
    frequency_band = [1, 45]
    band_name = "wide"
    print(f"[INFO] Modality: {modality.upper()} | Data Space: {data_space} | " + 
          f"Frequency Band: {band_name} ({frequency_band[0]}-{frequency_band[1]} Hz)")

    # Set directory paths
    BASE_DIR = "/home/scho/AnalyzeNTAD/results"
    DATA_DIR = "/ohba/pi/mwoolrich/scho/NTAD"
    SAVE_DIR = os.path.join(BASE_DIR, f"static/{modality}/aec_{data_space}_{band_name}")
    TMP_DIR = os.path.join(SAVE_DIR, "tmp")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    if data_space == "sensor":
        dataset_dir = os.path.join(DATA_DIR, f"preproc")
    if data_space == "source":
        dataset_dir = os.path.join(DATA_DIR, f"src")

    # Calculate subject-specific AEC
    print("Computing first-level AEC ...")
    conn_map = compute_aec(
        dataset_dir=dataset_dir,
        data_space=data_space,
        modality=modality,
        sampling_frequency=250,
        freq_range=frequency_band,
        tmp_dir=TMP_DIR,
    )

    # Save results
    output = {"conn_maps": conn_map}
    with open(SAVE_DIR + "/aec.pkl", "wb") as output_path:
        pickle.dump(output, output_path)
    output_path.close()

    print("AEC computation complete.")
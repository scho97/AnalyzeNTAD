"""Find common EEG sensors

"""

import os
import pickle
import pandas as pd


if __name__ == "__main__":
    # Set directory paths
    BASE_DIR = "/home/scho/AnalyzeNTAD"
    SAVE_DIR = os.path.join(BASE_DIR, "results/data")

    # Load EEG layout data
    file_path = os.path.join(BASE_DIR, "scripts_data/eeg_layout_info.xlsx")
    easycap60 = pd.read_excel(file_path, sheet_name="EasyCap60")
    easycap64 = pd.read_excel(file_path, sheet_name="EasyCap64")
    easycap70 = pd.read_excel(file_path, sheet_name="EasyCap70")

    # Drop the last four channels from EasyCap70
    easycap70 = easycap70[:-4] # to stay consistent with preprocessing

    # Adjust channel numbering in EasyCap70
    easycap70["ch_num"][-6:] = [61, 62, 63, 64, 65, 66]
    # NOTE: For EasyCap70 used in Cambridge, channels are numbered until 60 and 
    #       jumps to 65 (with no 61-64). To account for this, we renumber the channels.


    # Find common sensors based on standard 10-20 channel labels
    common_ch_name = list(set(easycap60["ch_name"]) & set(easycap64["ch_name"]) & set(easycap70["ch_name"]))

    # Get common sensor indices
    find_common_idx = lambda df: df.loc[df["ch_name"].isin(common_ch_name)]["ch_num"].to_numpy() - 1
    ec60_idx = find_common_idx(easycap60)
    ec64_idx = find_common_idx(easycap64)
    ec70_idx = find_common_idx(easycap70)

    if (len(ec60_idx) != len(ec64_idx)) or (len(ec60_idx) != len(ec70_idx)):
        raise ValueError("length of common sensor indices must be the same.")
    
    # Get common sensor labels
    common_ch_name = easycap60["ch_name"][ec60_idx].to_list()
    # NOTE: We extract common channel names again to preserve the ordering of channels
    #       that matches with indices.

    # Save results
    output = {
        "EasyCap60": ec60_idx,
        "EasyCap64": ec64_idx,
        "EasyCap70": ec70_idx,
        "common_ch_name": common_ch_name,
    }
    with open(SAVE_DIR + "/common_eeg_sensor.pkl", "wb") as output_path:
        pickle.dump(output, output_path)
    output_path.close()

    print("Computation completed.")
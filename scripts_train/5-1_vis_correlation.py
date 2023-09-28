"""Visualize correlations between matched state/mode time courses

"""

# Set up dependencies
import os
import pickle
from osl_dynamics.inference import modes
from utils.data import load_order
from utils.visualize import plot_correlations


if __name__ == "__main__":
    # Set hyperparameters
    run_ids = {
        "eeg": {"hmm": "run0_hmm", "dynemo": "run1_dynemo"},
        "meg": {"hmm": "run0_hmm", "dynemo": "run0_dynemo"},
    }

    # Set directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results"
    DATA_DIR = os.path.join(BASE_DIR, "dynamic")
    os.makedirs(DATA_DIR, exist_ok=True)
    SAVE_DIR = DATA_DIR

    # Load data
    time_course = dict(eeg = dict(hmm = [], dynemo = []),
                       meg = dict(hmm = [], dynemo= []))
    for mod in ["eeg", "meg"]:
        if mod == "eeg":
            data_name = "ntad_eeg"
        else: data_name = "ntad_meg"
        for mdl in ["hmm", "dynemo"]:
            data_dir = os.path.join(DATA_DIR, f"{data_name}/{mdl}/{run_ids[mod][mdl]}")
            data_path = os.path.join(data_dir, f"model/results/{data_name}_{mdl}.pkl")
            with open(data_path, "rb") as input_path:
                run_data = pickle.load(input_path)
            input_path.close()
            tc = run_data["alpha"]
            if mdl == "hmm":
                tc = modes.argmax_time_courses(tc)
            order = load_order(run_ids[mod][mdl], mod)
            if order is not None:
                print("Reordering reference state/mode time courses ...")
                tc = [arr[:, order] for arr in tc]
            time_course[mod][mdl] = tc

    # Plot correlations across modality
    plot_correlations(time_course["eeg"]["hmm"],
                      time_course["meg"]["hmm"],
                      filename=os.path.join(SAVE_DIR, "corr_eeg_meg_hmm.png"),
                      colormap="RdGy_r")
    plot_correlations(time_course["eeg"]["dynemo"],
                      time_course["meg"]["dynemo"],
                      filename=os.path.join(SAVE_DIR, "corr_eeg_meg_dynemo.png"),
                      colormap="RdGy_r")

    # Plot correlations across model within modality
    plot_correlations(time_course["eeg"]["hmm"],
                      time_course["eeg"]["dynemo"],
                      filename=os.path.join(SAVE_DIR, "corr_eeg_hmm_dynemo.png"))
    plot_correlations(time_course["meg"]["hmm"],
                      time_course["meg"]["dynemo"],
                      filename=os.path.join(SAVE_DIR, "corr_meg_hmm_dynemo.png"))

    print("Visualization complete.")
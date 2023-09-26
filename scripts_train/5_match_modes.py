"""Match states and modes across model types and datasets

"""

# Set up dependencies
import os
import pickle
import numpy as np
from osl_dynamics.inference import modes
from utils.visualize import plot_correlations


if __name__ == "__main__":
    # Set directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results/dynamic"
    EEG_DIR = os.path.join(BASE_DIR, "ntad_eeg")
    MEG_DIR = os.path.join(BASE_DIR, "ntad_meg")

    # Load data of the best runs
    def load_data(data_dir, run_id):
        # Get model type
        model_type = run_id.split('_')[-1]
        data_name = data_dir.split('/')[-1]
        # Set path to the data
        data_path = os.path.join(data_dir, f"{model_type}/{run_id}/model/results/{data_name}_{model_type}.pkl")
        # Load data
        with open(data_path, "rb") as input_path:
            run_data = pickle.load(input_path)
        input_path.close()
        return run_data

    eeg_hmm = load_data(EEG_DIR, "run0_hmm")
    eeg_dynemo = load_data(EEG_DIR, "run1_dynemo")
    meg_hmm = load_data(MEG_DIR, "run0_hmm")
    meg_dynemo = load_data(MEG_DIR, "run0_dynemo")

    # Extract alphas
    eeg_hmm_alpha = eeg_hmm["alpha"]
    eeg_dynemo_alpha = eeg_dynemo["alpha"]
    meg_hmm_alpha = meg_hmm["alpha"]
    meg_dynemo_alpha = meg_dynemo["alpha"]

    # Compute HMM state time courses
    eeg_hmm_stc = modes.argmax_time_courses(eeg_hmm_alpha)
    meg_hmm_stc = modes.argmax_time_courses(meg_hmm_alpha)

    # Concatenate state/alpha time courses subject-wise
    cat_eeg_hmm = np.concatenate(eeg_hmm_stc, axis=0)
    cat_eeg_dynemo = np.concatenate(eeg_dynemo_alpha, axis=0)
    print("[NTAD EEG]")
    print("\tShape of HMM state time courses: ", np.shape(cat_eeg_hmm))
    print("\tShape of Dynemo mode time courses: ", np.shape(cat_eeg_dynemo))

    cat_meg_hmm = np.concatenate(meg_hmm_stc, axis=0)
    cat_meg_dynemo = np.concatenate(meg_dynemo_alpha, axis=0)
    print("[NTAD MEG]")
    print("\tShape of HMM state time courses: ", np.shape(cat_meg_hmm))
    print("\tShape of Dynemo mode time courses: ", np.shape(cat_meg_dynemo))

    # [1] Align NTAD EEG and MEG states (matched by eye)
    order_eeg_hmm = [5, 2, 7, 4, 1, 0, 3, 6]
    order_meg_hmm = [0, 7, 4, 2, 5, 1, 6, 3]
    print("EEG STC: ", order_eeg_hmm)
    print("EEG STC -> MEG STC: ", order_meg_hmm)
    # NOTE: At the present stage, matching by eye is preferred when matching states
    # or modes across modalities due to underperformance of the exiting algorithms.

    # [2] Apply state orders
    eeg_hmm_stc = [stc[:, order_eeg_hmm] for stc in eeg_hmm_stc]
    meg_hmm_stc = [stc[:, order_meg_hmm] for stc in meg_hmm_stc]

    # [3] Align reordered EEG states and original EEG modes
    cat_eeg_hmm = np.concatenate(eeg_hmm_stc, axis=0)
    _, order_eeg_dynemo = modes.match_modes(cat_eeg_hmm, cat_eeg_dynemo, return_order=True)
    print("EEG STC -> EEG ATC: ", order_eeg_dynemo)

    # [4] Align reordered MEG states and original MEG modes
    cat_meg_hmm = np.concatenate(meg_hmm_stc, axis=0)
    _, order_meg_dynemo = modes.match_modes(cat_meg_hmm, cat_meg_dynemo, return_order=True)
    print("MEG STC -> MEG ATC: ", order_meg_dynemo)

    # [5] Apply mode orders
    eeg_dynemo_alpha = [alpha[:, order_eeg_dynemo] for alpha in eeg_dynemo_alpha]
    meg_dynemo_alpha = [alpha[:, order_meg_dynemo] for alpha in meg_dynemo_alpha]

    # Plot the correlations of matched time courses
    plot_verbose = True
    if plot_verbose:
        plot_correlations(
            eeg_hmm_stc,
            meg_hmm_stc,
            filename=os.path.join(BASE_DIR, "match_eeg_meg_hmm.png"),
        )
        plot_correlations(
            eeg_hmm_stc,
            eeg_dynemo_alpha,
            filename=os.path.join(BASE_DIR, "match_eeg_hmm_dynemo.png"),
        )
        plot_correlations(
            meg_hmm_stc,
            meg_dynemo_alpha,
            filename=os.path.join(BASE_DIR, "match_meg_hmm_dynemo.png"),
        )

    print("Matching complete.")
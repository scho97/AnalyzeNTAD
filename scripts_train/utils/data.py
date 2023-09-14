"""Functions to handle and inspect data

"""

import os
import numpy as np

def get_free_energy(data_dir, run_ids, data_type="eeg", model="hmm"):
    """Load free energies from given paths to model runs.

    Parameters
    ----------
    data_dir : str
        Directory path that contains all the model run data.
    run_ids: list of str
        List of sub-directory names within `data_dir` that store run data.
    data_type : str
        Type of modality. Defaults to "eeg".
    seed : int
        Type of the dynamic model used. Defaults to "hmm".

    Returns
    -------
    F : list
        Free energies from each run.
    """

    # Validation
    if data_type not in ["eeg", "meg"]:
        raise ValueError("data_type needs to be either 'eeg' or 'meg'.")
    if model not in ["hmm", "dynemo"]:
        raise ValueError("model needs to be either 'hmm' or 'dynemo'.")
    
    # Define dataset name
    if data_type == "eeg":
        dataset = "lemon"
    elif data_type == "meg":
        dataset = "camcan"
    print(f"[{model.upper()} Model] Loading free energies from {len(run_ids)} runs ({data_type.upper()} {dataset.upper()})...")
    
    # Load free energy
    F = []
    for run_id in run_ids:
        filepath = os.path.join(data_dir, f"{model}/{run_id}/model/results/free_energy.npy")
        print(f"\tReading file: {filepath}")
        free_energy = np.load(filepath)
        F.append(free_energy)
    
    return F
"""Functions to handle and inspect data

"""

import os
import numpy as np
import pandas as pd

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

def load_order(run_dir, modality):
    """Extract a state/mode order of a given run written on the
       excel sheet. This order can be used to match the states/
       modes of a run to those of the reference run.

    Parameters
    ----------
    run_dir : str
        Name of the directory containing the model run (e.g., "run6_hmm").
    modality : str
        Type of the modality. Should be either "eeg" or "meg".

    Returns
    -------
    order : list of int
        Order of the states/modes matched to the reference run.
        Shape is (n_states,). If there is no change in order, None is
        returned.
    """

    # Define model type and run ID
    model_type = run_dir.split("_")[-1]
    run_id = int(run_dir.split("_")[0][3:])
    
    # Get list of orders
    BASE_DIR = "/home/scho/AnalyzeNTAD"
    df = pd.read_excel(os.path.join(BASE_DIR, "scripts_dynamic/run_orders.xlsx"))

    # Extract the order of a given run
    index = np.logical_and.reduce((
        df.Modality == modality,
        df.Model == model_type,
        df.Run == run_id,
    ))
    order = df.Order[index].values[0]
    convert_to_list = lambda x: [int(n) for n in x[1:-1].split(',')]
    order = convert_to_list(order)
    if order == list(np.arange(8)):
        order = None
    
    return order
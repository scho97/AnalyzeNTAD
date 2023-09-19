"""Evaluate pre-trained HMM on the full NTAD dataset

"""

# Set up dependencies
import os
import glob
import pickle
import numpy as np
from sys import argv
from osl_dynamics import data
from osl_dynamics.models.hmm import Model


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("Step 1 - Setting up ...")

    # Set run name
    if len(argv) != 3:
        raise ValueError("Need to pass two arguments: run name, modality (e.g., python script.py 1_hmm eeg)")
    run = argv[1] # run ID
    modality = argv[2] # modality type
    print(f"[INFO] Input Run ID: {run} | Modality: {modality.upper()}")

    # Set output directory path
    BASE_DIR = "/home/scho/AnalyzeNTAD/results"
    output_dir = f"{BASE_DIR}/dynamic/ntad_{modality}/hmm/run{run}"
    os.makedirs(output_dir, exist_ok=True)

    # Set output sub-directory paths
    analysis_dir = f"{output_dir}/analysis"
    model_dir = f"{output_dir}/model"
    maps_dir = f"{output_dir}/maps"
    tmp_dir = f"{output_dir}/tmp"
    save_dir = f"{model_dir}/results"
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Define pre-trained dataset to use
    if modality == "eeg":
        pretrain_data_name = "lemon"
    if modality == "meg":
        pretrain_data_name = "camcan"

    # --------------- [2] --------------- #
    #      Prepare training dataset       #
    # ----------------------------------- #
    print("Step 2 - Preparing training dataset ...")

    # Load data
    dataset_dir = "/ohba/pi/mwoolrich/scho/NTAD/src"
    file_names = sorted(glob.glob(
        os.path.join(dataset_dir, modality, "*/sflip_parc-raw.fif")
    ))
    subject_ids = [file.split("/")[-2] for file in file_names]
    print(f"Total number of subjects available: {len(file_names)}")

    # Get principal components used for the pre-training
    pca_components = np.load(os.path.join(BASE_DIR, f"dynamic/{pretrain_data_name}/pca_components.npy"))

    # Prepare the data for training
    training_data = data.Data(
        file_names,
        picks=["misc"],
        reject_by_annotation="omit",
        store_dir=tmp_dir,
        use_tfrecord=True,
    )
    prepare_config = {
        "tde_pca": {"n_embeddings": 15, "pca_components": pca_components},
        "standardize": {},
    }
    training_data.prepare(methods=prepare_config)

    # ------------ [3] ------------- #
    #      Build the HMM model       #
    # ------------------------------ #
    print("Step 3 - Building model ...")

    # Get the best pre-trained model
    if modality == "eeg":
        best_run_id = 8
    if modality == "meg":
        best_run_id = 3    
    best_model_path = os.path.join(
        BASE_DIR,
        f"dynamic/{pretrain_data_name}/hmm/run{best_run_id}_hmm/model/trained_model"
    )

    # Load pre-trained model weights
    print("Loading pre-trained model weights ...")
    model = Model.load(best_model_path)
    model.summary()

    # -------- [4] ---------- #
    #      Save results       #
    # ----------------------- #
    print("Step 4 - Saving results ...")

    # Make inference and get results
    free_energy = model.free_energy(training_data) # free energy
    alpha = model.get_alpha(training_data) # inferred state probabilities (equivalent to HMM gamma)
    tp = model.get_trans_prob() # inferred transition probability matrices
    cov = model.get_covariances() # inferred covariances
    ts = model.get_training_time_series(training_data, prepared=False) # subject-specific training data
    # NOTE: Transition probabilities and covariances should be identical to those of a pre-trained 
    #       model as no training is done for this model.

    print("Free energy: ", free_energy)
    
    # Save results
    outputs = {
        "subject_ids": subject_ids,
        "free_energy": free_energy,
        "alpha": alpha,
        "transition_probability": tp,
        "covariance": cov,
        "training_time_series": ts,
        "n_embeddings": training_data.n_embeddings,
    }

    with open(save_dir + "/ntad_hmm.pkl", "wb") as output_path:
        pickle.dump(outputs, output_path)
    output_path.close()

    np.save(save_dir + "/free_energy.npy", free_energy)

    # ------- [5] ------- #
    #      Clean up       #
    # ------------------- #
    training_data.delete_dir()

    print("Model training complete.")
"""Train DyNeMo on the full NTAD dataset

"""

# Set up dependencies
import os
import glob
import pickle
import numpy as np
from sys import argv
from osl_dynamics import data
from osl_dynamics.models.dynemo import Config, Model


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("Step 1 - Setting up ...")

    # Set run name
    if len(argv) != 3:
        raise ValueError("Need to pass two arguments: run name, modality (e.g., python script.py 1_dynemo eeg)")
    run = argv[1] # run ID
    modality = argv[2] # modality type
    print(f"[INFO] Input Run ID: {run} | Modality: {modality.upper()}")

    # Set output direcotry path
    BASE_DIR = "/home/scho/AnalyzeNTAD/results"
    output_dir = f"{BASE_DIR}/dynamic/ntad_{modality}/dynemo/run{run}"
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

    # Define training hyperparameters
    config = Config(
        n_modes=8,
        n_channels=80,
        sequence_length=200,
        inference_n_units=64,
        inference_normalization="layer",
        model_n_units=64,
        model_normalization="layer",
        learn_alpha_temperature=True,
        initial_alpha_temperature=1.0,
        learn_means=False,
        learn_covariances=False,
        do_kl_annealing=False,
        batch_size=64,
        learning_rate=5e-4,
        n_epochs=5,
    )
    # NOTE: No KL annealing is performed. KL annealing factor is fixed to 
    #       1 throughout the training, as we wnat to freeze the model RNN 
    #       parameters and fine tune the inference RNN.

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
    model = Model(config)
    model.summary()

    # Get the best pre-trained model
    if modality == "eeg":
        best_run_id = 7
    if modality == "meg":
        best_run_id = 9
    best_model_path = os.path.join(
        BASE_DIR,
        f"dynamic/{pretrain_data_name}/dynemo/run{best_run_id}_dynemo/model/trained_model/weights"
    )

    # Load pre-trained model weights
    print("Loading pre-trained model weights ...")
    model.load_weights(best_model_path).expect_partial()
    model.compile() # reset the optimizer

    # Define layers to fix (i.e., make non-trainable)
    fixed_layers = [
        "means",
        "covs",
        "mod_rnn",
        "mod_mu",
        "mod_sigma",
    ]
    # NOTE: Freeze the parameters of the model RNN and observation model.

    # -------------- [4] -------------- #
    #      Train the DyNeMo model       #
    # --------------------------------- #
    print("Step 4 - Training the model ...")

    # Train the model on a full dataset
    with model.set_trainable(fixed_layers, False):
        model.summary() # after freezing parameters
        history = model.fit(training_data)

    # Save the trained model
    model.save(f"{model_dir}/trained_model")

    # Save training history
    with open(f"{model_dir}/history.pkl", "wb") as file:
        pickle.dump(history, file)

    # --------- [5] --------- #
    #      Save results       #
    # ----------------------- #
    print("Step 5 - Saving results ...")

    # Get results
    ll_loss = np.array(history["ll_loss"])
    kl_loss = np.array(history["kl_loss"])
    loss = ll_loss + kl_loss # training loss
    free_energy = model.free_energy(training_data) # free energy
    alpha = model.get_alpha(training_data) # inferred mixing coefficients for each subject
    cov = model.get_covariances() # inferred covariances (equivalent to D in the paper)
    ts = model.get_training_time_series(training_data, prepared=False) # subject-specific training data
    # NOTE: Covariances should be identical to those of a pre-trained 
    #       model as no training was done for this model.

    print("Final loss: ", loss[-1])
    print("Free energy: ", free_energy)

    # Save results
    outputs = {
        "subject_ids": subject_ids,
        "ll_loss": ll_loss,
        "kl_loss": kl_loss,
        "loss": loss,
        "free_energy": free_energy,
        "alpha": alpha,
        "covariance": cov,
        "training_time_series": ts,
        "n_embeddings": training_data.n_embeddings,
    }

    with open(save_dir + f"/ntad_{modality}_dynemo.pkl", "wb") as output_path:
        pickle.dump(outputs, output_path)
    output_path.close()

    np.save(save_dir + "/free_energy.npy", free_energy)

    # ------- [6] ------- #
    #      Clean up       #
    # ------------------- #
    training_data.delete_dir()

    print("Model training complete.")
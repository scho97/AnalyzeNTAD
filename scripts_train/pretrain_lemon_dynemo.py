"""Pre-train DyNeMo on the full EEG LEMON dataset

"""

# Set up dependencies
import os
import glob
import pickle
import numpy as np
from sys import argv
from osl_dynamics import data
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.dynemo import Config, Model


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("Step 1 - Setting up ...")

    # Set run name
    if len(argv) != 2:
        raise ValueError("Need to pass one argument: run name (e.g., python script.py 1_dynemo)")
    run = argv[1] # run ID

    # Set up GPU
    tf_ops.gpu_growth()

    # Set output direcotry path
    BASE_DIR = "/well/woolrich/users/olt015/AnalyzeNTAD/results"
    output_dir = f"{BASE_DIR}/dynamic/lemon/dynemo/run{run}"
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
        learn_covariances=True,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=10,
        n_kl_annealing_epochs=40,
        batch_size=64,
        learning_rate=5e-4,
        n_epochs=60,
    )

    # --------------- [2] --------------- #
    #      Prepare training dataset       #
    # ----------------------------------- #
    print("Step 2 - Preparing training dataset ...")

    # Load data
    dataset_dir = "/well/woolrich/projects/lemon/scho23/src_ec"
    file_names = sorted(glob.glob(dataset_dir + "/*/sflip_parc-raw.npy"))
    print(f"Total number of subjects available: {len(file_names)}")

    # Prepare the data for training
    training_data = data.Data(file_names, store_dir=tmp_dir, use_tfrecord=True)
    prepare_config = {
        "tde_pca": {"n_embeddings": 15, "n_pca_components": config.n_channels},
        "standardize": {},
    }
    training_data.prepare(methods=prepare_config)

    # ------------ [3] ------------- #
    #      Build the HMM model       #
    # ------------------------------ #
    print("Step 3 - Building model ...")
    model = Model(config)
    model.summary()

    # -------------- [4] -------------- #
    #      Train the DyNeMo model       #
    # --------------------------------- #
    print("Step 4 - Training the model ...")

    # Initialization
    print("Initializing weights based on the random subset of a dataset ...")
    init_history = model.random_subset_initialization(training_data, n_init=10, n_epochs=4, take=0.5)
    with open(f"{model_dir}/init_history.pkl", "wb") as file:
        pickle.dump(init_history, file)

    # Train the model on a full dataset
    history = model.fit(
        training_data,
        save_best_after=config.n_kl_annealing_epochs,
        save_filepath=f"{model_dir}/weights"
    )

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

    print("Final loss: ", loss[-1])
    print("Free energy: ", free_energy)

    # Save results
    outputs = {
        "ll_loss": ll_loss,
        "kl_loss": kl_loss,
        "loss": loss,
        "free_energy": free_energy,
        "alpha": alpha,
        "covariance": cov,
        "training_time_series": ts,
        "n_embeddings": training_data.n_embeddings,
    }

    with open(save_dir + "/lemon_dynemo.pkl", "wb") as output_path:
        pickle.dump(outputs, output_path)
    output_path.close()

    np.save(save_dir + "/free_energy.npy", free_energy)

    # ------- [6] ------- #
    #      Clean up       #
    # ------------------- #
    training_data.delete_dir()

    print("Model training complete.")
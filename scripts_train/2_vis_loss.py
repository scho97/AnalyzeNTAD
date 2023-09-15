"""Visualise HMM / DyNeMo loss curve

"""

# Set up dependencies
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from matplotlib.ticker import MaxNLocator
from osl_dynamics.utils import plotting


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 4:
        print("Need to pass three arguments: dataset name, model, run # (e.g., python script.py lemon hmm 1)")
        exit()
    dataset_name = argv[1]
    model_type = argv[2]
    run_id = argv[3]
    print(f"[INFO] Dataset: {dataset_name.upper()} | Model: {model_type.upper()} | Run: run{run_id}_{model_type}")

    # Get model run directory
    run_dir = f"run{run_id}_{model_type}"

    # Set up directories
    BASE_DIR = "/home/scho/AnalyzeNTAD/results/dynamic"
    DATA_DIR = os.path.join(BASE_DIR, f"{dataset_name}/{model_type}/{run_dir}/model/results")
    SAVE_DIR = os.path.join(BASE_DIR, f"{dataset_name}/{model_type}/{run_dir}/analysis")

    # Load loss values
    with open(os.path.join(DATA_DIR, f"{dataset_name}_{model_type}.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    if model_type == "hmm":
        loss = data["loss"]
        x_step = 5
        if len(loss) < x_step * 2:
            x_step = 1
    else:
        loss = data["loss"]
        ll_loss = data["ll_loss"]
        kl_loss = data["kl_loss"]
        x_step = 10
        if len(loss) < x_step * 2:
            x_step = 5
    epochs = np.arange(1, len(loss) + 1)

    # Plot training loss curve
    fig, ax = plotting.plot_line([epochs], [loss], plot_kwargs={"lw": 2})
    ax.set_xticks(np.arange(0, len(loss) + x_step, x_step))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(axis='both', which='both', labelsize=18, width=2)
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel("Epochs", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss.png"))
    plt.close(fig)

    if model_type == "dynemo":
        # Plot DyNeMo loss components
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
        ax[0].plot(epochs, ll_loss, color="tab:red", lw=3)
        ax[1].plot(epochs, kl_loss, color="tab:green", lw=3)
        for i in range(2):
            ax[i].set_xlabel("Epochs", fontsize=20)
            ax[i].set_xticks([0, len(epochs)])
            ax[i].yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            ax[i].tick_params(axis='both', which='both', labelsize=20, width=3)
            for axis in ['top','bottom','left','right']:
                ax[i].spines[axis].set_linewidth(3)
        ax[0].set_ylabel("Loss", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "loss_ll_kl.png"))
        plt.close(fig)

    print("Visualization complete.")
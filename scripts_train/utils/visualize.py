"""Functions for visualization

"""

import numpy as np
import matplotlib.pyplot as plt

def plot_free_energy(data, modality, filename):
    """Plot free energies for each model type across multiple runs.

    Parameters
    ----------
    data : dict
        Free energy values.
    modality : str
        Type of the data modality. Should be either "eeg" or "meg".
    filename : str
        Path for saving the figure.
    """

    # Validation
    y1 = np.array(data["hmm"][modality])
    y2 = np.array(data["dynemo"][modality])
    if len(y1) != len(y2):
        raise ValueError("number of runs should not be different between two models.")
    x = np.arange(len(y1)) + 1
    
    # Visualize line plots
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))
    for i, dat in enumerate([y1, y2]):
        vmin, vmax = dat.min(), dat.max()
        vmin -= (vmax - vmin) * 0.1
        vmax += (vmax - vmin) * 0.1
        ax[i].plot(x, dat, marker='o', linestyle='--', lw=2, markersize=10, color=plt.cm.Set2(i))
        ax[i].set_ylim([vmin, vmax])
        yticks = np.round(np.linspace(vmin, vmax, 3), 2)
        ax[i].set(
            xticks=x,
            yticks=yticks,
            xticklabels=x,
            yticklabels=np.char.mod("%.2f", yticks),
        )
    ax[0].set_title("HMM")
    ax[1].set_title("DyNeMo")
    ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[1].set_xlabel("Runs")
    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

    return None
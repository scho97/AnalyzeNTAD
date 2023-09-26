"""Functions for visualization

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from osl_dynamics.utils import plotting

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

def plot_correlations(data1, data2, filename, colormap="coolwarm"):
    """Computes correlation between the input data and plots 
    their correlation matrix.

    Parameters
    ----------
    data1 : np.ndarray or list of np.ndarray
        A data array containing multiple variables and observations.
        Shape can be (n_samples, n_modes) or (n_subjects, n_samples, n_modes).
    data2 : np.ndarray or list of np.ndarray
        A data array containing multiple variables and observations.
        Shape can be (n_samples, n_modes) or (n_subjects, n_samples, n_modes).
    filename : str
        Path for saving the figure.
    colormap : str
        Type of a colormap to use. Defaults to "coolwarm".
    """

    # Validation
    if len(data1) != len(data2):
        raise ValueError("length of input data sould be the same.")
    if type(data1) != type(data2):
        raise ValueError("type of input data should be the same.")
    if not isinstance(data1, list):
        data1, data2 = [data1], [data2]

    # Get data dimensions
    n_subjects = len(data1)
    n_modes1 = data1[0].shape[1]
    n_modes2 = data2[0].shape[1]
    min_samples = np.min(np.vstack((
        [d.shape[0] for d in data1],
        [d.shape[0] for d in data2],
    )), axis=0)
    
    # Match data lengths
    data1 = np.concatenate(
        [data1[n][:min_samples[n], :] for n in range(n_subjects)]
    )
    data2 = np.concatenate(
        [data2[n][:min_samples[n], :] for n in range(n_subjects)]
    )

    # Compute correlations between the data
    corr = np.corrcoef(data1, data2, rowvar=False)[n_modes1:, :n_modes2]

    # Plot correlation matrix
    fig, _ = plotting.plot_matrices(corr, cmap=colormap)
    ax = fig.axes[0] # to remain compatible with `osl-dynamics.plotting`
    im = ax.findobj()[0]
    vmax = np.max(np.abs(corr))
    im.set_clim([-vmax, vmax]) # make a symmetric colorbar
    ax.set(
        xticks=np.arange(0, n_modes1),
        xticklabels=np.arange(1, n_modes1 + 1),
        yticks=np.arange(0, n_modes2),
        yticklabels=np.arange(1, n_modes2 + 1),
    )
    ax.tick_params(labelsize=14, bottom=False, right=False)
    im.colorbar.ax.tick_params(labelsize=14)
    im.colorbar.ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
    im.colorbar.ax.yaxis.offsetText.set_fontsize(14)
    cbar_pos = im.colorbar.ax.get_position()
    im.colorbar.ax.set_position(
        Bbox([[cbar_pos.x0 - 0.05, cbar_pos.y0], [cbar_pos.x1 - 0.05, cbar_pos.y1]])
    )
    fig.savefig(filename)
    plt.close(fig)

    return None
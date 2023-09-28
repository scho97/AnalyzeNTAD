"""Functions for visualization

"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from osl_dynamics import analysis
from osl_dynamics.utils import plotting
from utils.array_ops import round_nonzero_decimal, round_up_half
from utils.statistics import fit_glm, cluster_perm_test
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_correlations(data1, data2, filename):
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
    fig, _ = plotting.plot_matrices(corr, cmap="coolwarm")
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
    cbar_pos = im.colorbar.ax.get_position()
    im.colorbar.ax.set_position(
        Bbox([[cbar_pos.x0 - 0.05, cbar_pos.y0], [cbar_pos.x1 - 0.05, cbar_pos.y1]])
    )
    fig.savefig(filename)
    plt.close(fig)

    return None

def plot_grouped_violin(data, group_label, method_name, filename, ylbl=None, pval=None):
    """Plots grouped violins.

    Parameters
    ----------
    data : np.ndarray or list
        Input data. Shape must be (n_subjects, n_features).
    group_label : list of str
        List containing group labels for each subject.
    method_name : str
        Type of the model used for getting the input data. Must be either
        "hmm" or "dynemo".
    filename : str
        Path for saving the figure.
    ylbl : str
        Y-axis tick label. Defaults to None.
    pval : str
        P-values for each violin indicating staticial differences between
        the groups. If provided, statistical significance is plotted above the
        violins. Defaults to None.
    """

    # Validation
    if isinstance(data, list):
        data = np.array(data)
    if method_name == "hmm": lbl = "State"
    elif method_name == "dynemo": lbl = "Mode"
    print("Plotting grouped violin plot ...")
    
    # Number of features
    n_features = data.shape[1]

    # Build dataframe
    data_flatten = np.reshape(data, data.size, order='F')
    df = pd.DataFrame(data_flatten, columns=["Statistics"])
    df["Group"] = group_label * n_features
    df[lbl] = np.concatenate([np.ones((data.shape[0],)) * n for n in range(n_features)])

    # Plot grouped split violins
    fig, ax = plt.subplots(nrows=1, ncols=1)
    vp = sns.violinplot(data=df, x=lbl, y="Statistics", hue="Group",
                        split=True, inner="box", linewidth=1,
                        palette={"AN": "b", "AP": "r"}, ax=ax)
    if pval is not None:
        vmin, vmax = [], []
        for collection in vp.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                vmin.append(np.min(collection.get_paths()[0].vertices[:, 1]))
                vmax.append(np.max(collection.get_paths()[0].vertices[:, 1]))
        vmin = np.min(np.array(vmin).reshape(-1, 2), axis=1)
        vmax = np.max(np.array(vmax).reshape(-1, 2), axis=1)
        ht = (vmax - vmin) * 0.045
        for i, p in enumerate(pval):
            p_lbl = categrozie_pvalue(p)
            if p_lbl != "n.s.":
                ax.text(
                    vp.get_xticks()[i], 
                    vmax[i] + ht[i],
                    p_lbl, 
                    ha="center", va="center", color="k", 
                    fontsize=15, fontweight="bold"
                )
    # sns.despine(fig=fig, ax=ax) # get rid of top and right axes
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] + np.max(vmax - vmin) * 0.05])
    ax.set(
        xticks=np.arange(n_features),
        xticklabels=np.arange(n_features) + 1,
    )
    ax.set_xlabel(f"{lbl}s", fontsize=18)
    ax.set_ylabel(ylbl, fontsize=18)
    ax.tick_params(labelsize=18)
    ax.get_legend().remove()
    # vp.legend(fontsize=10, bbox_to_anchor=(1.01, 1.15))
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

    return None

def categrozie_pvalue(pval):
    """Assigns a label indicating statistical significance that corresponds 
    to an input p-value.

    Parameters
    ----------
    pval : float
        P-value from a statistical test.

    Returns
    -------
    p_label : str
        Label representing a statistical significance.
    """ 

    thresholds = [1e-3, 0.01, 0.05]
    labels = ["***", "**", "*", "n.s."]
    ordinal_idx = np.max(np.where(np.sort(thresholds + [pval]) == pval)[0])
    # NOTE: use maximum for the case in which a p-value and threshold are identical
    p_label = labels[ordinal_idx]

    return p_label

def plot_power_map(
    power_map,
    mask_file,
    parcellation_file,
    filename,
    subtract_mean=False,
    mean_weights=None,
    colormap=None,
    fontsize=22,
):
    """Saves power maps. Wrapper for `osl_dynamics.analysis.power.save()`.

    Parameters
    ----------
    power_map : np.ndarray
        Power map to save. Can be of shape: (n_components, n_modes, n_channels),
        (n_modes, n_channels) or (n_channels,). A (..., n_channels, n_channels)
        array can also be passed. Warning: this function cannot be used if n_modes
        is equal to n_channels.
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filename : str
        Path for saving the power map.
    subtract_mean : bool
        Should we subtract the mean power across modes?
        Defaults to False.
    mean_weights : np.ndarray
        Numpy array with weightings for each mode to use to calculate the mean.
        Default is equal weighting.
    colormap : str
        Colors for connectivity edges. If None, a default colormap is used 
        ("cold_hot").
    fontsize : int
        Fontsize for a powre map colorbar. Defaults to 22.
    """

    # Set visualisation parameters
    if colormap is None:
        colormap = "cold_hot"

    # Plot power map
    figures, axes = analysis.power.save(
        power_map=power_map,
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        subtract_mean=subtract_mean,
        mean_weights=mean_weights,
        plot_kwargs={"cmap": colormap},
    )
    for i, fig in enumerate(figures):
        # Reset figure size
        fig.set_size_inches(5, 6)
        # Change colorbar position
        cbar_ax = axes[i][-1]
        pos = cbar_ax.get_position()
        new_pos = [pos.x0 * 0.92, pos.y0 + 0.02, pos.width * 1.20, pos.height * 1.10]
        cbar_ax.set_position(new_pos)
        # Edit colobar ticks
        if np.any(np.abs(np.array(cbar_ax.get_xlim())) < 1):
            hmin = round_nonzero_decimal(cbar_ax.get_xlim()[0], method="ceil") # ceiling for negative values
            hmax = round_nonzero_decimal(cbar_ax.get_xlim()[1], method="floor") # floor for positive values
            cbar_ax.set_xticks(np.array([hmin, 0, hmax]))
            cbar_ax.tick_params(labelsize=fontsize)
        else:
            cbar_ax.set_xticks(
                [round_up_half(val) for val in cbar_ax.get_xticks()[1:-1]]
            )
            cbar_ax.tick_params(labelsize=fontsize)
        # Set colorbar styles
        cbar_ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 6))
        cbar_xticks = cbar_ax.get_xticks()
        if (cbar_xticks[-1] >= 0.1) and (cbar_xticks[-1] < 1):
            cbar_ax.set_xticklabels([str(xt) if xt != 0 else "0" for xt in cbar_xticks])
        cbar_ax.xaxis.offsetText.set_fontsize(fontsize)
        if len(figures) > 1:
            fig.savefig(filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{i}"), bbox_inches="tight")
        else:
            fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    return None

def plot_connectivity_map(
    conn_map,
    parcellation_file,
    filename,
    colormap=None,
):
    """Saves connectivity maps. Wrapper for `osl_dynamics.analysis.connectivity.save()`.

    Parameters
    ----------
    conn_map : np.ndarray
        Matrices containing connectivity strengths to plot.
        Shape must be (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filename : str
        Path for saving the power map.
    colormap : str
        Colors for connectivity edges. If None, a default colormap is used 
        ("bwr").
    """

    # Validation
    if conn_map.ndim == 2:
        conn_map = conn_map[np.newaxis, ...]

    # Number of states/modes
    n_modes = conn_map.shape[0]

    # Set visualisation parameters
    if colormap is None:
        colormap = "bwr"
    
    # Plot connectivity map
    for n in range(n_modes):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        analysis.connectivity.save(
            connectivity_map=conn_map[n],
            parcellation_file=parcellation_file,
            plot_kwargs={"edge_cmap": colormap, "figure": fig, "axes": ax},
        )
        cb_ax = fig.get_axes()[-1]
        pos = cb_ax.get_position()
        new_pos = [pos.x0 * 1.05, pos.y0, pos.width, pos.height]
        cb_ax.set_position(new_pos)
        cb_ax.tick_params(labelsize=20)
        if n_modes != 1:
            fig.savefig(
                filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{n}"),
                transparent=True
            )
        else:
            fig.savefig(filename, transparent=True)
        plt.close(fig)

    return None

def plot_rsn_psd(f, psd_mean, psd_se, edges=None, filename=None, fontsize=22):
    """Plots state/mode-specific PSDs averaged over channels.

    Parameters
    ----------
    f : np.ndarray
        Frequencies of the power spectra.
    psd_mean : np.ndarray
        Power spectra averaged over subjects. Shape is (n_states, n_channels, 
        n_freqs).
    psd_se : np.ndarray
        Standard errors of power spectra over subjects. Shape is (n_states, 
        n_channels, n_freqs).
    edges : boolean array
        A boolean array marking specific brain regions. Shape must be 
        (n_modes, n_channels, n_channels). Defaults to None. If given,
        PSDs will be averaged over for selected channels.
    filename : str
        Path for saving the power map. Defaults to None, in which case the 
        figure will be saved in a current directory.
    fontsize : int
        Fontsize for axes ticks and labels. Defaults to 22.
    """
    # Validation
    if filename is None:
        filename = os.getcwd()

    # Number of states/modes
    n_class = psd_mean.shape[0]

    # Select PSDs of parcels with significant connection strengths
    psds, stes = [], []
    vmin, vmax = 0, 0
    for n in range(n_class):
        if edges is not None:
            parcel_idx = np.unique(np.concatenate(np.where(edges[n] == True)))
            mode_psd_mean = np.squeeze(psd_mean[n, parcel_idx, :])
            mode_psd_se = np.squeeze(psd_se[n, parcel_idx, :])
        else:
            mode_psd_mean = np.squeeze(psd_mean[n])
            mode_psd_se = np.squeeze(psd_se[n])
        psds.append(np.mean(mode_psd_mean, axis=0)) # average over parcels
        stes.append(np.mean(mode_psd_se, axis=0))
        vmin = np.min([vmin, np.min(psds[n] - stes[n])])
        vmax = np.max([vmax, np.max(psds[n] + stes[n])])
    vmin = vmin - (vmax - vmin) * 0.10
    vmax = vmax + (vmax - vmin) * 0.10
    hmin, hmax = 0, np.ceil(f[-1]) + 1

    # Plot averaged PSDs and their standard errors
    for n in range(n_class):
        fig, ax = plotting.plot_line(
            [f],
            [psds[n]],
            errors=[[psds[n] - stes[n]], [psds[n] + stes[n]]],
            x_range=[hmin, hmax],
            y_range=[vmin, vmax],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
        )
        # Reset figure size
        fig.set_size_inches(6, 4)
        # Set axes ticks
        ax.set_xticks(np.arange(hmin, hmax, 10))
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
        # Set axes tick style
        ax.tick_params(labelsize=fontsize)
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
        ax.yaxis.offsetText.set_fontsize(fontsize)
        if n_class != 1:
            fig.savefig(filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{n}"), bbox_inches="tight")
        else:
            fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)

    return None

def plot_mode_spectra_group_diff(f, psd, subject_ids, group_assignments, method, modality, bonferroni_ntest, filename):
    """Plots state/mode-specific PSDs and their between-group statistical differences.

    This function tests statistical differences using a cluster permutation test on the
    frequency axis.

    Parameters
    ----------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freqs,).
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape must be (n_subjects,
        n_states, n_channels, n_freqs).
    subject_ids : list of str
        Subject IDs corresponding to the input data.
    group_assignments : list of lists
        1D array containing gruop labels for input subjects. A value of 1 indicates
        Group 1 (amyloid positive w/ MCI & AD) and a value of 2 indicates Group 2 
        (amyloid negative).
    method : str
        Type of the dynamic model. Can be "hmm" or "dynemo".
    modality : str
        Type of the neuroimaging modality. Can be "eeg" or "meg".
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction. If None, Bonferroni
        correction will not take place.
    filename : str
        Path for saving the figure.
    test_type : str
        Type of the cluster permutation test function to use. Should be "mne"
        or "glmtools" (default).
    """

    # Set plot labels
    if method == "hmm":
        lbl = "State"
    elif method == "dynemo":
        lbl = "Mode"

    # Get group-averaged PSDs
    psd_model, _, _ = fit_glm(
        psd,
        subject_ids,
        group_assignments,
        dimension_labels=["Subjects", "States/Modes", "Channels", "Frequency"],
    )
    gpsd_an = psd_model.betas[1]
    gpsd_ap = psd_model.betas[0]
    # dim: (n_modes, n_channels, n_freqs)

    # Build a colormap
    qcmap = plt.rcParams["axes.prop_cycle"].by_key()["color"] # qualitative

    # Plot mode-specific PSDs and their statistical difference
    if psd.shape[1] > 1:
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(26, 10))
        k, j = 0, 0 # subplot indices
        for n in range(len(gpsd_an)):
            print(f"Plotting {lbl} {n + 1}")
            
            # Set the row index
            if (n % 4 == 0) and (n != 0):
                k += 1
            
            # Perform cluster permutation tests on mode-specific PSDs
            cpsd = np.mean(psd, axis=2)
            cpsd_model, cpsd_design, cpsd_data = fit_glm(
                cpsd[:, n, :],
                subject_ids,
                group_assignments,
                dimension_labels=["Subjects", "Frequency"],
            )
            t_obs, clu_idx = cluster_perm_test(
                cpsd_model,
                cpsd_data,
                cpsd_design,
                pooled_dims=(1,),
                contrast_idx=0,
                n_perm=1500,
                metric="tstats",
                bonferroni_ntest=bonferroni_ntest,
            )
            n_clusters = len(clu_idx)

            # Average group-level PSDs over the parcels
            p_an = np.mean(gpsd_an[n], axis=0)
            p_ap = np.mean(gpsd_ap[n], axis=0)
            e_an = np.std(gpsd_an[n], axis=0) / np.sqrt(gpsd_an.shape[0])
            e_ap = np.std(gpsd_ap[n], axis=0) / np.sqrt(gpsd_ap.shape[0])

            # Plot mode-specific group-level PSDs
            ax[k, j].plot(f, p_an, c=qcmap[n], label="AN")
            ax[k, j].plot(f, p_ap, c=qcmap[n], label="AP", linestyle="--")
            ax[k, j].fill_between(f, p_an - e_an, p_an + e_an, color=qcmap[n], alpha=0.1)
            ax[k, j].fill_between(f, p_ap - e_ap, p_ap + e_ap, color=qcmap[n], alpha=0.1)
            if n_clusters > 0:
                for c in range(n_clusters):
                    ax[k, j].axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)

            # Set labels
            ax[k, j].set_xlabel("Frequency (Hz)", fontsize=18)
            if j == 0:
                ax[k, j].set_ylabel("PSD (a.u.)", fontsize=18)
            ax[k, j].set_title(f"{lbl} {n + 1}", fontsize=18)
            ax[k, j].ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
            ax[k, j].tick_params(labelsize=18)
            ax[k, j].yaxis.offsetText.set_fontsize(18)

            # Plot observed statistics
            end_pt = np.mean([p_an[int(len(p_an) // 3):], p_ap[int(len(p_ap) // 3):]])
            criteria = np.mean([ax[k, j].get_ylim()[0], ax[k, j].get_ylim()[1]])
            if end_pt >= criteria:
                inset_bbox = (0, -0.22, 1, 1)
            if end_pt < criteria:
                inset_bbox = (0, 0.28, 1, 1)
            ax_inset = inset_axes(ax[k, j], width='40%', height='30%', 
                                loc='center right', bbox_to_anchor=inset_bbox,
                                bbox_transform=ax[k, j].transAxes)
            ax_inset.plot(f, t_obs, color='k', lw=2) # plot t-spectra
            for c in range(len(clu_idx)):
                ax_inset.axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)
            ax_inset.set(
                xticks=np.arange(0, max(f), 20),
                ylabel="t-stats",
            )
            ax_inset.set_ylabel('t-stats', fontsize=16)
            ax_inset.tick_params(labelsize=16)

            # Set the column index
            j += 1
            if (j % 4 == 0) and (j != 0):
                j = 0

        plt.subplots_adjust(hspace=0.5)
        fig.savefig(filename)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        for n in range(len(gpsd_an)):
            print(f"Plotting {lbl} {n + 1}")

            # Perform cluster permutation tests on mode-specific PSDs
            cpsd = np.mean(psd, axis=2)
            cpsd_model, cpsd_design, cpsd_data = fit_glm(
                cpsd[:, n, :],
                subject_ids,
                group_assignments,
                dimension_labels=["Subjects", "Frequency"],
            )
            t_obs, clu_idx = cluster_perm_test(
                cpsd_model,
                cpsd_data,
                cpsd_design,
                pooled_dims=(1,),
                contrast_idx=0,
                n_perm=1500,
                metric="tstats",
                bonferroni_ntest=bonferroni_ntest,
            )
            n_clusters = len(clu_idx)

            # Average group-level PSDs over the parcels
            p_an = np.mean(gpsd_an[n], axis=0)
            p_ap = np.mean(gpsd_ap[n], axis=0)
            e_an = np.std(gpsd_an[n], axis=0) / np.sqrt(gpsd_an.shape[0])
            e_ap = np.std(gpsd_ap[n], axis=0) / np.sqrt(gpsd_ap.shape[0])

            # Plot mode-specific group-level PSDs
            ax.plot(f, p_an, c="k", label="AN")
            ax.plot(f, p_ap, c="k", label="AP", linestyle="--")
            ax.fill_between(f, p_an - e_an, p_an + e_an, color="k", alpha=0.1)
            ax.fill_between(f, p_ap - e_ap, p_ap + e_ap, color="k", alpha=0.1)
            if n_clusters > 0:
                for c in range(n_clusters):
                    ax.axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)

            # Set labels
            ax.set_xlabel("Frequency (Hz)", fontsize=16)
            ax.set_ylabel("PSD (a.u.)", fontsize=16)
            ax.set_title(f"Static mean across {lbl.lower()}s", fontsize=16)
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
            ax.tick_params(labelsize=16)
            ax.yaxis.offsetText.set_fontsize(16)

            # Plot observed statistics
            end_pt = np.mean([p_an[-1], p_ap[-1]])
            criteria = np.mean([ax.get_ylim()[0], ax.get_ylim()[1] * 0.95])
            if end_pt >= criteria:
                inset_bbox = (0, -0.22, 1, 1)
            if end_pt < criteria:
                inset_bbox = (0, 0.28, 1, 1)
            ax_inset = inset_axes(ax, width='40%', height='30%', 
                                loc='center right', bbox_to_anchor=inset_bbox,
                                bbox_transform=ax.transAxes)
            ax_inset.plot(f, t_obs, color='k', lw=2) # plot t-spectra
            for c in range(len(clu_idx)):
                ax_inset.axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)
            ax_inset.set_ylabel('t-statistics', fontsize=14)
            ax_inset.tick_params(labelsize=14)

        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)

    return None

def plot_pow_vs_coh(freqs, psd, coh, subject_ids, group_assignments, method, modality, filenames, freq_range = None, legend=False):
    """Saves a scatter plot of group-level power and coherence values for each group.

    Parameters
    ----------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freqs,).
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape must be (n_subjects,
        n_states, n_channels, n_freqs).
    coh : np.ndarray
        Coherences for each state/mode. Shape is (n_subjects, n_states, n_channels,
        n_channels, n_freqs).
    subject_ids : list of str
        Subject IDs corresponding to the input data.
    group_assignments : list of lists
        1D array containing gruop labels for input subjects. A value of 1 indicates
        Group 1 (amyloid positive w/ MCI & AD) and a value of 2 indicates Group 2 
        (amyloid negative).
    method : str
        Type of the dynamic model. Can be "hmm" or "dynemo".
    modality : str
        Type of the neuroimaging modality. Can be "eeg" or "meg".
    filenames : list of str
        Paths for saving the figures.
    freq_range : list of int
        Frequency range (in Hz) to integrate the PSD and coherence over.
        Defaults to None, which integrates over the full range.
    legend : bool
        Whether to plot the legend box. Defaults to False.
    """

    # Set plot labels
    if method == "hmm":
        lbl = "State"
    elif method == "dynemo":
        lbl = "Mode"

    # Compute power of each ROI for each mode
    po = analysis.power.variance_from_spectra(freqs, psd, frequency_range=freq_range)
    # dim: (n_subjects, n_states, n_channels)

    # Compute sum of coherences between a given ROI and all others
    co = analysis.connectivity.mean_coherence_from_spectra(freqs, coh, frequency_range=freq_range)
    # dim: (n_subjects, n_modes, n_channels, n_channels)
    sum_co = analysis.connectivity.mean_connections(co)
    # dim: (n_subjects, n_modes, n_channels)

    # Get group-specific PSDs and coherences
    po_model, _, _ = fit_glm(
        po,
        subject_ids,
        group_assignments,
        dimension_labels=["Subjects", "States/Modes", "Channels"],
    )
    sum_co_model, _, _ = fit_glm(
        sum_co,
        subject_ids,
        group_assignments,
        dimension_labels=["Subjects", "States/Modes", "Channels"],
    )
    pos = [po_model.betas[1], po_model.betas[0]]
    sum_cos = [sum_co_model.betas[1], sum_co_model.betas[0]]

    # Get axis limits
    hmin, hmax = _get_lim(pos)
    vmin, vmax = _get_lim(sum_cos)

    # Plot the relationships between PSDs and coherences
    for g in range(len(pos)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        for n in range(pos[g].shape[0]): # iterate across states/modes
            ax.scatter(pos[g][n], sum_cos[g][n], alpha=0.6, label=f"{lbl} {n + 1}")
        ax.set_xlim([hmin, hmax])
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel("Power (a.u.)", fontsize=18)
        ax.set_ylabel("Coherence", fontsize=18)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=18)
        if legend:
            ax.legend(bbox_to_anchor=(1.5, 0.5))
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(filenames[g])
        plt.close(fig)

    return None

def _get_lim(data, scale=0.1):
    """Get lower and upper limits of the input data.

    Parameters
    ----------
    data : np.ndarray or list
        Data to compute minimum and maximum from.
    scale : float
        Scaling factor to adjust minimum and maximum values.
        Defaults to 0.1.

    Returns
    -------
    minimum : float
        Minimum axis limit.
    maximum : float
        Maximum axis limit.
    """

    diff = np.abs(np.max(data) - np.min(data))
    minimum = np.min(data) - scale * diff
    maximum = np.max(data) + scale * diff

    return minimum, maximum

def plot_thresholded_map(tstats, pvalues, map_type, mask_file, parcellation_file, filenames):
   """Plot a thresholded power or functional connectivity map.

    Parameters
    ----------
    tstats : np.ndarray
        Statistic observed for all variables. Shape should be (n_features,).
    pvalues : np.ndarray
        P-values for the features. Shape should be (n_features,).
    map_type : str
        Type of the map used to compute t-statistics. Can be "power" 
        or "connectivity".
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filenames : list of str
        Paths for saving the unthresholded and thresholded maps, respectively.
    """
   
   # Get indices of significant parcels/connections
   thr_idx = pvalues < 0.05

   # Plot original and thresholded t-statistics values
   if np.any(thr_idx):
      print("Significant parcels identified under Bonferroni-corrected p=0.05.\n" +
            "Plotting Results ...")
      if map_type == "power":
         # Plot unthresholded t-map
         plot_power_map(
            tstats,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            subtract_mean=False,
            mean_weights=None,
            colormap="RdBu_r",
            filename=filenames[0],
            fontsize=24,
         )
         # Plot thresholded t-map
         tstats_sig = np.zeros((tstats.shape))
         tstats_sig[thr_idx] = tstats[thr_idx]
         print("\tSelected parcels: ", np.arange(len(tstats))[thr_idx])
         plot_power_map(
            tstats_sig,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            subtract_mean=False,
            mean_weights=None,
            colormap="RdBu_r",
            filename=filenames[1],
            fontsize=24,
         )
      elif map_type == "connectivity":
         n_parcels = np.ceil(np.sqrt(len(tstats) * 2)).astype(int)
         i, j = np.triu_indices(n_parcels, 1)
         # Plot unthresholded t-map
         tmap = np.zeros((n_parcels, n_parcels))
         tmap[i, j] = tstats
         tmap += tmap.T
         tmap = analysis.connectivity.threshold(tmap, absolute_value=True, percentile=97)
         plot_connectivity_map(
            tmap,
            parcellation_file=parcellation_file,
            colormap="RdBu_r",
            filename=filenames[0],
         )
         # Plot thresholded t-map
         tstats_sig = np.zeros((tstats.shape))
         tstats_sig[thr_idx] = tstats[thr_idx]
         print("\tSelected connections: ", np.arange(len(tstats))[thr_idx])
         tmap = np.zeros((n_parcels, n_parcels))
         tmap[i, j] = tstats_sig
         tmap += tmap.T
         plot_connectivity_map(
            tmap,
            parcellation_file=parcellation_file,
            colormap="RdBu_r",
            filename=filenames[1],
         )
   else:
      print("No significant parcels identified under Bonferroni-corrected p=0.05.")

   return None
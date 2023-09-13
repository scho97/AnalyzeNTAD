"""Functions for visualization

"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_age_distributions(ages_young, ages_old, data_name="", nbins="auto", bar_label=False, save_dir=""):
        """Plots an age distribution of each group as a histogram.

        Parameters
        ----------
        ages_x1 : list or np.ndarray
            Ages of Group 1. Shape is (n_subjects,)
        ages_x2 : list or np.ndarray
            Ages of Group 2. Shape is (n_subjects,)
        data_name : str
            Name of the dataset. Defaults to an empty string. If a name is provided, 
            it will be appended to the saved figure path.
        nbins : str, int, or list
            Number of bins to use for each histograms. Different nbins can be given
            for each age group in a list form. Defaults to "auto". Can take options
            described in `numpy.histogram_bin_edges()`.
        bar_label : bool
            Whether to print the number of counts at the top of each bar. Defaults 
            to False.
        save_dir : str
            Path to a directory in which the plot should be saved. By default, the 
            plot will be saved to a user's current directory.
        """

        # Validation
        if not isinstance(nbins, list):
            nbins = [nbins, nbins]

        # Set visualization parameters
        cmap = sns.color_palette("deep")
        sns.set_style("white")

        # Sort ages for ordered x-tick labels
        ages_young, ages_old = sorted(ages_young), sorted(ages_old)

        # Plot histograms
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
        sns.histplot(x=ages_young, ax=ax[0], color=cmap[0], bins=nbins[0])
        sns.histplot(x=ages_old, ax=ax[1], color=cmap[3], bins=nbins[1])
        ax[0].set_title(f"Amyloid Negative (n={len(ages_young)})")
        ax[1].set_title(f"Amyloid Positive (n={len(ages_old)})")
        for i in range(2):
            ax[i].set_xlabel("Age")
            ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            if bar_label:
                for n in ax[i].containers:
                    ax[i].bar_label(n)
                ylim = list(ax[i].get_ylim())
                ylim[-1] += 5
                ax[i].set_ylim(ylim)
        plt.suptitle(f"{data_name.upper()} Age Distribution")
        plt.tight_layout()
        save_name = "age_dist.png"
        if data_name:
            save_name = save_name.replace(".png", f"_{data_name}.png")
        fig.savefig(os.path.join(save_dir, save_name))
        plt.close(fig)

        return None
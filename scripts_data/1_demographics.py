"""Analyze NTAD participant data

"""

import os
import numpy as np
import pandas as pd
from utils.data import get_subject_ids
from utils.statistics import stat_ind_two_samples
from utils.visualize import plot_age_distributions


if __name__ == "__main__":
    # Set directory paths
    BASE_DIR = "/home/scho/AnalyzeNTAD"
    DATA_DIR = "/ohba/pi/mwoolrich/scho/NTAD"
    SRC_DIR = os.path.join(DATA_DIR, "src")
    SAVE_DIR = os.path.join(BASE_DIR, "results/data")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load meta data
    filepath = os.path.join(BASE_DIR, "scripts_data/all_data_info.xlsx")
    df_meta = pd.read_excel(filepath) # contains subject demographics

    # Get subjects with source reconstructed data
    subjects, n_subjects = get_subject_ids(SRC_DIR, "meg")
    print(f"Number of subjects available: {n_subjects}")

    # Filter dataframe with subjects available
    mask = df_meta["ID"].isin(subjects)
    df = df_meta[mask]

    # Separate data by groups
    df_an = df[df["disease"] == 0] # amyloid-negative (healthy)
    df_ap = df[np.logical_or(df["disease"] == 1, df["disease"] == 2)] # amyloid-positive with mild cognitive impariments or AD
    print(f"Number of participants: {len(df)} | AN = {len(df_an)} | AP = {len(df_ap)}")

    # Select variables of interest
    dep_var = ["Fixed_Age", "MMSE", "education", "Hippo_Vol_Norm", "Brain_Vol", "GM_Vol_Norm", "WM_Vol_Norm"]

    # ----------- [1] ------------ #
    #      Statistical Tests       #
    # ---------------------------- #

    # Explore descriptive statistics
    mean = df.groupby(["Class"])[dep_var].mean()
    std = df.groupby(["Class"])[dep_var].std(ddof=0) # stay consistent with numpy
    print("\n*** Mean of numeric data ***\n", mean)
    print("\n*** Standard deviation of numeric data ***\n", std)

    # Count the number of subjects by sex in each group
    print("\n*** Information on participant sex in NTAD groups ***")
    for group_name in ["control", "patient"]:
        n_female = df[df["Class"] == group_name]["Sex (1=female, 2=male)"].tolist().count(1)
        n_male = df[df["Class"] == group_name]["Sex (1=female, 2=male)"].tolist().count(2)
        print(f"\t[{group_name.capitalize()}] M/F: {n_male}/{n_female}")

    # Conduct statistical tests comparing AP and AN groups
    dep_var = ["Fixed_Age", "MMSE", "education", "Hippo_Vol_Norm", "Brain_Vol", "GM_Vol_Norm", "WM_Vol_Norm"]
    for var_name in dep_var:
        print(f"Testing the variable {var_name.upper()}")
        var_an = df_an[var_name].copy().to_numpy()
        var_ap = df_ap[var_name].copy().to_numpy()
        if any(np.isnan(var_an)):
            print(f"{np.sum(np.isnan(var_an))} NaN values excluded from Group AN.")
            var_an = var_an[~np.isnan(var_an)]
        if any(np.isnan(var_ap)):
            print(f"{np.sum(np.isnan(var_ap))} NaN values excluded from Group AP.")
            var_ap = var_ap[~np.isnan(var_ap)]
        stat, pval = stat_ind_two_samples(var_an, var_ap)

    # ------------ [2] ------------ #
    #      Data Visualization       #
    # ----------------------------- #

    # Plot age distributions of two groups
    ages_an = df_an["Fixed_Age"].to_numpy()
    ages_ap = df_ap["Fixed_Age"].to_numpy()

    plot_age_distributions(
        ages_an,
        ages_ap,
        data_name="ntad",
        nbins=[
            [55, 60, 65, 70, 75, 80],
            [55, 60, 65, 70, 75, 80]
        ], # enforce equal bins for each group
        bar_label=True,
        save_dir=SAVE_DIR,
    )

    print("Analysis complete.")
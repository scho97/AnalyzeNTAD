"""Functions for statistical tests and validations

"""

import warnings
import mne
import numpy as np
from scipy import stats

def cluster_perm_test(x1, x2, bonferroni_ntest=None):
    """Wrapper for mne.stats.permutation_cluster_test.
    This function performs a cluter permutaiton test on 2D arrays.

    Parameters
    ----------
    x1 : np.ndarray
        PSD of the first group. Shape must be (n_subjects, n_channels, n_freqs).
    x2 : np.ndarray
        PSD of the second group. Shape must be (n_subjects, n_channels, n_freqs).
    bonferroni_ntest : int
        Number of tests to be used for Bonferroni correction. Default to None.

    Returns
    -------
    t_obs : np.ndarray
        t-statistic values for all variables. Shape is (n_freqs,).
    clusters : list
        List of tuple of ndarray, each of which contains the indices that form the
        given cluster along the tested dimension. If bonferroni_ntest was given,
        clusters after Bonferroni correction are returned.
    cluster_pv : np.ndarray
        P-value for each cluster. If bonferroni_ntest was given, corrected p-values
        are returned.
    H0 : np.ndarray 
        Max cluster level stats observed under permutation.
        Shape is (n_permutations,)
    """

    # Average PSD over channels/parcels
    X = [
        np.mean(x1, axis=1),
        np.mean(x2, axis=1)
    ] # dim: (n_subjects, n_parcels, n_freqs) -> (n_subjects, n_freqs)

    # Perform cluster permutations over frequencies
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
        X,
        threshold=3, # cluster-forming threshold
        n_permutations=5000,
        tail=0,
        stat_fun=mne.stats.ttest_ind_no_p,
        adjacency=None,
    )

    # Apply Bonferroni correction
    if bonferroni_ntest:
        cluster_pv_corrected = np.array(cluster_pv) * bonferroni_ntest
        sel_idx = np.where(cluster_pv_corrected < 0.05)[0]
        clusters = [clusters[i] for i in sel_idx]
        cluster_pv = cluster_pv[sel_idx]
        print(f"After Boneferroni correction: Found {len(clusters)} clusters")
        print(f"\tCluster p-values: {cluster_pv}")

    return t_obs, clusters, cluster_pv, H0

def _check_stat_assumption(samples1, samples2, ks_alpha=0.05, ev_alpha=0.05):
    """Checks normality of each sample and whether samples have an equal variance.

    Parameters
    ----------
    samples1 : np.ndarray
        Array of sample data (group 1). Shape must be (n_samples,).
    samples2 : np.ndarray
        Array of sample data (group 2). Shape must be (n_samples,).
    ks_alpha : float
        Threshold to use for null hypothesis rejection in the Kolmogorov-Smirnov test.
        Defaults to 0.05.
    ev_alpha : float
        Threshold to use for null hypothesis rejection in the equal variance test.
        This test can be the Levene's test or Bartlett's test, depending on the 
        normality of sample distributions. Defaults to 0.05.

    Returns
    -------
    nm_flag : bool
        If True, both samples follow a normal distribution.
    ev_flag : bool
        If True, two sample gruops have an equal variance.
    """

    # Set flags for normality and equal variance
    nm_flag, ev_flag = True, True
    print("*** Checking Normality & Equal Variance Assumptions ***")

    # Check normality assumption
    ks_pvals = []
    for s, samples in enumerate([samples1, samples2]):
        stand_samples = stats.zscore(samples)
        res = stats.ks_1samp(stand_samples, cdf=stats.norm.cdf)
        ks_pvals.append(res.pvalue)
        print(f"\t[KS Test] p-value (Sample #{s}): {res.pvalue}")
        if res.pvalue < ks_alpha:
             print(f"\t[KS Test] Sample #{s}: Null hypothesis rejected. The data are not distributed " + 
                   "according to the standard normal distribution.")
    
    # Check equal variance assumption
    if np.sum([pval < ks_alpha for pval in ks_pvals]) != 0:
        nm_flag = False
        # Levene's test
        _, ev_pval = stats.levene(samples1, samples2)
        ev_test_name = "Levene's"
    else:
        # Bartlett's test
        _, ev_pval = stats.bartlett(samples1, samples2)
        ev_test_name = "Bartlett's"
    print(f"\t[{ev_test_name} Test] p-value: ", ev_pval)
    if ev_pval < ev_alpha:
        print(f"\t[{ev_test_name} Test] Null hypothesis rejected. The populations do not have equal variances.")
        ev_flag = False

    return nm_flag, ev_flag

def stat_ind_two_samples(samples1, samples2, bonferroni_ntest=None, test=None):
    """Performs a statistical test comparing two independent samples.

    Parameters
    ----------
    samples1 : np.ndarray
        Array of sample data (group 1). Shape must be (n_samples,).
    samples2 : np.ndarray
        Array of sample data (group 2). Shape must be (n_samples,).
    bonferroni_ntest : int
        Number of tests to be used for Bonferroni correction. Default to None.
    test : str
        Statistical test to use. Defaults to None, which automatically selects
        the test after checking the assumptions.

    Returns
    -------
    stat : float
        The test statistic. The test can be the Student's t-test, Welch's t-test, 
        or Wilcoxon Rank Sum test depending on the test assumptions.
    pval : float
        The p-value of the test. If bonferroni_ntest is given, the corrected 
        p-value is returned.
    """

    # Check normality and equal variance assumption
    if test is None:
        nm_flag, ev_flag = _check_stat_assumption(samples1, samples2)
    else:
        if test == "ttest":
            nm_flag, ev_flag = True, True
        elif test == "welch":
            nm_flag, ev_flag = True, False
        elif test == "wilcoxon":
            nm_flag, ev_flag = False, True

    # Compare two independent groups
    print("*** Comparing Two Independent Groups ***")
    if nm_flag and ev_flag:
        print("\tConducting the two-samples independent T-Test ...")
        stat, pval = stats.ttest_ind(samples1, samples2, equal_var=True)
    if nm_flag and not ev_flag:
        print("\tConducting the Welch's t-test ...")
        stat, pval = stats.ttest_ind(samples1, samples2, equal_var=False)
    if not nm_flag:
        print("\tConducting the Wilcoxon Rank Sum test ...")
        if not ev_flag:
            warnings.warn("Caution: Distributions have unequal variances.", UserWarning)
        stat, pval = stats.ranksums(samples1, samples2)
    print(f"\tResult: statistic={stat} | p-value={pval}")

    # Apply Bonferroni correction
    if bonferroni_ntest is not None:
        pval *= bonferroni_ntest
    print(f"[Bonferroni Correction] p-value={pval}")

    return stat, pval
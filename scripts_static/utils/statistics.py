"""Functions for statistical tests and validations

"""

import warnings
import mne
import numpy as np
import glmtools as glm
from scipy import stats
from utils.data import (get_subject_ids,
                        load_group_information,
                        load_site_information)

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

def fit_glm(
        input_data,
        modality,
        dimension_labels=None,
        plot_verbose=False,
        save_path=""
    ):
    """Fit a General Linear Model (GLM) to an input data given a design matrix.

    Parameters
    ----------
    input_data : np.ndarray
        Data to fit. Shape must be (n_subjects, n_features1, n_features2, ...).
    modality : str
        Type of neuroimaging modality.
    dimension_labels : list of str
        Labels for the dimensions of an input data. Defaults to None, in which 
        case the labels will set as ["Subjects", "Features1", "Features2", ...].
    plot_verbose : bool
        Whether to plot the deisign matrix. Defaults to False.
    save_path : str
        File path to save the design matrix plot. Relevant only when plot_verbose 
        is set to True.
    
    Returns
    -------
    model : glmtools.fit.OLSModel
        A fiited GLM OLS model.
    design : glmtools.design.DesignConfig
        Design matrix object for GLM modelling.
    glm_data : glmtools.data.TrialGLMData
        Data object for GLM modelling.
    """

    # Validation
    ndim = input_data.ndim
    if ndim == 1:
        raise ValueError("data must be 2D or greater.")

    if dimension_labels is None:
        dimension_labels = ["Subjects"] + [f"Features {i}" for i in range(1, ndim)]

    # Load group information
    SRC_DIR = "/ohba/pi/mwoolrich/scho/NTAD/src"
    subject_ids, n_subjects = get_subject_ids(SRC_DIR, modality)
    an_idx, ap_idx = load_group_information(subject_ids)
    print(f"Number of available subjects: {n_subjects} | AN={len(an_idx)} | AP={len(ap_idx)}")
    if n_subjects != (len(an_idx) + len(ap_idx)):
        raise ValueError("one or more groups lacking subjects.")

    # Define group assignments
    group_assignments = np.zeros((n_subjects,))
    group_assignments[ap_idx] = 1 # amyloid positive (w/ MCI & AD)
    group_assignments[an_idx] = 2 # amyloid negative (controls)

    # Define covariates (to regress out)
    site_assignments = load_site_information(subject_ids)
    covariates = {
        "Site (Oxford)": site_assignments,
        "Site (Cambridge)": 1 - site_assignments,
    }

    # Create GLM dataset
    glm_data = glm.data.TrialGLMData(
        data=input_data,
        **covariates,
        category_list=group_assignments,
        dim_labels=dimension_labels,
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Group1", rtype="Categorical", codes=1) # amyloid positive
    DC.add_regressor(name="Group2", rtype="Categorical", codes=2) # amyloid negative
    for name in covariates:
        DC.add_regressor(
            name=name,
            rtype="Parametric",
            datainfo=name,
            preproc="demean",
        )
    DC.add_contrast(
        name="GroupDiff",
        values=[1, -1] + [0] * len(covariates),
    ) # amyloid positive - amyloid negative
    DC.add_contrast(
        name="OverallMean",
        values=[0.5, 0.5] + [0] * len(covariates),
    )
    design = DC.design_from_datainfo(glm_data.info)
    if plot_verbose:
        design.plot_summary(show=False, savepath=save_path)

    # Fit GLM model
    model = glm.fit.OLSModel(design, glm_data)

    return model, design, glm_data

def max_stat_perm_test(
        glm_model,
        glm_data,
        design_matrix,
        pooled_dims,
        contrast_idx,
        n_perm=10000,
        metric="tstats",
        n_jobs=1,
        return_perm=False,
    ):
    """Perform a max-t permutation test to evaluate statistical significance 
       for the given contrast.

    Parameters
    ----------
    glm_model : glmtools.fit.OLSModel
        A fitted GLM OLS model.
    glm_data : glmtools.data.TrialGLMData
        Data object for GLM modelling.
    design_matrix : glmtools.design.DesignConfig
        Design matrix object for GLM modelling.
    pooled_dims : int or tuples
        Dimension(s) to pool over.
    contrast_idx : int
        Index indicating which contrast to use. Dependent on glm_model.
    n_perm : int, optional
        Number of iterations to permute. Defaults to 10,000.
    metric : str, optional
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    n_jobs : int, optional
        Number of processes to run in parallel.
    return_perm : bool, optional
        Whether to return a glmtools permutation object. Defaults to False.
    
    Returns
    -------
    pvalues : np.ndarray
        P-values for the features. Shape is (n_features1, n_features2, ...).
    perm : glm.permutations.MaxStatPermutation
        Permutation object in the `glmtools` package.
    """

    # Run permutations and get null distributions
    perm = glm.permutations.MaxStatPermutation(
        design_matrix,
        glm_data,
        contrast_idx=contrast_idx,
        nperms=n_perm,
        metric=metric,
        tail=0, # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    if metric == "tstats":
        print("Using tstats as metric")
        tstats = abs(glm_model.tstats[0])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        print("Using copes as metric")
        copes = abs(glm_model.copes[0])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    if return_perm:
        return pvalues, perm
    return pvalues

def cluster_perm_test(
        glm_model,
        glm_data,
        design_matrix,
        pooled_dims,
        contrast_idx,
        n_perm=1500,
        metric="tstats",
        bonferroni_ntest=1,
        n_jobs=1,
        return_perm=False,
    ):
    """Perform a cluster permutation test to evaluate statistical significance 
       for the given contrast.

    Parameters
    ----------
    glm_model : glmtools.fit.OLSModel
        A fitted GLM OLS model.
    glm_data : glmtools.data.TrialGLMData
        Data object for GLM modelling.
    design_matrix : glmtools.design.DesignConfig
        Design matrix object for GLM modelling.
    pooled_dims : int or tuples
        Dimension(s) to pool over.
    contrast_idx : int
        Index indicating which contrast to use. Dependent on glm_model.
    n_perm : int
        Number of iterations to permute. Defaults to 1,500.
    metric : str, optional
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction. Defaults to 1 (i.e., no
        Bonferroni correction applied).
    n_jobs : int, optional
        Number of processes to run in parallel.
    return_perm : bool, optional
        Whether to return a glmtools permutation object. Defaults to False.
    
    Returns
    -------
    obs : np.ndarray
        Statistic observed for all variables. Values can be 'tstats' or 'copes'
        depending on the `metric`. Shape is (n_freqs,).
    clusters : list of np.ndarray
        List of ndarray, each of which contains the indices that form the given 
        cluster along the tested dimension. If bonferroni_ntest was given, clusters 
        after Bonferroni correction are returned.
    perm : glm.permutations.ClusterPermutation
        Permutation object in the `glmtools` package.
    """

    # Get metric values and define cluster forming threshold
    if metric == "tstats":
        obs = np.squeeze(glm_model.tstats)
        cft = 3
    if metric == "copes":
        obs = np.squeeze(glm_model.copes)
        cft = 0.001

    # Run permutations and get null distributions
    perm = glm.permutations.ClusterPermutation(
        design=design_matrix,
        data=glm_data,
        contrast_idx=contrast_idx,
        nperms=n_perm,
        metric=metric,
        tail=0, # two-sided test
        cluster_forming_threshold=cft,
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )

    # Extract significant clusters
    percentile = (1 - (0.05 / bonferroni_ntest)) * 100
    # NOTE: We use alpha threshold of 0.05.
    clu_masks, clu_stats = perm.get_sig_clusters(glm_data, percentile)
    if clu_stats is not None:
        n_clusters = len(clu_stats)
    else: n_clusters = 0
    print(f"After Boneferroni correction: Found {n_clusters} clusters")

    # Get indices of significant channels and frequencies
    clusters = [
        np.arange(len(clu_masks))[clu_masks == n]
        for n in range(1, n_clusters + 1)
    ]

    if return_perm:
        return obs, clusters, perm
    return obs, clusters

def cluster_perm_test_mne(x1, x2, bonferroni_ntest=None):
    """[***DEPRECATED***]
    Wrapper for mne.stats.permutation_cluster_test.
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
"""Functions for statistical tests and validations

"""

import numpy as np
import pandas as pd
import glmtools as glm
from scipy import stats
from utils.data import load_site_information, load_scanner_information

def fit_glm(
        input_data,
        subject_ids,
        group_assignments,
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
    subject_ids : list of str
        Subject IDs corresponding to the input data.
    group_assignments : np.ndarray
        1D array containing gruop labels for input subjects. A value of 1 indicates
        Group 1 (amyloid positive w/ MCI & AD) and a value of 2 indicates Group 2 
        (amyloid negative).
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
    
    # Load meta data
    df_meta = pd.read_excel(
        "/home/scho/AnalyzeNTAD/scripts_data/all_data_info.xlsx"
    )

    # Define covariates (to regress out)
    site_assignments = load_site_information(subject_ids)
    scanner_assignments = load_scanner_information(subject_ids, df_meta, modality)
    covariates = {
        "site": site_assignments,
        "scanner": scanner_assignments,
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
            preproc=None,
        )
    DC.add_contrast(
        name="GroupDiff",
        values=[1, -1] + [0] * len(covariates),
    ) # amyloid positive - amyloid negative
    DC.add_contrast(
        name="GroupMean",
        values=[0.5, 0.5] + [0] * len(covariates),
    )
    design = DC.design_from_datainfo(glm_data.info)
    if plot_verbose:
        design.plot_summary(show=False, savepath=save_path)

    # Fit GLM model
    model = glm.fit.OLSModel(design, glm_data)

    return model, design, glm_data

def fit_glm_confound_regression(
        input_data,
        subject_ids,
        modality,
        dimension_labels=None,
        plot_verbose=False,
        save_path=""
    ):
    """Fit a General Linear Model (GLM) to an input data given a design matrix
       to perofrm confound regression.

    Parameters
    ----------
    input_data : np.ndarray
        Data to fit. Shape must be (n_subjects, n_features1, n_features2, ...).
    subject_ids : list of str
        Subject IDs corresponding to the input data.
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
    
    # Load meta data
    df_meta = pd.read_excel(
        "/home/scho/AnalyzeNTAD/scripts_data/all_data_info.xlsx"
    )

    # Define covariates (to regress out)
    site_assignments = load_site_information(subject_ids)
    scanner_assignments = load_scanner_information(subject_ids, df_meta, modality)
    covariates = {
        "site": site_assignments,
        "scanner": scanner_assignments,
    }

    # Create GLM dataset
    glm_data = glm.data.TrialGLMData(
        data=input_data,
        **covariates,
        dim_labels=dimension_labels,
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Constant", rtype="Constant")
    for name in covariates:
        DC.add_regressor(
            name=name,
            rtype="Parametric",
            datainfo=name,
            preproc=None,
        )
    DC.add_contrast(
        name="Intercept",
        values=[1] + [0] * len(covariates),
    )
    DC.add_contrast(
        name="ArtefactEffect",
        values=[0, 1, 1],
    )
    DC.add_contrast(
        name="ArtefactMean",
        values=[1, 1, 1],
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
        obs = np.squeeze(glm_model.tstats[contrast_idx])
        cft = 3
    if metric == "copes":
        obs = np.squeeze(glm_model.copes[contrast_idx])
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
    percentile = (1 - (0.05 / (2 * bonferroni_ntest))) * 100
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

def detect_outliers(data, group_idx, group_labels=None):
    """Detects outliers from the data.

    This function first standardizes the data into a standard normal distribution
    and detects values outside [-3, 3]. The mean and standard deviation for the
    standardization is computed from the entire data. Outliers are then detected
    from each feature (column) vector and appended together.

    Parameters
    ----------
    data : np.ndarray
        The data to detect outliers from. Shape must be (n_subjects,)
        or (n_subjects, n_features).
    group_idx : list of list
        List containing subject indices of each group.
    group_lbl : list of str
        List of group labels. Defaults to None, in which case labels are 
        set to "AN" and "AP".

    Returns
    -------
    outlier_idx : np.ndarray or None
        Unique subject indices of the outliers. If no outliers are detected,
        returns None.
    outlier_lbl : list of str or None
        Labels for each outlier indicating which group it comes from. If no
        outliers are detected, returns None.
    """

    # Validation
    if isinstance(data, list):
        data = np.array(data)
    if data.ndim == 1:
        data = data[..., np.newaxis]
    n_features = data.shape[1]
    
    if group_labels is None:
        group_labels = ["AN", "AP"]
    
    # Standardize data
    z_scores = (data - np.mean(data)) / np.std(data)

    # Detect outliers
    outliers = []
    for n in range(n_features):
        outlier_flag = np.abs(z_scores[:, n]) > 3
        if np.any(outlier_flag):
            outliers.append([(idx, group_labels[0]) for idx in group_idx[0] if outlier_flag[idx] == True])
            outliers.append([(idx, group_labels[1]) for idx in group_idx[1] if outlier_flag[idx] == True])
    
    if outliers:
        outliers = [item for olr in outliers for item in olr]
        outlier_idx = list(list(zip(*outliers))[0])
        outlier_lbl = list(list(zip(*outliers))[1])
    else:
        outlier_idx, outlier_lbl = None, None

    # Exclude repeating outliers
    if outlier_idx is not None:
        outlier_idx, unique_idx = np.unique(outlier_idx, return_index=True)
        outlier_lbl = [outlier_lbl[idx] for idx in unique_idx]

    return outlier_idx, outlier_lbl
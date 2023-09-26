"""Perform maximum statistics non-parameteric permutation testing 
   on power and connectivity maps
"""

# Set up dependencies
import os
import pickle
import warnings
import numpy as np
from sys import argv
from osl_dynamics import analysis
from osl_dynamics.inference import modes
from utils.analysis import get_psd_coh
from utils.data import (load_order,
                        load_group_information,
                        load_outlier,
                        get_dynemo_mtc)
from utils.statistics import fit_glm, fit_glm_confound_regression, max_stat_perm_test
from utils.visualize import plot_thresholded_map


if __name__ == "__main__":
   # ------- [1] ------- #
   #      Settings       #
   # ------------------- #
   print("Step 1 - Setting up ...")

   # Set hyperparameters
   if len(argv) != 4:
      print("Need to pass three arguments: modality, model type, and run ID (e.g., python script.py eeg hmm 6)")
   modality = argv[1]
   model_type = argv[2]
   run_id = argv[3]
   print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()} | Run: run{run_id}_{model_type}")

   # Get state/mode orders for the specified run
   run_dir = f"run{run_id}_{model_type}"
   order = load_order(run_dir, modality)

   # Define training hyperparameters
   Fs = 250 # sampling frequency
   n_channels = 80 # number of channels
   if model_type == "hmm":
      n_class = 8 # number of states
      seq_len = 800 # sequence length for HMM training
   if model_type == "dynemo":
      n_class = 8 # number of modes
      seq_len = 200 # sequence length for DyNeMo training
   if modality == "eeg":
      data_name = "ntad_eeg"
   elif modality == "meg":
      data_name = "ntad_meg"

   # Set parcellation file paths
   mask_file = "MNI152_T1_8mm_brain.nii.gz"
   parcellation_file = (
      "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
   )

   # Set up directories
   BASE_DIR = "/home/scho/AnalyzeNTAD/results"
   DATA_DIR = os.path.join(BASE_DIR, f"dynamic/{data_name}/{model_type}/{run_dir}")

   # Load data
   with open(os.path.join(DATA_DIR, f"model/results/{data_name}_{model_type}.pkl"), "rb") as input_path:
      data = pickle.load(input_path)
   alpha = data["alpha"]
   cov = data["covariance"]
   ts = data["training_time_series"]

   # Load group information
   subject_ids = data["subject_ids"]
   n_subjects = len(subject_ids)
   an_idx, ap_idx = load_group_information(subject_ids)
   print("Total {} subjects | AN: {} | AP: {}".format(
      n_subjects, len(an_idx), len(ap_idx)
   ))

   # Define group assignments
   group_assignments = np.zeros((n_subjects,))
   group_assignments[ap_idx] = 1 # amyloid positive (w/ MCI & AD)
   group_assignments[an_idx] = 2 # amyloid negative (controls)

   # Validation
   if len(alpha) != n_subjects:
      warnings.warn(f"The length of alphas does not match the number of subjects. n_subjects reset to {len(alpha)}.")
      n_subjects = len(alpha)

   # ----------------- [2] ------------------- #
   #      Preprocess inferred parameters       #
   # ----------------------------------------- #
   print("Step 2 - Preparing state/mode time courses ...")

   # Reorder states or modes if necessary
   if order is not None:
      print(f"Reordering {modality.upper()} state/mode time courses ...")
      print(f"\tOrder: {order}")
      alpha = [a[:, order] for a in alpha] # dim: n_subjects x n_samples x n_modes
      cov = cov[order] # dim: n_modes x n_channels x n_channels

   # Get HMM state time courses
   if model_type == "hmm":
      btc = modes.argmax_time_courses(alpha)

   # Get DyNeMo mode activation time courses
   if model_type == "dynemo":
      btc = get_dynemo_mtc(alpha, Fs, data_dir=DATA_DIR)

   # --------- [3] --------- #
   #      Load Spectra       #
   # ----------------------- #
   print("Step 3 - Loading spectral information ...")
   
   # Set the number of CPUs to use for parallel processing
   n_jobs = 16

   # Calculate subject-specific PSDs and coherences
   if model_type == "hmm":
      print("Computing HMM multitaper spectra ...")
      f, psd, coh, w = get_psd_coh(
         ts, btc, Fs,
         calc_type="mtp",
         save_dir=DATA_DIR,
         n_jobs=n_jobs,
      )
   if model_type == "dynemo":
      print("Computing DyNeMo glm spectra ...")
      f, psd, coh, w = get_psd_coh(
         ts, alpha, Fs,
         calc_type="glm",
         save_dir=DATA_DIR,
         n_jobs=n_jobs,
      )

   # Exclude specified outliers
   if (modality == "eeg") and (model_type == "dynemo"):
      outlier_idx = load_outlier(run_dir, modality)
      print("Excluding subject outliers ...\n"
              "\tOutlier indices: ", outlier_idx)
      not_olr_idx = np.setdiff1d(np.arange(n_subjects), outlier_idx)
      btc = [btc[idx] for idx in not_olr_idx]
      psd = psd[not_olr_idx]
      coh = coh[not_olr_idx]
      print(f"\tPSD shape: {psd.shape} | Coherence shape: {coh.shape}")
      # Reorganize group assignments
      group_assignments = group_assignments[not_olr_idx]
      subject_ids = [subject_ids[idx] for idx in not_olr_idx]
      n_subjects = len(subject_ids)
      an_idx, ap_idx = load_group_information(subject_ids)
      print("\tTotal {} subjects (after excluding outliers) | AN: {} | AP: {}".format(
         n_subjects, len(an_idx), len(ap_idx),
      ))
      if (np.count_nonzero(group_assignments == 2) != len(an_idx)) or (np.count_nonzero(group_assignments == 1) != len(ap_idx)):
         raise ValueError("group assignments do not match with group information.")

   # Get fractional occupancies to be used as weights
   fo = modes.fractional_occupancies(btc) # dim: (n_subjects, n_states)

   # Fit GLM model to fractional occupancies
   fo_model, fo_design, fo_data = fit_glm_confound_regression(
      fo,
      subject_ids,
      modality,
      dimension_labels=["Subjects", "States/Modes"],
   )
   fo = fo_model.copes[0] # dim: (n_states,); FO after confound regression

   # ------------ [4] ----------- #
   #      Statistical Tests       #
   # ---------------------------- #
   print("Step 4 - Performing statistical tests ...")

   # Separate static and dynamic components in PSDs
   if model_type == "hmm":
      psd_static_mean = np.average(psd, axis=1, weights=fo, keepdims=True)
      psd_dynamic = psd - psd_static_mean
      # the mean across states/modes is subtracted from the PSDs subject-wise
   if model_type == "dynemo":
      psd_static_mean = psd[:, 1, :, :, :] # use regression intercepts
      psd_static_mean = np.expand_dims(psd_static_mean[:, 0, :, :], axis=1)
      # all modes have same regression intercepts
      psd_dynamic = psd[:, 0, :, :, :] # use regression coefficients only
      psd = np.sum(psd, axis=1) # sum coefficients and intercepts

   # Separate static and dynamic components in coherences
   coh_static_mean = np.average(coh, axis=1, weights=fo, keepdims=True)
   coh_dynamic = coh - coh_static_mean

   # Compute power maps
   power_map_dynamic = analysis.power.variance_from_spectra(f, psd_dynamic)
   # dim: (n_subjects, n_modes, n_parcels)
   power_map_static = analysis.power.variance_from_spectra(f, psd_static_mean)
   # dim: (n_subjects, n_parcels)

   # Compute connectivity maps
   conn_map_dynamic = analysis.connectivity.mean_coherence_from_spectra(f, coh_dynamic)
   # dim: (n_subjects, n_modes, n_parcels, n_parcels)
   conn_map_static = analysis.connectivity.mean_coherence_from_spectra(f, coh_static_mean)
   # dim: (n_subjects, n_parcels, n_parcels)

   # Define the number of tests for Bonferroni correction
   bonferroni_ntest = n_class + 1 # tests repeated over states/modes and static mean

   # Preallocate output data
   map_statistics = {
      "power_dynamic": {"tstats": [], "pvalues": []},
      "power_static": {"tstats": [], "pvalues": []},
      "connectivity_dynamic": {"tstats": [], "pvalues": []},
      "connectivity_static": {"tstats": [], "pvalues": []},
   }
   
   # Max-t permutation tests on the power maps
   print("[Power (mean-subtracted)] Running Max-t Permutation Test ...")
   
   for n in range(n_class):
      # Fit GLM model to dynamic power maps
      power_model, power_design, power_data = fit_glm(
         power_map_dynamic[:, n, :],
         subject_ids,
         group_assignments,
         modality=modality,
         dimension_labels=["Subjects", "Channels"],
      )
      tstats = np.squeeze(power_model.tstats[0])
      # Perform max-t permutation test
      pvalues = max_stat_perm_test(
         power_model,
         power_data,
         power_design,
         pooled_dims=1,
         contrast_idx=0,
         n_perm=10000,
         metric="tstats",
      )
      pvalues *= bonferroni_ntest
      # Plot dynamic power maps
      plot_thresholded_map(
         tstats,
         pvalues,
         map_type="power",
         mask_file=mask_file,
         parcellation_file=parcellation_file,
         filenames=[
            os.path.join(DATA_DIR, "maps", f"maxt_pow_map_dynamic_{n}_{lbl}.png")
            for lbl in ["unthr", "thr"]
         ]
      )
      # Store test statistics
      map_statistics["power_dynamic"]["tstats"].append(tstats)
      map_statistics["power_dynamic"]["pvalues"].append(pvalues)

   print("[Power (mean-only)] Running Max-t Permutation Test ...")

   # Fit GLM model to static mean power map
   power_model, power_design, power_data = fit_glm(
      power_map_static,
      subject_ids,
      group_assignments,
      modality=modality,
      dimension_labels=["Subjects", "Channels"],
   )
   tstats = np.squeeze(power_model.tstats[0])
   # Perform max-t permutation test
   pvalues = max_stat_perm_test(
      power_model,
      power_data,
      power_design,
      pooled_dims=1,
      contrast_idx=0,
      n_perm=10000,
      metric="tstats",
   )
   pvalues *= bonferroni_ntest
   # Plot static mean power map
   plot_thresholded_map(
      tstats,
      pvalues,
      map_type="power",
      mask_file=mask_file,
      parcellation_file=parcellation_file,
      filenames=[
         os.path.join(DATA_DIR, "maps", f"maxt_pow_map_static_{lbl}.png")
         for lbl in ["unthr", "thr"]
      ]
   )
   # Store test statistics
   map_statistics["power_static"]["tstats"].append(tstats)
   map_statistics["power_static"]["pvalues"].append(pvalues)

   # Max-t permutation tests on the connectivity maps
   print("[Connectivity (mean-subtracted)] Running Max-t Permutation Test ...")

   for n in range(n_class):
      # Vectorize an upper triangle of the connectivity matrix
      n_parcels = conn_map_dynamic.shape[-1]
      i, j = np.triu_indices(n_parcels, 1) # excluding diagonals
      conn_map_vec = conn_map_dynamic[:, n, :, :]
      conn_map_vec = conn_map_vec[:, i, j]
      # dim: (n_subjects, n_connections)
      # Fig GLM model to dynamic connectivity values
      conn_model, conn_design, conn_data = fit_glm(
         conn_map_vec,
         subject_ids,
         group_assignments,
         modality=modality,
         dimension_labels=["Subjects", "Connections"],
      )
      tstats = np.squeeze(conn_model.tstats[0])
      # Perform max-t permutation test
      pvalues = max_stat_perm_test(
         conn_model,
         conn_data,
         conn_design,
         pooled_dims=1,
         contrast_idx=0,
         n_perm=10000,
         metric="tstats",
      )
      pvalues *= bonferroni_ntest
      # Plot dynamic connectivity maps
      plot_thresholded_map(
         tstats,
         pvalues,
         map_type="connectivity",
         mask_file=mask_file,
         parcellation_file=parcellation_file,
         filenames=[
            os.path.join(DATA_DIR, "maps", f"maxt_conn_map_dynamic_{n}_{lbl}.png")
            for lbl in ["unthr", "thr"]
         ]
      )
      # Store t-statistics
      tstats_map = np.zeros((n_parcels, n_parcels))
      tstats_map[i, j] = tstats
      tstats_map += tstats_map.T
      map_statistics["connectivity_dynamic"]["tstats"].append(tstats_map)    
      # Store p-values
      pvalues_map = np.zeros((n_parcels, n_parcels))
      pvalues_map[i, j] = pvalues
      pvalues_map += pvalues_map.T
      map_statistics["connectivity_dynamic"]["pvalues"].append(pvalues_map)

   print("[Connectivity (mean-only)] Running Max-t Permutation Test ...")

   # Vectorize an upper triangle of the connectivity matrix
   n_parcels = conn_map_static.shape[-1]
   i, j = np.triu_indices(n_parcels, 1) # excluding diagonals
   conn_map_vec = conn_map_static
   conn_map_vec = conn_map_vec[:, i, j]
   # dim: (n_subjects, n_connections)
   # Fit GLM model to static mean connectivity map
   conn_model, conn_design, conn_data = fit_glm(
      conn_map_vec,
      subject_ids,
      group_assignments,
      modality=modality,
      dimension_labels=["Subjects", "Connections"],
   )
   tstats = np.squeeze(conn_model.tstats[0])
   # Perform max-t permutation test
   pvalues = max_stat_perm_test(
      conn_model,
      conn_data,
      conn_design,
      pooled_dims=1,
      contrast_idx=0,
      n_perm=10000,
      metric="tstats",
   )
   pvalues *= bonferroni_ntest
   # Plot static mean connectivity map
   plot_thresholded_map(
      tstats,
      pvalues,
      map_type="connectivity",
      mask_file=mask_file,
      parcellation_file=parcellation_file,
      filenames=[
         os.path.join(DATA_DIR, "maps", f"maxt_conn_map_static_{lbl}.png")
         for lbl in ["unthr", "thr"]
      ]
   )
   # Store t-statistics
   tstats_map = np.zeros((n_parcels, n_parcels))
   tstats_map[i, j] = tstats
   tstats_map += tstats_map.T
   map_statistics["connectivity_static"]["tstats"].append(tstats_map)
   # Store p-values
   pvalues_map = np.zeros((n_parcels, n_parcels))
   pvalues_map[i, j] = pvalues
   pvalues_map += pvalues_map.T
   map_statistics["connectivity_static"]["pvalues"].append(pvalues_map)

   # Save statistical test results
   with open(os.path.join(DATA_DIR, f"model/results/map_statistics.pkl"), "wb") as output_path:
      pickle.dump(map_statistics, output_path)
   output_path.close()

   print("Analysis complete.")
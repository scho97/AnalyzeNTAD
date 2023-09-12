"""Preprocess the NTAD data with a manual ICA rejection

"""

# Install dependencies
import os
import osl
import logging
from sys import argv
from osl import utils

# Define modality
if len(argv) != 2:
    raise ValueError("need to pass one argument: modality (e.g., python script.py eeg)")
modality = argv[1]
print(f"[INFO] Modality: {modality.upper()}")

# Start logger
logger = logging.getLogger("osl")

# Set directories
BASE_DIR = "/ohba/pi/mwoolrich/datasets/NTAD"
OXF_DIR = os.path.join(BASE_DIR, "oxford")
CAM_DIR = os.path.join(BASE_DIR, "cambridge")
PREPROC_DIR = f"/ohba/pi/mwoolrich/scho/NTAD/preproc/{modality}"

# Select subjects
subject_ids = ["P1009", "P1010", "P1031"]
print(f"Number of subjects that need manual ICA rejection: {len(subject_ids)}")

# Define custom functions
def detect_bad_channels_manual(dataset, userargs):
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_bad_channels_manual"))
    logger.info("userargs: {0}".format(str(userargs)))

    # Define bad channels for specific subjects
    bad_ch_info = {"C1019": ["EEG050"],
                   "P1010": ["EEG003", "EEG033"],
                   "P1030": ["EEG037", "EEG039"],
                   "P1054": ["EEG033", "EEG047"],
                   "P1060": ["EEG048"]}

    # Concatenate manually found bad channels to existing list
    s = "Manual bad channel detection - {0} channels appended as bad channels."
    bad_subject_ids = list(bad_ch_info.keys())
    if "first_name" in dataset["raw"].info["subject_info"]:
        subject_id = dataset["raw"].info["subject_info"]["first_name"].split("_")[-1].upper()
    else:
        raise ValueError("subject ID information not available.")
    if subject_id in bad_subject_ids:
        for ch_name in bad_ch_info[subject_id]:
            if ch_name not in dataset["raw"].info["bads"]:
                dataset["raw"].info["bads"].extend([ch_name])
        logger.info(s.format(bad_ch_info[subject_id]))

    return dataset

# Get file paths
inputs = []
for id in subject_ids:
    if int(id[1]) in [1, 3]: # site: Cambridge
        inputs.append(os.path.join(CAM_DIR, f"{id}/{id}_Baseline_MEG/6/{id}_resting_close_bl_raw_tsss.fif"))
    if int(id[1]) == 2: # site: Oxford
        inputs.append(os.path.join(OXF_DIR, f"{id}/{id}_Baseline_MEG/6/{id}_resting_close_bl_raw_tsss.fif"))
# NOTE: Only eyes-closed resting-state data are being used here.

# Print available subjects
print(f"Number of available subjects: {len(inputs)}")

# Configure pipeline
if modality == "meg":
    config = """
        preproc:
            - pick_types: {meg: true, eeg: false, eog: true, ecg: true, stim: true, ref_meg: false}
            - crop: {tmin: 20}
            - filter: {l_freq: 0.5, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
            - notch_filter: {freqs: 50 100}
            - resample: {sfreq: 250, n_jobs: 1}
            - bad_segments: {segment_len: 600, picks: 'mag'}
            - bad_segments: {segment_len: 600, picks: 'grad'}
            - bad_segments: {segment_len: 600, picks: 'mag', mode: diff}
            - bad_segments: {segment_len: 600, picks: 'grad', mode: diff}
            - bad_channels: {picks: 'mag', significance_level: 0.1}
            - bad_channels: {picks: 'grad', significance_level: 0.1}
            - ica_raw: {picks: 'meg', n_components: 64}
            - ica_manualreject: {apply: True}
            - interpolate_bads: {}
    """
elif modality == "eeg":
    config = """
        preproc:
            - pick_types: {meg: false, eeg: true, eog: true, ecg: true, stim: true, ref_meg: false}
            - drop_channels: {ch_names: ['EEG071', 'EEG072', 'EEG073', 'EEG074'], on_missing: 'ignore'}
            - crop: {tmin: 20}
            - filter: {l_freq: 0.5, h_freq: 125, method: 'iir', iir_params: {order: 5, ftype: butter}}
            - notch_filter: {freqs: 50 100}
            - resample: {sfreq: 250, n_jobs: 1}
            - bad_segments: {segment_len: 200, picks: 'eeg', metric: 'kurtosis'}
            - bad_segments: {segment_len: 200, picks: 'eeg', metric: 'kurtosis', mode: diff}
            - bad_segments: {segment_len: 400, picks: 'eeg', metric: 'kurtosis'}
            - bad_segments: {segment_len: 400, picks: 'eeg', metric: 'kurtosis', mode: diff}
            - bad_segments: {segment_len: 200, picks: 'eeg'}
            - bad_segments: {segment_len: 200, picks: 'eeg', mode: diff}
            - bad_segments: {segment_len: 400, picks: 'eeg', significance_level: 0.01}
            - bad_segments: {segment_len: 400, picks: 'eeg', significance_level: 0.01, mode: diff}
            - bad_segments: {segment_len: 800, picks: 'eeg', significance_level: 0.01}
            - bad_segments: {segment_len: 800, picks: 'eeg', significance_level: 0.01, mode: diff}
            - bad_channels: {picks: 'eeg', significance_level: 0.1}
            - detect_bad_channels_manual: {}
            - ica_raw: {picks: 'eeg', n_components: 40}
            - ica_manualreject: {apply: True}
            - interpolate_bads: {}
            - set_eeg_reference: {projection: true}
    """
# NOTE: If we want to have same data length of EEG and MEG, then we should preprocess M/EEG all at once.

# PREPROCESS DATA
if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Initiate preprocessing
    if modality == "meg":
        osl.preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=PREPROC_DIR,
            overwrite=True,
            dask_client=False,
        )
    else:
        osl.preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=PREPROC_DIR,
            overwrite=True,
            extra_funcs=[detect_bad_channels_manual],
            dask_client=False,
        )

print("Preprocessing complete.")
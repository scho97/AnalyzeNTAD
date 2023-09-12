"""Apply dipole sign flipping

"""

# Install dependencies
import os
import glob
import logging
from sys import argv
from osl import utils, source_recon
from dask.distributed import Client

# Start logger
logger = logging.getLogger("osl")

# Define modality
if len(argv) != 2:
    raise ValueError("need to pass one argument: modality (e.g., python script.py eeg)")
modality = argv[1]
print(f"[INFO] Modality: {modality.upper()}")

# Set directories
BASE_DIR = "/ohba/pi/mwoolrich/scho/NTAD"
SRC_DIR = os.path.join(BASE_DIR, f"src/{modality}")

# Get subject ID numbers
src_files = sorted(glob.glob(os.path.join(SRC_DIR, "*/parc/parc-raw.fif")))
subjects = [file.split("/")[-3] for file in src_files]
print(f"Number of subjects available: {len(subjects)}")

# Find subject to use as a template
template = source_recon.find_template_subject(
    SRC_DIR, subjects, n_embeddings=15, standardize=True
)

# Configure pipeline
config = f"""
    source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: true
            n_init: 3
            n_iter: 5000
            max_flips: 20
"""

# SIGN FLIP DATA
if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Set up FSL
    source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")

    # Set up parallel processing
    client = Client(n_workers=4, threads_per_worker=1)

    # Initiate sign flipping
    source_recon.run_src_batch(
        config,
        SRC_DIR,
        subjects,
        dask_client=True,
    )

print("Sign flipping complete.")
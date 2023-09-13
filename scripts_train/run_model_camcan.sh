#!/bin/bash

#SBATCH -D /well/woolrich/users/olt015/AnalyzeNTAD/results/dynamic/camcan/hmm
#SBATCH -p gpu_long
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 200G
#SBATCH --constraint "rtx8000|a100"

# Validate CLI input arguments
if [ ${1} != "lemon" ] && [ ${1} != "camcan" ]; then
    echo "Invalid argument. Usage: $0 [lemon|camcan]"
    exit 1
fi
if [ ${2} != "hmm" ] && [ ${2} != "dynemo" ]; then
    echo "Invalid argument. Usage: $0 $1 [hmm|dynemo]"
    exit 1
fi
if ! [[ ${3} =~ ^[0-9]+$ ]]; then
    echo "Invalid argument. Usage: $0 $1 $2 [int]"
    exit 1
fi
echo "Input arguments submitted. Data Name: ${1}, MODEL TYPE: ${2}, RUN #: ${3}"

# Set up your environment
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
module load cuDNN

# Run scripts
conda activate /well/woolrich/users/olt015/conda/skylake/envs/osld
python /well/woolrich/users/olt015/AnalyzeNTAD/scripts_train/pretrain_${1}_${2}.py ${3}_${2}
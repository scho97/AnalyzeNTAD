# AnalyzeNTAD
Analysis of New Therapeutics in Alzheimer's Disease Longitudinal Cohort (NTAD) M/EEG Data

In Chapter 3 of my master's thesis: [_Inferring brain network dynamics of MEG and EEG in healthy aging and Alzheimer's disease_](https://ora.ox.ac.uk/objects/uuid:27faa894-b350-4da4-a7e4-7611dbd86791)

💡 Please email SungJun Cho at sungjun.cho@ndcn.ox.ac.uk with any questions or concerns.

---

## ⚡️ Getting Started

This repository contains all the scripts necessary to reproduce the analysis and figures shown in Chapter 3 of my thesis. It is divided into five main directories:

1. `scripts_data`: Contains the scripts for inspecting subject demographics and data characteristics.
2. `scripts_preparation`: Contains the scripts for data preprocessing and source reconstruction.
3. `scripts_static`: Contains the scripts for analyzing static power and functional connectivity of resting-state electrophysiological data.
4. `scripts_dynamic`: Contains the scripts for analyzing power and functional connectivity of dynamic resting-state networks.
5. `scripts_train`: Contains the scripts for (pre-)training HMM and DyNeMo models to infer resting-state network dynamics.

### Installation Guide
To start, you first need to install the `osl-dynamics` package and set up its environment. Its installation guide can be found [here](https://github.com/OHBA-analysis/osl-dynamics).

The `seaborn` and `openpyxl` packages need to be additionally installed for visualization and compatibility with excel files. Next, download this repository to your designated folder location as below. Once these steps are complete, you're ready to go!

```
conda activate osld
pip install seaborn
pip install openpyxl
git clone https://github.com/scho97/AnalyzeNTAD.git
cd AnalyzeNTAD
```

## 📄 Detailed Descriptions

Each directory named `scripts_*` contains a `utils` folder, where the functions required to execute the scripts are kept. Within this directory, scripts are numerically prefixed, indicating the sequence in which they should run. A few exceptions are:

- Excel files that store channel labels for different EEG montages. (Located in `scripts_data`)
- A script for manually detecting bad channels for data preprocessing. (Located in `scripts_preparation`)
- Excel files that store (1) state/mode orders of inferred RSNs and (2) subject outlier indices for DyNeMo trained on the EEG data. (Located in `scripts_dynamic`)
- Scripts for pre-training HMM and DyNeMo models on the LEMON and Cam-CAN data. (Located in `scripts_train`)

Note that the information regarding participant demographics are not shared due to data privacy and ethicial issues. For more details, please refer to the thesis. All code within this repository was executed on the Oxford Centre for Human Brain Activity (OHBA) `hbaws` servers.

## 🎯 Requirements
The analyses and visualizations in this paper had following dependencies:

```
python==3.8.16
osl-dynamics==1.2.6
seaborn==0.12.2
openpyxl==3.1.2
tensorflow==2.9.1
tensorflow-probability==0.17.0
```

NOTE: The `tensorflow` pakage was downgraded to ensure compatibility with `tfrecord` functionality.

## 🪪 License
Copyright (c) 2023 [SungJun Cho](https://github.com/scho97) and [OHBA Analysis Group](https://github.com/OHBA-analysis). `AnalyzeNTAD` is a free and open-source software licensed under the [MIT License](https://github.com/scho97/AnalyzeNTAD/blob/main/LICENSE).

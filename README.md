# CodeTrio_Mujhackx
Brain Signal Analysis for Attention Identification

Project Overview

CodeTrio's Brain Signal Analysis project aims to develop a machine learning model that processes EEG data to distinguish between different states of attention. The system analyzes EEG signals to detect attention levels by extracting frequency bands and time-frequency features, classifying brain states into focused or unfocused using machine learning algorithms like Random Forest.

Table of Contents

Getting Started
Technology Used
Methodology and Process
Input Data
Feasibility and Viability
Impacts and Benefits
Contributors



To get started with the project, follow these steps:

Clone the repository: git clone https://github.com/CodeTrio/Brain-Signal-Analysis.git
Install the required libraries: pip install -r requirements.txt
Run the data preprocessing script: python preprocess_data.py
Train the model: python train_model.py
Evaluate the model: python evaluate_model.py
Technology Used

Python
NumPy
SciPy
scikit-learn
RandomForest
EEG hardware for signal collection
Short-Time Fourier Transform (STFT) for feature extraction
Methodology and Process

The process includes:

EEG data acquisition
Preprocessing with filtering techniques
Feature extraction using STFT
Calculation of alpha-beta ratio and theta-beta ratio
Classification of brain states using Random Forest
Input Data

The input data is a .mat file containing EEG plots.

Feasibility and Viability

The solution is technically feasible with current EEG hardware and data processing tools. However, challenges like noisy data, variability in brainwave patterns, and real-time processing may impact viability and require further research and testing.

Impacts and Benefits

The system could enhance focus tracking in educational settings, improve cognitive load monitoring in medical diagnostics, and boost productivity by providing real-time feedback on attention levels, helping users optimize their performance and concentration.

Contributors

Tanmay Talreja
Ananya Sinha
Subhojeet Roy

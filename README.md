# MSGI: Time-to-Aging-Failure Prediction of Software Systems

Official PyTorch implementation of:

**MSGI: Feature Selection Enhanced Multiscale Graph Informer for Predicting the Time to Aging Failure of Software Systems**

---

## Overview

MSGI is a deep learning framework developed for **Time-to-Aging-Failure (TTAF) prediction** in long-running software systems. The framework aims to accurately estimate the remaining operating time before software aging failures occur by modeling the complex interactions among heterogeneous system monitoring metrics.

Software aging prediction is challenging due to three major issues:

- **Feature redundancy:** Large-scale monitoring metrics often contain irrelevant or redundant information.
- **Temporal uncertainty:** Aging indicators exhibit nonlinear fluctuations and multi-scale temporal variations during continuous operation.
- **Complex metric dependencies:** Different system-level indicators interact with each other and jointly influence aging evolution.

To address these challenges, MSGI integrates four complementary components:

- **Boruta-SHAP Feature Selection:**  
  Identifies aging-related system indicators by combining the robustness of Boruta with the interpretability of SHAP values, reducing redundant information and improving feature quality.

- **Multiscale Decomposable Mixing (MDM):**  
  Decomposes complex aging indicator sequences into multiple temporal scales, enabling the model to capture both short-term fluctuations and long-term degradation trends.

- **Graph Convolutional Kernel Machine (GCKM):**  
  Dynamically constructs feature-level graph structures based on metric relationships, capturing latent dependencies among heterogeneous monitoring variables.

- **Informer Backbone:**  
  Utilizes ProbSparse attention mechanisms to model long-range temporal dependencies and efficiently learn the evolutionary patterns of software aging processes.

By jointly exploiting feature selection, multiscale temporal representation, dynamic graph modeling, and long-range dependency learning, MSGI provides a robust solution for TTAF prediction under different software operating environments.

---

## Environment

The experiments were conducted under the following environment:

- Python 3.9
- PyTorch 2.3.1
- CUDA 12.1 (recommended for GPU acceleration)

### Installation

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

### Main Dependencies

```
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1

timm==1.0.15
einops

numpy==1.26.4
pandas==2.2.2

scipy==1.13.1
scikit-learn==1.5.1
networkx==3.3

BorutaShap==1.0.17
shap==0.46.0

joblib==1.4.2
numba==0.60.0
llvmlite==0.43.0

matplotlib==3.9.1
seaborn==0.13.2

tqdm==4.67.1
PyYAML==6.0.2
```

---

# Dataset

MSGI is evaluated on two representative software platforms using real-world **run-to-failure datasets** collected through accelerated life testing (ALT).

## Android Platform

The Android dataset contains system-level monitoring metrics collected during continuous operation until aging failure.

The monitored indicators include:

- CPU utilization
- Load averages (Load1, Load5, Load15)
- Temperature
- PSS of system_server
- Free memory
- Used memory
- Native Heap
- Dalvik Heap
- Swap usage
- Page fault statistics (pgfault, pgfree)

## OpenStack Platform

The OpenStack dataset represents a cloud computing environment with continuous workload execution.

The monitored indicators include:

- CPU utilization
- Load averages (Load1, Load5, Load15)
- Free memory
- Active and inactive memory
- Swap usage
- Disk read requests per second (r/s)
- Disk write requests per second (w/s)

---

## Dataset Organization

Please organize the datasets according to the following structure:

```
dataset/
│
├── Android/
│   ├── dataset_1.csv
│   ├── dataset_2.csv
│   ├── dataset_3.csv
│   └── dataset_4.csv
│
└── OpenStack/
    ├── dataset_5.csv
    ├── dataset_6.csv
    ├── dataset_7.csv
    └── dataset_8.csv
```

Each CSV file represents a complete run-to-failure sequence containing multiple system monitoring variables and the corresponding TTAF labels.

---

# Quick Start

To reproduce the experimental results reported in the paper, please follow the execution order below.

## Step 1: Feature Selection

The raw monitoring data contain redundant and irrelevant variables. First, execute the Boruta-SHAP feature selection module to identify informative aging-related features.

Run:

```bash
python BorutaShap.py
```

The selected feature subsets will be generated for subsequent model training.

---

## Step 2: Model Training and Evaluation

After feature selection, train the MSGI model and evaluate prediction performance using:

```bash
python run.py
```

The program automatically performs:

- Data preprocessing and normalization
- Sliding-window sequence generation
- MSGI model training
- Prediction on test datasets
- Performance evaluation

---

# Evaluation Metrics

MSGI performance is evaluated using three commonly used regression metrics:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual TTAF values.

### Root Mean Square Error (RMSE)

Measures prediction errors by assigning larger penalties to larger deviations.

### Coefficient of Determination (R²)

Evaluates the goodness-of-fit between predictions and ground truth values.

---

# Project Structure

```
MSGI/
│
├── dataset/
│   ├── Android/
│   └── OpenStack/
│
├── models/
│   ├── __init__.py
│   ├── attn.py
│   ├── decoder.py
│   ├── embed.py
│   ├── encoder.py
│   ├── fourier.py
│   ├── MDM.py
│   ├── GCKM.py
│   └── model.py
│
├── utils/
│   ├── masking.py
│   └── metrics.py
│
├── exp/
│
├── BorutaShap.py
├── run.py
├── requirements.txt
└── README.md
```

---

# Reproducibility

All experiments presented in the paper can be reproduced using the provided source code, datasets, and configuration settings.

The implementation follows the complete MSGI pipeline:

1. **Feature selection using Boruta-SHAP**
2. **Multiscale temporal representation learning using MDM**
3. **Dynamic feature graph construction using GCKM**
4. **Sequence-to-label transformation using sliding windows**
5. **End-to-end TTAF prediction and evaluation**

The experimental environment, dependency versions, and implementation details are provided in this repository to facilitate reproducible research.

---

# Citation

If you find this repository useful for your research, please cite:

```bibtex
@article{MSGI,
  title={MSGI: Feature Selection Enhanced Multiscale Graph Informer for Predicting the Time to Aging Failure of Software Systems},
  author={},
  journal={},
  year={}
}
```

---

# Contact

For questions regarding the methodology, datasets, or implementation details, please contact the authors through the corresponding publication.

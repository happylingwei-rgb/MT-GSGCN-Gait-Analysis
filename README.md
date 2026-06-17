# MT-GSGCN & MT-GSGCN-CBAM: Multi-Task Spatial-Temporal Graph Convolutional Networks for Gait Analysis

This repository contains the official implementation of **MT-GSGCN** (Multi-Task Graph Spatial-Temporal Graph Convolutional Network) and its CBAM-enhanced variant **MT-GSGCN-CBAM**, designed for gait classification and analysis.

This codebase is structured to support reproducible evaluations on multiple datasets, including MMGS, GIST, and GIST with different temporal convolutional kernel sizes.

---

## Repository Structure

The repository is organized as follows:

```
├── MMGS/
│   └── code/
│       ├── MT-GSGCN.ipynb            # Training/evaluation of MT-GSGCN on MMGS
│       ├── MT-GSGCN-CBAM.ipynb       # Training/evaluation of MT-GSGCN-CBAM on MMGS
│       ├── LossandVcc.ipynb          # Notebook for analyzing loss and validation metrics
│       ├── net/                      # Network architectures and model definitions
│       └── mmgs_method*_trainlog.txt # Training logs for reproducibility
│
├── GIST/
│   └── code/
│       ├── MT-GSGCN.ipynb            # Training/evaluation of MT-GSGCN on GIST
│       ├── MT-GSGCN-CBAM.ipynb       # Training/evaluation of MT-GSGCN-CBAM on GIST
│       ├── net/                      # Model definitions
│       ├── logs/                     # Training logs
│       └── 畫圖/                     # Visualization scripts for GIST experiments
│
├── GIST使用不同捲積核的結果/           # GIST experiments with different kernel configurations
│   ├── GIST - 使用1-3-5捲積核/         # Configuration using temporal kernel sizes 1-3-5
│   │   ├── MT-GSGCN.ipynb
│   │   ├── MT-GSGCN-CBAM.ipynb
│   │   ├── net/
│   │   ├── logs/
│   │   └── 畫圖/                     # Scripts for plotting learning curves
│   ├── GIST - 使用5-7-9捲積核/         # Configuration using temporal kernel sizes 5-7-9
│   │   ├── MT-GSGCN.ipynb
│   │   ├── MT-GSGCN-CBAM.ipynb
│   │   ├── net/
│   │   └── logs/
│   └── 不同卷積核比較.pptx              # Presentation summarizing kernel size comparison
│
└── .gitignore                        # Standard exclusions for datasets, weights, and cache
```

---

## Environment Setup

To run the notebooks, please set up a Python environment with the following dependencies. We recommend using [Conda](https://docs.conda.io/en/latest/) for environment management.

### Prerequisites
- Python 3.8+
- PyTorch (matching your system's CUDA version)
- Jupyter Notebook or JupyterLab

### Installation
You can install the required packages using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib torch torchvision torchaudio
```

Or create a conda environment from scratch:

```bash
conda create -n gait-gcn python=3.9
conda activate gait-gcn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas scikit-learn matplotlib jupyter
```

---

## Dataset Setup

> [!NOTE]  
> The raw datasets are excluded from this Git repository due to size constraints. You must download the datasets and place them under the `datasets/` directory at the root level using the following structure:

```
gait-analysis-root/
├── datasets/
│   ├── MMGS datasets/
│   │   └── datasets/
│   │       └── database_joints/
│   │           └── data_with_cycles_27_all.mat    # Matlab data file
│   │
│   ├── GIST datasets/
│   │   └── datasets/
│   │       └── processed_data/
│   │           └── patho50_data_noNorm.npz        # Preprocessed numpy array
│   │
│   └── GIST_Temporal_Kernel_Experiments datasets/
│       └── datasets/
│           ├── pathological_gait_datasets-master/ # Raw gait data
│           ├── processed_data/                   # Processed train/test splits
│           └── demo_samples/                     # Demo samples for verification
```

---

## How to Run

1. **Start Jupyter Notebook**:
   Launch Jupyter Notebook from the root of the project:
   ```bash
   jupyter notebook
   ```

2. **Run Training**:
   Open any of the training notebooks (e.g., `MMGS/code/MT-GSGCN-CBAM.ipynb` or `GIST/code/MT-GSGCN.ipynb`) and run the cells sequentially to preprocess the data, build the graph model, and execute the training loop.

3. **Plot and Analyze**:
   The training logs are saved automatically. You can run the notebooks inside the `畫圖/` subdirectories to reproduce the accuracy and loss curves shown in our manuscript.

---

## Citation

If you find this repository helpful for your research, please consider citing our paper:

```bibtex
@article{chen2026multiscale,
  title={A Multi-Scale Temporal Graph Convolutional Network Model for Skeleton-Based Abnormal Gait Recognition},
  author={Chen, Young-Long and Lien, Shih-Sheng and Lin, Ling-Wei},
  journal={To be submitted},
  year={2026}
}
```

Or in text format:
> Young-Long Chen, Shih-Sheng Lien, and Ling-Wei Lin. "A Multi-Scale Temporal Graph Convolutional Network Model for Skeleton-Based Abnormal Gait Recognition." (To be submitted).

# DUET-Net

Official implementation of the paper **DUET-Net: Dual Uncertainty-Driven Semi-Supervised Learning for Dense Maize Tassel Counting**.

All source code will be made public after the paper is accepted.

This project implements a semi-supervised counting model, incorporating a Parallel Perception Attention (PPA) module for feature extraction and an uncertainty estimation strategy for pseudo-label generation.

***

# Running Environment

The model is implemented using PyTorch. The recommended environment configuration is:

- **OS**: Windows 10
- **Python**: 3.8
- **PyTorch**: 1.10.1
- **CUDA**: 11.3 (Recommended for PyTorch 1.10.1)
- **GPU**: NVIDIA RTX 4090 (24GB)

# Run

To train the model:

```bash
python train.py
```

# Data Processing

- **Annotation Conversion**: Use `mat_to_npy_converter.py` to convert `.mat` annotation files to `.npy` format if needed.
- **Dataset Loading**: The dataset loading logic is implemented in `datasets/maize_semi.py`.

# Requirements

The project requires the following dependencies:

- h5py==3.1.0
- matplotlib==3.7.5
- numpy==1.24.4
- opencv-python==4.12.0.88
- pandas==2.0.3
- Pillow==9.5.0
- PyYAML==6.0.3
- scikit-learn==1.3.2
- scipy==1.10.1
- tensorboardX==2.6.2.2
- torch==1.10.1
- torchvision==0.11.2
- tqdm==4.67.1

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

# Model Architecture

- **PPA Module**: The Parallel Perception Attention (PPA) module implementation is located in `models/model_ppa.py`.
- **Loss Functions**: The loss functions used for training are defined in `losses/losses.py`.

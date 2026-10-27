# PET/CT Multimodal Multi-task Learning for Lung Cancer Invasiveness Classification

## Introduction

Lung cancer is one of the malignancies with the highest incidence and mortality rates
worldwide. Its degree of invasiveness (e.g., visceral pleural invasion, lymph node
metastasis, vascular invasion) directly affects clinical staging, surgical planning,
and prognosis assessment. Traditionally, invasiveness evaluation relies on
postoperative pathological examination and cannot support preoperative clinical
decision-making.

This project proposes an intelligent lung cancer invasiveness assessment method based
on **multimodal PET/CT medical imaging**. PET images reflect the metabolic activity of
tumors, while CT images provide detailed anatomical structural information — the two
modalities are strongly complementary. Built on a 3D deep learning backbone, the
framework fuses imaging features from both modalities and predicts multiple
invasiveness-related indicators simultaneously through a **multi-task learning**
scheme, exploiting inter-task correlations to improve overall predictive performance.

## Technical Pipeline

1. **Data preprocessing**: DICOM series conversion, image resampling, PET/CT
   registration, SUV conversion, tumor ROI extraction and multi-stage cropping, to
   ensure standardized and consistent input data
2. **Data loading and augmentation**: window level/width adjustment, aligned
   multi-modal slice loading, sampling strategies for class imbalance, and 3D data
   augmentation
3. **Multimodal multi-task modeling**: a 3D ResNet18 backbone extracts deep imaging
   features from PET and CT separately; after feature fusion, multi-task
   classification heads jointly optimize multiple invasiveness prediction objectives
4. **Training and evaluation**: a joint multi-task training framework, comprehensively
   evaluated with sensitivity, specificity, AUC, ROC curves and other metrics

## Requirements

- PyTorch / torchvision
- SimpleITK (medical image I/O and processing)
- scikit-learn, pandas, numpy
- matplotlib

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Data Preparation

The code indexes data through Excel files (`.xlsx`), where each row corresponds to one
case. The training/testing tables (e.g., `./Dataset/train/train.xlsx`,
`./Dataset/test/test.xlsx`) should contain the following key columns:

| Column | Description |
|---|---|
| `name` | Case identifier |
| `CT_Path` | Path to the CT tumor-region image (NIfTI format, `.nii/.nii.gz`) of the case |
| `PET_Path` | Path to the PET tumor-region image (SUV values) of the case |
| `T_Label` / `p_label` | Invasiveness label (0/1) |

## File Overview and Usage

### `Data_loader.py` — Data loading pipeline

The core is the `ReadDataFromCSV` dataset class:

```python
dataset = ReadDataFromCSV(csv_path, mode='CP', transform=transform, Data_Aug=0, fusion='late')
```

- `mode`: `'CT'` / `'PET'` (single modality), `'CP'` (CT+PET dual modality)
- `fusion`: dual-modality fusion strategy — `'early'` (channel concatenation fed into
  a single branch) or `'late'` (two branches extract features separately before fusion)
- Built-in preprocessing: CT window level/width adjustment, PET normalization,
  standardization, etc.

### `model/ResNet18.py` — Network definitions

Contains the 3D ResNet18 backbone and its multimodal / multi-task variants:

- `ResNet18`: single-modality 3D classification network
- `CP_ResNet18`: CT+PET dual-modality fusion network
- `*_mutli_attention` family: variants that introduce multi-head self-attention at the
  fusion stage
- `*_MultiLabel` family: multi-task (multi-label) classification-head variants

### `ResNet18_train.py` — Training entry point

1. Point `train_csv` / `test_csv` in `__main__` to your own data tables
2. Choose a model (default `CP_ResNet18_mutli_attention`); other variants can be
   switched in
3. Hyperparameters can be adjusted at the top of the script: defaults are 400 epochs,
   batch size 16, SGD (lr=1e-4), `BCEWithLogitsLoss`, and `WeightedRandomSampler` to
   handle class imbalance

Run:

```bash
python ResNet18_train.py
```

Training logs are written to `./result/log/`; the best model (by validation AUC) is
saved to `./result/checkpoint/`.

### `model_test.py` — Evaluation entry point

1. Point `checkpoint = torch.load(...)` to the trained model weights
2. Point `train_csv` / `test_csv` to the data tables

Run:

```bash
python model_test.py
```

Outputs accuracy, sensitivity, specificity, F1, AUC and other metrics to
`./result/log/result.csv`, and saves the ROC curve as `./result/log/ROC.jpg`.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare ./Dataset/train/train.xlsx and ./Dataset/test/test.xlsx in the format above

# 3. Train
python ResNet18_train.py

# 4. Evaluate
python model_test.py
```

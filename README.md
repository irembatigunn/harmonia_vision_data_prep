# Harmonia Vision — Data Preprocessing Pipeline

**CBIS-DDSM Mammogram Data Cleaning for Breast Cancer Mass Segmentation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository contains the **10-step data cleaning pipeline** for the [Harmonia Vision](https://github.com/irembatigunn) project — a privacy-preserving Federated Learning system for breast cancer mass segmentation using U-Net.

The pipeline transforms **CBIS-DDSM** (Curated Breast Imaging Subset of Digital Database for Screening Mammography) mammogram images into clean, model-ready `.npy` arrays.

**Data source:** [CBIS-DDSM on TCIA](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) — we use the [Kaggle version (awsaf49)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) which provides JPEG-converted mammograms and ROI masks.

### Pipeline Summary

```
1696 raw rows (mass_train + mass_test CSV)
    → 10-step cleaning pipeline
    → 1582 processed 256×256 mammograms + binary masks
    → 886 patients, patient-based 70/15/15 train/val/test split
```

---

## Project Structure

```
harmonia_vision_data_prep/
│
├── scripts/                                 # Pipeline code (all steps)
│   ├── config.py                            # Shared paths & parameters
│   ├── step1_load_and_merge.py              # Load & merge mass CSVs
│   ├── step2_join_metadata.py               # Join DICOM metadata
│   ├── step3_consolidate_labels.py          # Consolidate pathology labels
│   ├── step4_file_integrity.py              # File existence & corruption check
│   ├── step5_mask_validation.py             # Binary mask validation
│   ├── step6_combine_masks.py               # Multi-lesion mask combination
│   ├── step7_breast_extraction.py           # Breast region bounding box
│   ├── step8_crop_resize.py                 # Crop, pad & resize to 256×256
│   ├── step9_split_data.py                  # Patient-level stratified split
│   ├── step10_save_npy.py                   # Export final .npy arrays
│   ├── diag_step7.py                        # Visual diagnostics (Step 7)
│   └── diag_step8.py                        # Visual diagnostics (Step 8)
│
├── data_cleaning_steps.md                   # Detailed pipeline documentation
│
├── CBIS DDSM Breast Cancer Dataset/         # Raw data (git-ignored)
│   ├── csv/                                 # 6 metadata CSVs
│   └── jpeg/                                # 6775 UID folders with JPEGs
│
├── outputs/                                 # Pipeline outputs (git-ignored)
│   ├── step1_mass_merged.csv ... step9_splits.csv
│   ├── combined_masks/                      # 1592 combined mask PNGs
│   ├── processed/
│   │   ├── images/                          # 1582 preprocessed 256×256 PNGs
│   │   └── masks/                           # 1582 preprocessed 256×256 masks
│   └── final_npy/                           # ⭐ Model-ready arrays
│       ├── X_{split}.npy                    # (N, 256, 256, 1) uint8 images
│       ├── masks_{split}.npy                # (N, 256, 256, 1) uint8 masks
│       ├── y_{split}.npy                    # (N,) uint8 labels
│       ├── metadata_{split}.csv             # Per-sample metadata
│       ├── pipeline_summary.json            # Full pipeline statistics
│       └── dataset_info.txt                 # Human-readable summary
│
├── logs/                                    # Diagnostic logs (git-ignored)
│   ├── step7_breast_extraction.csv          # Alignment errors (10 dropped)
│   ├── step7_diag/                          # Bounding box visualizations
│   └── step8_diag/                          # 256×256 output samples
│
├── .gitignore
└── README.md
```

---

## Pipeline Steps

| # | Script | Description | Input → Output |
|---|--------|-------------|----------------|
| 1 | `step1_load_and_merge.py` | Loads `mass_train` + `mass_test` CSVs, normalizes column names to `snake_case`, merges into single DataFrame | Raw CSVs → `step1_mass_merged.csv` (1696 rows) |
| 2 | `step2_join_metadata.py` | Joins with `dicom_info.csv` to resolve JPEG paths. Filters out cropped images. Maps each mass row to absolute `full_image` and `roi_mask` paths | `step1` → `step2_mass_joined.csv` (1696 rows, 100% join) |
| 3 | `step3_consolidate_labels.py` | Merges `BENIGN_WITHOUT_CALLBACK` → `BENIGN`. Adds binary `label` column (0/1). Preserves original labels in `pathology_original` | `step2` → `step3_labels_consolidated.csv` |
| 4 | `step4_file_integrity.py` | Validates every image and mask file exists and is not corrupt (PIL open test) | `step3` → `step4_files_verified.csv` (0 issues) |
| 5 | `step5_mask_validation.py` | Verifies masks are truly binary via bimodal pixel analysis (≥98% in [0..30] or [225..255]). Detects empty masks | `step4` → `step5_masks_validated.csv` (0 issues) |
| 6 | `step6_combine_masks.py` | Groups by `(patient, breast, view)`. Combines multi-lesion masks via logical OR. Applies MALIGNANT precedence for mixed-class groups | `step5` → `step6_masks_combined.csv` (1696 → 1592 rows) |
| 7 | `step7_breast_extraction.py` | Otsu threshold + morphological cleaning → largest connected component → breast bounding box. Drops samples with critical mask-bbox misalignment (>50% mask outside bbox) | `step6` → `step7_breast_bbox.csv` (1592 → 1582 rows) |
| 8 | `step8_crop_resize.py` | Crops to bbox, square-pads (preserves aspect ratio), resizes to 256×256. Image: `INTER_AREA`, mask: `INTER_NEAREST` + strict binarization | `step7` → `step8_processed.csv` + 1582 PNG pairs |
| 9 | `step9_split_data.py` | Patient-level stratified split (70/15/15). No patient appears in multiple splits. Seed=42 for reproducibility | `step8` → `step9_splits.csv` |
| 10 | `step10_save_npy.py` | Stacks PNGs into NumPy arrays per split. Generates metadata CSVs, `pipeline_summary.json`, `dataset_info.txt` | `step9` → `final_npy/` |

---

## Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/irembatigunn/harmonia_vision_data_prep.git
cd harmonia_vision_data_prep
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install pandas numpy pillow opencv-python scikit-learn tqdm
```

### 2. Download Dataset
Download from [Kaggle — CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data) and extract the `csv/` and `jpeg/` folders into `CBIS DDSM Breast Cancer Dataset/`.

### 3. Run Pipeline
```bash
cd scripts
python step1_load_and_merge.py
python step2_join_metadata.py
python step3_consolidate_labels.py
python step4_file_integrity.py
python step5_mask_validation.py
python step6_combine_masks.py
python step7_breast_extraction.py
python step8_crop_resize.py
python step9_split_data.py
python step10_save_npy.py
```

Each step is **idempotent** — it reads the previous step's output and produces its own.

---

## Final Outputs

### File Structure (`outputs/final_npy/`)

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `X_train.npy` | (1102, 256, 256, 1) | uint8 | Grayscale mammogram images |
| `X_val.npy` | (237, 256, 256, 1) | uint8 | |
| `X_test.npy` | (243, 256, 256, 1) | uint8 | |
| `masks_train.npy` | (1102, 256, 256, 1) | uint8 (0/255) | Binary ROI segmentation masks |
| `masks_val.npy` | (237, 256, 256, 1) | uint8 | |
| `masks_test.npy` | (243, 256, 256, 1) | uint8 | |
| `y_train.npy` | (1102,) | uint8 (0/1) | 0 = BENIGN, 1 = MALIGNANT |
| `y_val.npy` | (237,) | uint8 | |
| `y_test.npy` | (243,) | uint8 | |

### Class Distribution

| Split | Samples | Patients | BENIGN | MALIGNANT |
|-------|---------|----------|--------|-----------|
| Train | 1102 | 620 | 581 (52.7%) | 521 (47.3%) |
| Val | 237 | 133 | 128 (54.0%) | 109 (46.0%) |
| Test | 243 | 133 | 130 (53.5%) | 113 (46.5%) |

---

## Usage Example

### PyTorch
```python
import numpy as np

X_train = np.load("outputs/final_npy/X_train.npy")        # (1102, 256, 256, 1) uint8
y_train = np.load("outputs/final_npy/y_train.npy")        # (1102,) uint8
masks   = np.load("outputs/final_npy/masks_train.npy")    # (1102, 256, 256, 1) uint8

# Normalize at training time
X = X_train.astype(np.float32) / 255.0                    # [0, 1]
M = (masks > 127).astype(np.float32)                      # binary

# PyTorch: transpose to (N, 1, H, W)
X_pt = np.transpose(X, (0, 3, 1, 2))
```

### TensorFlow / Keras
```python
import numpy as np
import tensorflow as tf

X = np.load("outputs/final_npy/X_train.npy").astype("float32") / 255.0
y = np.load("outputs/final_npy/y_train.npy")

ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(1024).batch(32)
```

---

## Key Design Decisions

1. **Mass cases only** — `calc_*.csv` excluded from the pipeline
2. **BENIGN_WITHOUT_CALLBACK → BENIGN** — original labels preserved in metadata
3. **MALIGNANT precedence** — if any lesion in a mammogram is malignant, the sample is labeled malignant
4. **JPEG-tolerant mask validation** — bimodal ratio ≥ 0.98 (not exact 0/255)
5. **Patient-based split** — prevents data leakage across train/val/test
6. **Aspect ratio preserved** — square padding before resize (no stretching)
7. **No normalization in pipeline** — stored as uint8; normalize at training time
8. **10 samples dropped** (0.6%) — critical mask-image misalignment in CBIS-DDSM

---

## Configuration

All parameters are centralized in `scripts/config.py`:

```python
IMAGE_SIZE = 256
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

The config also handles **cross-platform path resolution** (macOS ↔ Windows) automatically.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | CSV metadata management |
| `NumPy` | Array operations & NPY export |
| `Pillow` | Image loading & validation |
| `OpenCV` | Otsu thresholding, morphological ops, resize |
| `scikit-learn` | Stratified patient-level splitting |
| `tqdm` | Progress bars |

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [CBIS-DDSM on TCIA](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) |
| **Kaggle** | [awsaf49/cbis-ddsm](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) |
| **Modality** | Digital Mammography |
| **Subset** | Mass cases only |
| **Format** | JPEG (pre-converted from DICOM) |
| **Patients** | 886 unique patients (after cleaning) |
| **Samples** | 1582 mammogram-mask pairs |

---

## Authors

- **İrem Batıgün** — [GitHub](https://github.com/irembatigunn)

## Related Repositories

- **Harmonia Vision — FL Training**: *(coming soon)*
- **Harmonia Vision — Infrastructure**: *(coming soon)*

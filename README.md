# Harmonia Vision — Data Preprocessing Pipeline

**CBIS-DDSM Cropped-Image Data Cleaning for Lesion-Centered Mass Segmentation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository contains the **11-step data cleaning pipeline** for the [Harmonia Vision](https://github.com/irembatigunn) project — a privacy-preserving Federated Learning system for lesion-centered breast cancer mass segmentation using U-Net.

The pipeline transforms **CBIS-DDSM** (Curated Breast Imaging Subset of Digital Database for Screening Mammography) cropped mammogram images into clean, model-ready `.npy` arrays, avoiding the complexities of full mammogram processing by focusing directly on the lesions.

**Data source:** [CBIS-DDSM on TCIA](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) — we use the [Kaggle version (awsaf49)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) which provides JPEG-converted mammograms and ROI masks.

### Pipeline Summary

```
1696 raw rows (mass_train + mass_test CSV)
    → 11-step cleaning pipeline
    → Processed 256×256 cropped mammograms + binary masks
    → Patient-based 70/15/15 train/val/test split (FL-ready)
```

---

## Project Structure

```
harmonia_vision_data_prep/
│
├── scripts/                                 # Pipeline code (all steps)
│   ├── config.py                            # Shared paths & parameters
│   ├── step1_load_and_merge.py              # Load & merge mass CSVs
│   ├── step2_join_metadata.py               # Join DICOM metadata for cropped images
│   ├── step3_consolidate_labels.py          # Consolidate pathology labels
│   ├── step4_file_integrity.py              # File existence & corruption check
│   ├── step5_mask_validation.py             # Binary mask validation
│   ├── step6_combine_masks.py               # Crop-level duplicate mask resolver
│   ├── step7_crop_mask_validation.py        # Spatial validation for crop and mask
│   ├── step8_crop_resize.py                 # Pad & resize to 256×256
│   ├── step9_split_data.py                  # Patient-level stratified split
│   ├── step10_save_npy.py                   # Export final .npy arrays
│   ├── step11_create_fl_partitions.py       # Generate FL client partitions
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
│   ├── combined_masks/                      # Combined mask PNGs
│   ├── processed/
│   │   ├── images/                          # Preprocessed 256×256 PNGs
│   │   └── masks/                           # Preprocessed 256×256 masks
│   └── final_npy/                           # ⭐ Model-ready arrays
│       ├── X_{split}.npy                    # (N, 256, 256, 1) uint8 images
│       ├── masks_{split}.npy                # (N, 256, 256, 1) uint8 masks
│       ├── y_{split}.npy                    # (N,) uint8 labels
│       ├── metadata_{split}.csv             # Per-sample metadata
│       ├── pipeline_summary.json            # Full pipeline statistics
│       └── dataset_info.txt                 # Human-readable summary
│
├── logs/                                    # Diagnostic logs (git-ignored)
│   ├── step7_spatial_validation.csv         # Crop-mask alignment errors
│   ├── step7_diag/                          # Validation visualizations
│   └── step8_diag/                          # 256×256 output samples
│
├── .gitignore
└── README.md
```

---

## Pipeline Steps

| # | Script | Description |
|---|--------|-------------|
| 1 | `step1_load_and_merge.py` | Loads `mass_train` + `mass_test` CSVs, normalizes column names, merges into single DataFrame. |
| 2 | `step2_join_metadata.py` | Joins with `dicom_info.csv` to resolve JPEG paths. Extracts absolute paths for `cropped_images` and their ROI masks. |
| 3 | `step3_consolidate_labels.py` | Merges `BENIGN_WITHOUT_CALLBACK` → `BENIGN`. Adds binary `label` column (0/1). |
| 4 | `step4_file_integrity.py` | Validates every image and mask file exists and is not corrupt (PIL open test). |
| 5 | `step5_mask_validation.py` | Verifies masks are truly binary via bimodal pixel analysis. Detects empty masks. |
| 6 | `step6_combine_masks.py` | Resolves multiple ROI mask candidates for the same crop (`patient_id`, `breast`, `view`, `abnormality_id`). |
| 7 | `step7_crop_mask_validation.py` | Validates crop and mask spatial alignment (resolution consistency, mask area ratio, edge-touching, misalignment). |
| 8 | `step8_crop_resize.py` | Square-pads (preserves aspect ratio) and resizes to 256×256. Image: `INTER_AREA`, mask: `INTER_NEAREST`. |
| 9 | `step9_split_data.py` | Patient-level stratified split (70/15/15). No patient appears in multiple splits. Seed=42 for reproducibility. |
| 10 | `step10_save_npy.py` | Stacks PNGs into NumPy arrays per split. Generates metadata CSVs, `pipeline_summary.json`. |
| 11 | `step11_create_fl_partitions.py` | Generates IID and Non-IID client partitions from the training set for Federated Learning experiments. |

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
python step7_crop_mask_validation.py
python step8_crop_resize.py
python step9_split_data.py
python step10_save_npy.py
python step11_create_fl_partitions.py
```

Each step is **idempotent** — it reads the previous step's output and produces its own.

---

## Final Outputs

### File Structure (`outputs/final_npy/`)

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `X_{split}.npy` | (N, 256, 256, 1) | uint8 | Grayscale cropped lesion mammograms |
| `masks_{split}.npy` | (N, 256, 256, 1) | uint8 (0/255) | Binary ROI segmentation masks |
| `y_{split}.npy` | (N,) | uint8 (0/1) | 0 = BENIGN, 1 = MALIGNANT |

---

## Key Design Decisions

1. **Cropped Lesion Inputs** — The pipeline utilizes cropped images directly to simplify the model task to lesion segmentation rather than full-breast search.
2. **BENIGN_WITHOUT_CALLBACK → BENIGN** — original labels preserved in metadata.
3. **Aspect ratio preserved** — square padding before resize (no stretching).
4. **Patient-based split** — prevents data leakage across train/val/test.
5. **Federated Learning Ready** — Step 11 creates predefined IID and Non-IID metadata splits across simulated clients.

---

## Authors

- **İrem Batıgün** — [GitHub](https://github.com/irembatigunn)

"""
Step 10: Nihai .npy çıktılarını kaydet.

Her split (train/val/test) için:
  - X_{split}.npy      : (N, 256, 256, 1) uint8 görüntüler (Otsu+CLAHE işlenmiş)
  - y_{split}.npy      : (N,) uint8 binary etiket (0=BENIGN, 1=MALIGNANT)
  - masks_{split}.npy  : (N, 256, 256, 1) uint8 binary maske (0/255)
  - metadata_{split}.csv: pipeline için tüm metadata

Ek olarak: pipeline_summary.json, dataset_info.txt.

Girdi : outputs/step9_splits.csv
Çıktı : outputs/final_npy/
"""
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import OUTPUT_DIR, IMAGE_SIZE


FINAL_DIR = OUTPUT_DIR / "final_npy"


def build_arrays(sub_df: pd.DataFrame) -> tuple:
    n = len(sub_df)
    X = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
    M = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
    y = np.empty(n, dtype=np.uint8)
    for i, (_, row) in enumerate(tqdm(sub_df.iterrows(), total=n)):
        img = np.asarray(Image.open(row["processed_image_path"]).convert("L"),
                         dtype=np.uint8)
        msk = np.asarray(Image.open(row["processed_mask_path"]).convert("L"),
                         dtype=np.uint8)
        msk = ((msk > 127).astype(np.uint8)) * 255  # kesin binary
        X[i, :, :, 0] = img
        M[i, :, :, 0] = msk
        y[i] = int(row["label"])
    return X, M, y


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(OUTPUT_DIR / "step9_splits.csv")
    print(f"[in] step9_splits.csv shape: {df.shape}")

    summary = {
        "image_size": IMAGE_SIZE,
        "preprocessing": "otsu_background_removal + clahe",
        "image_type": "full_mammogram (not cropped)",
        "class_mapping": {"BENIGN": 0, "MALIGNANT": 1},
        "splits": {},
        "total_samples": int(len(df)),
        "total_patients": int(df["patient_id"].nunique()),
    }

    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name].reset_index(drop=True)
        print(f"\n[{split_name}] {len(sub)} samples, "
              f"{sub['patient_id'].nunique()} patients")

        X, M, y = build_arrays(sub)

        np.save(FINAL_DIR / f"X_{split_name}.npy", X)
        np.save(FINAL_DIR / f"masks_{split_name}.npy", M)
        np.save(FINAL_DIR / f"y_{split_name}.npy", y)

        # Metadata CSV
        meta_cols = [
            "patient_id", "left_or_right_breast", "image_view",
            "pathology", "label", "pathology_original_combined",
            "n_abnormalities", "breast_density", "assessment",
            "subtlety", "collection", "mask_fg_ratio", "mask_align_ratio",
            "breast_area_ratio",
        ]
        available_cols = [c for c in meta_cols if c in sub.columns]
        sub[available_cols].to_csv(FINAL_DIR / f"metadata_{split_name}.csv",
                                   index=False)

        # Maske foreground istatistikleri
        fg_per_sample = (M > 127).reshape(len(M), -1).sum(axis=1)
        fg_ratios = fg_per_sample / (IMAGE_SIZE * IMAGE_SIZE)

        split_info = {
            "n_samples": int(len(sub)),
            "n_patients": int(sub["patient_id"].nunique()),
            "n_benign": int((y == 0).sum()),
            "n_malignant": int((y == 1).sum()),
            "shape_X": list(X.shape),
            "shape_masks": list(M.shape),
            "shape_y": list(y.shape),
            "dtype": str(X.dtype),
            "image_min_max": [int(X.min()), int(X.max())],
            "mask_unique": sorted(np.unique(M).tolist()),
            "mask_fg_ratio_mean": float(fg_ratios.mean()),
            "mask_fg_ratio_median": float(np.median(fg_ratios)),
            "mask_fg_ratio_min": float(fg_ratios.min()),
            "mask_fg_ratio_max": float(fg_ratios.max()),
        }
        summary["splits"][split_name] = split_info

        print(f"  X: {X.shape} {X.dtype}  masks: {M.shape} {M.dtype}  y: {y.shape}")
        print(f"  benign: {split_info['n_benign']}  "
              f"malignant: {split_info['n_malignant']}")
        print(f"  mask fg ratio: mean={fg_ratios.mean():.4f}, "
              f"median={np.median(fg_ratios):.4f}")

    with open(FINAL_DIR / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # İnsan-okunabilir özet
    lines = [
        "CBIS-DDSM Mass Dataset - Full-Image Pipeline (Otsu+CLAHE)",
        "=" * 60,
        f"Goruntu boyutu   : {IMAGE_SIZE}x{IMAGE_SIZE}",
        f"Preprocessing    : Otsu background removal + CLAHE",
        f"Image type       : Full mammogram (no crop)",
        f"Toplam sample    : {summary['total_samples']}",
        f"Toplam hasta     : {summary['total_patients']}",
        f"Sinif eslemesi   : BENIGN=0, MALIGNANT=1",
        f"Dtype            : uint8 (image 0-255, mask 0/255)",
        "",
        "Split istatistikleri:",
    ]
    for s, info in summary["splits"].items():
        lines.append(
            f"  {s:5s}: {info['n_samples']:4d} sample "
            f"({info['n_patients']:3d} patient) | "
            f"BENIGN {info['n_benign']:3d} / MALIGNANT {info['n_malignant']:3d} | "
            f"fg_ratio_mean={info['mask_fg_ratio_mean']:.4f}"
        )
    lines += [
        "",
        "Dosyalar:",
        "  X_{split}.npy       (N,256,256,1) uint8 goruntuler (Otsu+CLAHE)",
        "  masks_{split}.npy   (N,256,256,1) uint8 maskeler (0/255)",
        "  y_{split}.npy       (N,)          uint8 etiketler",
        "  metadata_{split}.csv                pathology, patient, view, fg_ratio vb.",
        "",
        "NOT: Bu pipeline full mammogram kullanir (crop yok).",
        "Mask foreground orani dusuk olabilir (~%0.3) — bu normaldir.",
        "Model egitiminde Dice Loss agirligi arttirilmali.",
    ]
    with open(FINAL_DIR / "dataset_info.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    print(f"\n[ok] Tum ciktilar: {FINAL_DIR}")


if __name__ == "__main__":
    main()

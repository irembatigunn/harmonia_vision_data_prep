"""
Step 8: Square pad and resize cropped images.

Her geçerli görüntü için:
  1. Görüntüyü ve maskeyi yükle.
  2. Square pad (aspect ratio korunarak, dar kenara siyah padding).
  3. 256x256 resize.
  4. Maskeyi kesin binarize et.

Full mammogram kullanılmadığı için CLAHE veya Otsu background removal 
gibi karmaşık işlemlere gerek yoktur. Sadece resize yapılıp kaydedilir.

Girdi : outputs/step7_validated.csv
Çıktı : outputs/step8_processed.csv
        outputs/processed/images/  (256×256 PNG)
        outputs/processed/masks/   (256×256 PNG)
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import OUTPUT_DIR, LOG_DIR, IMAGE_SIZE, fix_path


PROCESSED_DIR = OUTPUT_DIR / "processed"
IMG_OUT_DIR = PROCESSED_DIR / "images"
MASK_OUT_DIR = PROCESSED_DIR / "masks"


def pad_to_square(arr: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """En-boy oranını koruyarak kare yap (dar kenara padding)."""
    h, w = arr.shape[:2]
    if h == w:
        return arr
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bot = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    return cv2.copyMakeBorder(arr, pad_top, pad_bot, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=pad_value)


def process_single(img_path: str, mask_path: str, size: int = IMAGE_SIZE) -> tuple:
    """Tek bir görüntü-maske çifti: pad + resize."""
    with Image.open(img_path) as im:
        img = np.asarray(im.convert("L")).astype(np.uint8)

    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L")).astype(np.uint8)

    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    img_sq = pad_to_square(img, pad_value=0)
    mask_sq = pad_to_square(mask, pad_value=0)

    img_r = cv2.resize(img_sq, (size, size), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask_sq, (size, size), interpolation=cv2.INTER_NEAREST)
    mask_r = ((mask_r > 127).astype(np.uint8)) * 255

    return img_r, mask_r


def main():
    IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step7_validated.csv"
    df = pd.read_csv(in_path)
    print(f"[in] {in_path.name} shape: {df.shape}")

    out_rows = []
    dropped = 0

    for i, row in tqdm(list(df.iterrows()), total=len(df)):
        img_path = fix_path(row["cropped_image_abs_path"])
        mask_path = fix_path(row["combined_mask_abs_path"])
        abnormality = row.get("abnormality_id", "")

        try:
            img_r, mask_r = process_single(img_path, mask_path, IMAGE_SIZE)
        except Exception as e:
            print(f"[warn] {row['patient_id']} "
                  f"{row['left_or_right_breast']}_{row['image_view']}_{abnormality}: "
                  f"{type(e).__name__}: {e}")
            dropped += 1
            continue

        name = f"{row['patient_id']}_{row['left_or_right_breast']}_{row['image_view']}_{abnormality}"
        img_out = IMG_OUT_DIR / f"{name}.png"
        mask_out = MASK_OUT_DIR / f"{name}.png"
        
        Image.fromarray(img_r).save(img_out)
        Image.fromarray(mask_r).save(mask_out)

        out_rows.append({
            "patient_id": row["patient_id"],
            "left_or_right_breast": row["left_or_right_breast"],
            "image_view": row["image_view"],
            "abnormality_id": abnormality,
            "pathology": row["pathology"],
            "label": row["label"],
            "pathology_original": row.get("pathology_original", ""),
            "breast_density": row.get("breast_density", ""),
            "assessment": row.get("assessment", ""),
            "subtlety": row.get("subtlety", ""),
            "source_split": row.get("source_split", ""),
            "processed_image_path": str(img_out),
            "processed_mask_path": str(mask_out),
            "mask_area_ratio": row.get("mask_area_ratio", ""),
            "preprocessing": "square_pad+resize",
        })

    out_df = pd.DataFrame(out_rows)

    print(f"\n{'='*60}")
    print(f"STEP 8 RAPOR: Pad + Resize")
    print(f"{'='*60}")
    print(f"  Giris       : {len(df)}")
    print(f"  Islenen     : {len(out_df)}")
    print(f"  Atilan      : {dropped}")

    print(f"\n  Sinif dagilimi:")
    print(out_df["pathology"].value_counts().to_string())

    out_csv = OUTPUT_DIR / "step8_processed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[ok] {out_csv}")
    print(f"[ok] Goruntuler: {IMG_OUT_DIR}")
    print(f"[ok] Maskeler  : {MASK_OUT_DIR}")


if __name__ == "__main__":
    main()

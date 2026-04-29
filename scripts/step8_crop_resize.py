"""
Step 8: Full-image Otsu background removal + CLAHE + resize.

Her geçerli görüntü için:
  1. Otsu threshold → meme bölgesi tespiti (background sıfırlama)
  2. CLAHE kontrast artırımı (sadece meme bölgesine)
  3. Square pad (aspect ratio korunur) + 256×256 resize
  4. ROI maske aynı transform ile işlenir (INTER_NEAREST)

Full mammogram korunur — tight crop YOK.
Model "tüm mamogramda kitleyi bul ve segment et" görevini öğrenir.

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


# --- CLAHE Parametreleri ---
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# --- Otsu Parametreleri (step7 ile aynı) ---
BORDER_CLEAR = 30
OPEN_KSIZE = 15

# Çıktı klasörleri
PROCESSED_DIR = OUTPUT_DIR / "processed"
IMG_OUT_DIR = PROCESSED_DIR / "images"
MASK_OUT_DIR = PROCESSED_DIR / "masks"


def get_breast_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu + kenar temizliği + largest CC -> binary meme maskesi."""
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    proc = cv2.medianBlur(gray, 5)
    _, binar = cv2.threshold(proc, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if BORDER_CLEAR > 0:
        binar[:BORDER_CLEAR, :] = 0
        binar[-BORDER_CLEAR:, :] = 0
        binar[:, :BORDER_CLEAR] = 0
        binar[:, -BORDER_CLEAR:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (OPEN_KSIZE, OPEN_KSIZE))
    binar = cv2.morphologyEx(binar, cv2.MORPH_OPEN, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binar, connectivity=8)
    if num <= 1:
        return np.ones_like(gray, dtype=np.uint8) * 255

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return ((labels == largest_idx).astype(np.uint8)) * 255


def apply_clahe(gray: np.ndarray, breast_mask: np.ndarray) -> np.ndarray:
    """CLAHE'yi sadece meme bölgesine uygula, arkaplan siyah kalır."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced = clahe.apply(gray)
    return np.where(breast_mask > 0, enhanced, 0).astype(np.uint8)


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


def process_single(img_path: str, mask_path: str,
                   size: int = IMAGE_SIZE) -> tuple:
    """Tek bir görüntü-maske çifti: Otsu + CLAHE + pad + resize."""
    # Görüntü yükle
    with Image.open(img_path) as im:
        img = np.asarray(im.convert("L")).astype(np.uint8)

    # ROI maske yükle
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L")).astype(np.uint8)

    # Maske boyutunu görüntüye eşitle
    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # 1. Otsu background removal
    breast_mask = get_breast_mask(img)

    # 2. CLAHE (sadece meme bölgesine)
    clahe_img = apply_clahe(img, breast_mask)

    # 3. Maske: breast bölgesi dışını temizle + binarize
    mask_cleaned = np.where(breast_mask > 0, mask, 0).astype(np.uint8)
    mask_cleaned = ((mask_cleaned > 127).astype(np.uint8)) * 255

    # 4. Square pad + resize
    img_sq = pad_to_square(clahe_img, pad_value=0)
    mask_sq = pad_to_square(mask_cleaned, pad_value=0)

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
    fg_ratios = []
    dropped = 0

    for i, row in tqdm(list(df.iterrows()), total=len(df)):
        img_path = fix_path(row["full_image_abs_path"])
        mask_path = fix_path(row["combined_mask_abs_path"])

        try:
            img_r, mask_r = process_single(img_path, mask_path, IMAGE_SIZE)
        except Exception as e:
            print(f"[warn] {row['patient_id']} "
                  f"{row['left_or_right_breast']}_{row['image_view']}: "
                  f"{type(e).__name__}: {e}")
            dropped += 1
            continue

        # Foreground kontrolü
        fg = int((mask_r > 127).sum())
        if fg < 10:
            dropped += 1
            continue

        fg_ratio = fg / (IMAGE_SIZE * IMAGE_SIZE)
        fg_ratios.append(fg_ratio)

        # Kaydet
        name = f"{row['patient_id']}_{row['left_or_right_breast']}_{row['image_view']}"
        img_out = IMG_OUT_DIR / f"{name}.png"
        mask_out = MASK_OUT_DIR / f"{name}.png"
        Image.fromarray(img_r).save(img_out)
        Image.fromarray(mask_r).save(mask_out)

        out_rows.append({
            "patient_id": row["patient_id"],
            "left_or_right_breast": row["left_or_right_breast"],
            "image_view": row["image_view"],
            "pathology": row["pathology"],
            "label": row["label"],
            "pathology_original_combined": row["pathology_original_combined"],
            "abnormality_ids": row["abnormality_ids"],
            "n_abnormalities": row["n_abnormalities"],
            "breast_density": row.get("breast_density", ""),
            "assessment": row.get("assessment", ""),
            "subtlety": row.get("subtlety", ""),
            "collection": row.get("collection", ""),
            "source_split": row.get("source_split", ""),
            "processed_image_path": str(img_out),
            "processed_mask_path": str(mask_out),
            "mask_fg_pixels": fg,
            "mask_fg_ratio": fg_ratio,
            "breast_area_ratio": row.get("breast_area_ratio", ""),
            "mask_align_ratio": row.get("mask_align_ratio", ""),
            "preprocessing": "otsu_bg_removal+clahe",
        })

    out_df = pd.DataFrame(out_rows)

    # Rapor
    print(f"\n{'='*60}")
    print(f"STEP 8 RAPOR: Otsu + CLAHE + Resize")
    print(f"{'='*60}")
    print(f"  Giris       : {len(df)}")
    print(f"  Islenen     : {len(out_df)}")
    print(f"  Atilan      : {dropped}")

    if fg_ratios:
        fr = np.asarray(fg_ratios)
        print(f"\n  Maske foreground orani (256x256):")
        print(f"    mean   : {fr.mean():.4f}")
        print(f"    median : {np.median(fr):.4f}")
        print(f"    min    : {fr.min():.4f}")
        print(f"    max    : {fr.max():.4f}")

    print(f"\n  Sinif dagilimi:")
    print(out_df["pathology"].value_counts().to_string())

    out_csv = OUTPUT_DIR / "step8_processed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[ok] {out_csv}")
    print(f"[ok] Goruntuler: {IMG_OUT_DIR}")
    print(f"[ok] Maskeler  : {MASK_OUT_DIR}")


if __name__ == "__main__":
    main()

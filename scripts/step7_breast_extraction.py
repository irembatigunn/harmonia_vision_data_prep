"""
Step 7: Breast region validation & filtering.

Her full mammogram için:
  1. Otsu threshold + largest CC → meme bölgesi tespiti
  2. ROI mask-breast bölge hizalama kontrolü
  3. Kritik hizalama hatası olan satırlar atılır (CBIS-DDSM veri hatası)
  4. Breast bölge istatistikleri CSV'ye yazılır

Bu adım sadece FİLTRELEME + DOĞRULAMA yapar.
Görüntü işleme (CLAHE, resize) Step 8'de yapılır.

Girdi : outputs/step6_masks_combined.csv
Çıktı : outputs/step7_validated.csv
Log   : logs/step7_breast_extraction.csv
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import OUTPUT_DIR, LOG_DIR, fix_path


# --- Sabitler ---
BORDER_CLEAR = 30       # Kenar piksel temizliği (scanner marker)
OPEN_KSIZE = 15         # Morfolojik opening kernel boyutu
MASK_ALIGN_THRESH = 0.50  # Maske hizalama eşik değeri


def get_breast_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu + kenar temizliği + largest CC -> binary meme maskesi.

    Returns:
        breast_mask: uint8 (0/255) — meme bölgesi 255, arkaplan 0
    """
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


def load_gray(path: str) -> np.ndarray:
    """PIL ile açıp grayscale numpy döndür (8-bit)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")).astype(np.uint8)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step6_masks_combined.csv"
    df = pd.read_csv(in_path)
    print(f"[in] {in_path.name} shape: {df.shape}")

    results = []
    warnings = []

    for i, row in tqdm(list(df.iterrows()), total=len(df)):
        img_path = fix_path(row["full_image_abs_path"])
        mask_path = fix_path(row["combined_mask_abs_path"])
        pid = row["patient_id"]
        view = f"{row['left_or_right_breast']}_{row['image_view']}"

        try:
            img = load_gray(img_path)
        except Exception as e:
            warnings.append({
                "patient_id": pid, "view": view, "path": img_path,
                "reason": f"image_load_fail:{type(e).__name__}",
            })
            results.append({"idx": i, "valid": False})
            continue

        # Meme bölgesi tespiti
        breast_mask = get_breast_mask(img)
        breast_area_ratio = float((breast_mask > 0).sum()) / (img.shape[0] * img.shape[1])

        # ROI maske yükle ve hizalama kontrolü
        try:
            mask = load_gray(mask_path)
            if mask.shape != img.shape:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            roi_bin = mask > 127
            breast_bin = breast_mask > 127
            total_roi = int(roi_bin.sum())
            inside = int((roi_bin & breast_bin).sum())
            align_ratio = (inside / total_roi) if total_roi > 0 else 0.0
        except Exception as e:
            warnings.append({
                "patient_id": pid, "view": view, "path": mask_path,
                "reason": f"mask_check_fail:{type(e).__name__}",
            })
            results.append({"idx": i, "valid": False})
            continue

        # Kritik hizalama sorunu
        if align_ratio < MASK_ALIGN_THRESH:
            warnings.append({
                "patient_id": pid, "view": view, "path": img_path,
                "reason": f"critical_mask_misalign:ratio={align_ratio:.3f}",
            })
            results.append({"idx": i, "valid": False})
            continue

        # Boş maske kontrolü
        if total_roi < 10:
            warnings.append({
                "patient_id": pid, "view": view, "path": img_path,
                "reason": f"empty_roi_mask:fg_pixels={total_roi}",
            })
            results.append({"idx": i, "valid": False})
            continue

        results.append({
            "idx": i,
            "valid": True,
            "img_height": img.shape[0],
            "img_width": img.shape[1],
            "breast_area_ratio": breast_area_ratio,
            "mask_align_ratio": align_ratio,
            "mask_fg_pixels_original": total_roi,
        })

    # Geçerli satırları filtrele
    valid_indices = [r["idx"] for r in results if r["valid"]]
    valid_stats = {r["idx"]: r for r in results if r["valid"]}

    out_df = df.iloc[valid_indices].copy()
    out_df["img_height"] = [valid_stats[i]["img_height"] for i in valid_indices]
    out_df["img_width"] = [valid_stats[i]["img_width"] for i in valid_indices]
    out_df["breast_area_ratio"] = [valid_stats[i]["breast_area_ratio"] for i in valid_indices]
    out_df["mask_align_ratio"] = [valid_stats[i]["mask_align_ratio"] for i in valid_indices]
    out_df["mask_fg_pixels_original"] = [valid_stats[i]["mask_fg_pixels_original"] for i in valid_indices]

    # Rapor
    dropped = len(df) - len(out_df)
    print(f"\n{'='*60}")
    print(f"STEP 7 RAPOR: Validation & Filtering")
    print(f"{'='*60}")
    print(f"  Giris         : {len(df)}")
    print(f"  Gecerli       : {len(out_df)}")
    print(f"  Atilan        : {dropped} ({dropped/len(df):.1%})")
    print(f"  Breast area   : mean={out_df['breast_area_ratio'].mean():.3f}")
    print(f"  Align ratio   : min={out_df['mask_align_ratio'].min():.3f}")
    print(f"\n  Sinif dagilimi:")
    print(out_df["pathology"].value_counts().to_string())

    # Log
    log_df = pd.DataFrame(warnings)
    log_path = LOG_DIR / "step7_breast_extraction.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[log] {len(log_df)} uyari -> {log_path}")
    if len(log_df):
        print(log_df["reason"].apply(lambda s: s.split(":")[0]).value_counts().to_string())

    # Kaydet
    out_csv = OUTPUT_DIR / "step7_validated.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[ok] {out_csv}")


if __name__ == "__main__":
    main()

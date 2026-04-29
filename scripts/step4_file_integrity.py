"""
Step 4: Fiziksel dosya kontrolleri.
- os.path.exists() ile tum full mammogram ve ROI mask adayi dosyalari kontrol.
- PIL/cv2 ile dosyayi acmaya calisarak corrupt (bozuk) dosyalari tespit et.
- Eksik/bozuk dosyalari logla; full mammogram bozuksa veya tum maske adaylari
  bozuksa satiri at. Bazi adaylar bozuksa sadece bozuk olanlari eler.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from config import OUTPUT_DIR, LOG_DIR


def check_image(path: str) -> tuple[bool, str]:
    """(ok, reason). ok=False ise reason ne oldugunu soyler."""
    if not isinstance(path, str) or not path:
        return False, "empty_path"
    if not os.path.exists(path):
        return False, "missing"
    try:
        with Image.open(path) as im:
            im.verify()  # header dogrulamasi
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, f"corrupt:{type(e).__name__}"
    # verify() sonrasi dosya yeniden acilip gercekten decode edilmeli
    try:
        with Image.open(path) as im:
            arr = np.asarray(im)
            if arr.size == 0:
                return False, "empty_array"
    except Exception as e:
        return False, f"decode_fail:{type(e).__name__}"
    return True, "ok"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step3_labels_consolidated.csv"
    df = pd.read_csv(in_path)
    print(f"[in] {in_path.name} shape: {df.shape}")

    # ROI mask aday listesini geri ac
    df["roi_mask_abs_paths_list"] = df["roi_mask_abs_paths"].fillna("").apply(
        lambda s: [p for p in s.split("|") if p]
    )

    log_rows = []

    # --- 1. Full mammogram kontrolu ---
    print("\n[check] full mammogram dosyalari...")
    full_ok = []
    full_reason = []
    for path in tqdm(df["full_image_abs_path"].tolist()):
        ok, reason = check_image(path)
        full_ok.append(ok)
        full_reason.append(reason)

    df["full_ok"] = full_ok
    df["full_check_reason"] = full_reason

    for i, row in df[~df["full_ok"]].iterrows():
        log_rows.append({
            "kind": "full_mammogram",
            "patient_id": row["patient_id"],
            "view": f"{row['left_or_right_breast']}_{row['image_view']}",
            "abnormality_id": row["abnormality_id"],
            "path": row["full_image_abs_path"],
            "reason": row["full_check_reason"],
        })

    n_full_bad = (~df["full_ok"]).sum()
    print(f"  bozuk/eksik full mammogram: {n_full_bad} / {len(df)}")

    # --- 2. ROI mask adaylari kontrolu ---
    print("\n[check] ROI mask adaylari...")
    valid_masks_list = []
    n_candidates_total = sum(len(x) for x in df["roi_mask_abs_paths_list"])
    pbar = tqdm(total=n_candidates_total)

    for i, row in df.iterrows():
        valid = []
        for path in row["roi_mask_abs_paths_list"]:
            ok, reason = check_image(path)
            pbar.update(1)
            if ok:
                valid.append(path)
            else:
                log_rows.append({
                    "kind": "mask_candidate",
                    "patient_id": row["patient_id"],
                    "view": f"{row['left_or_right_breast']}_{row['image_view']}",
                    "abnormality_id": row["abnormality_id"],
                    "path": path,
                    "reason": reason,
                })
        valid_masks_list.append(valid)
    pbar.close()

    df["valid_mask_paths"] = valid_masks_list
    df["n_valid_masks"] = df["valid_mask_paths"].apply(len)

    n_no_masks = (df["n_valid_masks"] == 0).sum()
    print(f"  hicbir gecerli maske bulunmayan satir: {n_no_masks} / {len(df)}")
    print(f"  gecerli maske sayisi dagilimi:")
    print(df["n_valid_masks"].value_counts().sort_index())

    # --- 3. Rapor + log kaydet ---
    log_df = pd.DataFrame(log_rows)
    log_path = LOG_DIR / "step4_file_integrity.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[log] sorunlu dosya kayitlari: {len(log_df)} -> {log_path}")
    if len(log_df):
        print("  reason dagilimi:")
        print(log_df["reason"].value_counts())

    # --- 4. Satir eleme ---
    before = len(df)
    keep = df["full_ok"] & (df["n_valid_masks"] > 0)
    df_clean = df[keep].copy()
    dropped = before - len(df_clean)
    print(f"\n[drop] tam sebepten atilan satir: {dropped} "
          f"({before} -> {len(df_clean)})")

    # Listeyi pipe-string'e geri cevir ve kaydet
    df_clean["valid_mask_paths_str"] = df_clean["valid_mask_paths"].apply(
        lambda x: "|".join(x)
    )
    df_clean = df_clean.drop(columns=[
        "roi_mask_abs_paths_list", "valid_mask_paths",
        "full_ok", "full_check_reason"
    ])

    out = OUTPUT_DIR / "step4_files_verified.csv"
    df_clean.to_csv(out, index=False)
    print(f"\n[ok] Kaydedildi: {out}")


if __name__ == "__main__":
    main()

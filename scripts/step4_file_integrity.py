"""
Step 4: Physical file integrity checks.
- Verifies existence and readability of cropped mammograms and ROI mask candidates.
- Drops rows if cropped image is corrupt or missing.
- Filters out corrupt ROI mask candidates.
"""
import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from config import OUTPUT_DIR, LOG_DIR


def check_image(path: str) -> tuple[bool, str]:
    """Returns (ok, reason)."""
    if not isinstance(path, str) or not path:
        return False, "empty_path"
    if not os.path.exists(path):
        return False, "missing"
    try:
        with Image.open(path) as im:
            im.verify()  # header verification
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, f"corrupt:{type(e).__name__}"
    
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

    # Reconstruct ROI mask candidate list
    df["roi_mask_abs_paths_list"] = df["roi_mask_abs_paths"].fillna("").apply(
        lambda s: [p for p in s.split("|") if p]
    )

    log_rows = []

    # --- 1. Check cropped images ---
    print("\n[check] cropped image files...")
    crop_ok = []
    crop_reason = []
    for path in tqdm(df["cropped_image_abs_path"].tolist()):
        ok, reason = check_image(path)
        crop_ok.append(ok)
        crop_reason.append(reason)

    df["crop_ok"] = crop_ok
    df["crop_check_reason"] = crop_reason

    for i, row in df[~df["crop_ok"]].iterrows():
        log_rows.append({
            "kind": "cropped_image",
            "patient_id": row["patient_id"],
            "view": f"{row['left_or_right_breast']}_{row['image_view']}",
            "abnormality_id": row["abnormality_id"],
            "path": row["cropped_image_abs_path"],
            "reason": row["crop_check_reason"],
        })

    n_crop_bad = (~df["crop_ok"]).sum()
    print(f"  corrupt/missing cropped images: {n_crop_bad} / {len(df)}")

    # --- 2. Check ROI mask candidates ---
    print("\n[check] ROI mask candidates...")
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
    print(f"  rows with no valid mask candidates: {n_no_masks} / {len(df)}")
    print(f"  valid mask counts distribution:")
    print(df["n_valid_masks"].value_counts().sort_index())

    # --- 3. Save logs ---
    log_df = pd.DataFrame(log_rows)
    log_path = LOG_DIR / "step4_file_integrity.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[log] problematic files logged: {len(log_df)} -> {log_path}")

    # --- 4. Drop invalid rows ---
    before = len(df)
    keep = df["crop_ok"] & (df["n_valid_masks"] > 0)
    df_clean = df[keep].copy()
    dropped = before - len(df_clean)
    print(f"\n[drop] rows dropped due to missing/corrupt files: {dropped} "
          f"({before} -> {len(df_clean)})")

    # Serialize valid mask paths back to string
    df_clean["valid_mask_paths_str"] = df_clean["valid_mask_paths"].apply(
        lambda x: "|".join(x)
    )
    df_clean = df_clean.drop(columns=[
        "roi_mask_abs_paths_list", "valid_mask_paths",
        "crop_ok", "crop_check_reason"
    ])

    out = OUTPUT_DIR / "step4_files_verified.csv"
    df_clean.to_csv(out, index=False)
    print(f"\n[ok] Saved: {out}")


if __name__ == "__main__":
    main()

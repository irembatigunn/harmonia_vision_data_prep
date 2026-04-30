"""
Step 7: Crop and mask spatial validation.

Validates the following for cropped images and their corresponding ROI masks:
1. Resolution consistency (crop and mask sizes match).
2. Empty masks (foreground pixels < 10).
3. Mask area ratio (mask shouldn't cover > 95% or < 0.1% of the crop).
4. Edge-touching masks (if the mask touches the very border, it might be truncated).
5. Obvious crop-mask misalignment (mask center of mass too far from crop center).

Rows failing critical checks are dropped.
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import center_of_mass

from config import OUTPUT_DIR, LOG_DIR, fix_path


# --- Thresholds ---
MIN_FG_PIXELS = 10
MAX_AREA_RATIO = 0.95
MIN_AREA_RATIO = 0.001
CENTER_TOLERANCE = 0.45  # Mask center can be up to 45% away from crop center


def load_gray(path: str) -> np.ndarray:
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
        img_path = fix_path(row["cropped_image_abs_path"])
        mask_path = fix_path(row["combined_mask_abs_path"])
        pid = row["patient_id"]
        view = f"{row['left_or_right_breast']}_{row['image_view']}"
        abnormality = row.get("abnormality_id", "")

        try:
            img = load_gray(img_path)
            mask = load_gray(mask_path)
        except Exception as e:
            warnings.append({
                "patient_id": pid, "view": view, "abnormality": abnormality,
                "reason": f"load_fail:{type(e).__name__}",
            })
            results.append({"idx": i, "valid": False})
            continue

        h, w = img.shape

        # 1. Resolution Consistency
        if mask.shape != img.shape:
            # We can resize, but we log it.
            warnings.append({
                "patient_id": pid, "view": view, "abnormality": abnormality,
                "reason": f"resolution_mismatch:{img.shape}_vs_{mask.shape}",
            })
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        binar = mask > 127
        fg_pixels = int(binar.sum())

        # 2. Empty mask
        if fg_pixels < MIN_FG_PIXELS:
            warnings.append({
                "patient_id": pid, "view": view, "abnormality": abnormality,
                "reason": f"empty_mask:fg={fg_pixels}",
            })
            results.append({"idx": i, "valid": False})
            continue

        # 3. Mask Area Ratio
        area_ratio = fg_pixels / (h * w)
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            warnings.append({
                "patient_id": pid, "view": view, "abnormality": abnormality,
                "reason": f"anomalous_area_ratio:{area_ratio:.4f}",
            })
            # Drop if extremely large (probably background mislabeled)
            if area_ratio > MAX_AREA_RATIO:
                results.append({"idx": i, "valid": False})
                continue

        # 4. Edge-touching
        touches_edge = False
        if binar[0, :].any() or binar[-1, :].any() or binar[:, 0].any() or binar[:, -1].any():
            touches_edge = True
            warnings.append({
                "patient_id": pid, "view": view, "abnormality": abnormality,
                "reason": "edge_touching",
            })

        # 5. Misalignment (Center of Mass check)
        cy, cx = center_of_mass(binar)
        img_cy, img_cx = h / 2.0, w / 2.0
        dist_y = abs(cy - img_cy) / h
        dist_x = abs(cx - img_cx) / w

        if dist_y > CENTER_TOLERANCE or dist_x > CENTER_TOLERANCE:
            warnings.append({
                "patient_id": pid, "view": view, "abnormality": abnormality,
                "reason": f"center_misaligned:dx={dist_x:.2f},dy={dist_y:.2f}",
            })
            results.append({"idx": i, "valid": False})
            continue

        results.append({
            "idx": i,
            "valid": True,
            "crop_height": h,
            "crop_width": w,
            "mask_area_ratio": area_ratio,
            "touches_edge": touches_edge,
            "mask_dist_from_center_x": dist_x,
            "mask_dist_from_center_y": dist_y,
        })

    # Filter valid
    valid_indices = [r["idx"] for r in results if r["valid"]]
    valid_stats = {r["idx"]: r for r in results if r["valid"]}

    out_df = df.iloc[valid_indices].copy()
    out_df["crop_height"] = [valid_stats[i]["crop_height"] for i in valid_indices]
    out_df["crop_width"] = [valid_stats[i]["crop_width"] for i in valid_indices]
    out_df["mask_area_ratio"] = [valid_stats[i]["mask_area_ratio"] for i in valid_indices]
    out_df["touches_edge"] = [valid_stats[i]["touches_edge"] for i in valid_indices]
    
    dropped = len(df) - len(out_df)
    print(f"\n{'='*60}")
    print(f"STEP 7 RAPOR: Spatial Validation")
    print(f"{'='*60}")
    print(f"  Input         : {len(df)}")
    print(f"  Valid         : {len(out_df)}")
    print(f"  Dropped       : {dropped} ({dropped/len(df):.1%})")
    
    if len(out_df):
        print(f"  Area ratio    : mean={out_df['mask_area_ratio'].mean():.3f}")
        print(f"  Edge touching : {out_df['touches_edge'].sum()} samples")

    log_df = pd.DataFrame(warnings)
    log_path = LOG_DIR / "step7_spatial_validation.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[log] {len(log_df)} warnings -> {log_path}")

    out_csv = OUTPUT_DIR / "step7_validated.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[ok] Saved: {out_csv}")


if __name__ == "__main__":
    main()

"""
Step 6: Resolve duplicate masks for a single cropped abnormality.

- For cropped images, each row usually corresponds to a single `abnormality_id`.
- We group by (patient_id, left_or_right_breast, image_view, abnormality_id).
- If multiple masks exist for the same crop, we combine them via logical OR to ensure a single resolved mask per crop.
- The resolved mask is saved as PNG in outputs/combined_masks/.
"""
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import OUTPUT_DIR, LOG_DIR

COMBINED_MASK_DIR = OUTPUT_DIR / "combined_masks"

def or_combine_masks(paths: list[str]) -> tuple[np.ndarray, tuple[int, int]]:
    """Open mask jpegs, binarize and combine via OR."""
    combined = None
    target_shape = None
    for p in paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("L"))
        binar = (arr > 127).astype(np.uint8)  # 0/1 binary
        if combined is None:
            combined = binar
            target_shape = binar.shape
        else:
            if binar.shape != target_shape:
                im_r = Image.fromarray(binar * 255).resize(
                    (target_shape[1], target_shape[0]),
                    resample=Image.NEAREST
                )
                binar = (np.asarray(im_r) > 127).astype(np.uint8)
            combined = np.logical_or(combined, binar).astype(np.uint8)
    return combined, target_shape


def main():
    COMBINED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step5_masks_validated.csv"
    df = pd.read_csv(in_path)
    print(f"[in] {in_path.name} shape: {df.shape}")

    # Group by crop specific identifiers
    group_cols = ["patient_id", "left_or_right_breast", "image_view", "abnormality_id"]
    groups = df.groupby(group_cols)
    print(f"[group] total unique crops: {groups.ngroups}")

    out_rows = []
    empty_after_combine = []

    for key, g in tqdm(groups, total=groups.ngroups):
        patient_id, breast, view, abnormality_id = key
        mask_paths = g["selected_mask_path"].tolist()

        combined_mask, shape = or_combine_masks(mask_paths)
        fg_pixels = int(combined_mask.sum())
        fg_ratio = fg_pixels / combined_mask.size

        if fg_pixels < 100:
            empty_after_combine.append({
                "patient_id": patient_id,
                "view": f"{breast}_{view}",
                "abnormality_id": abnormality_id,
                "fg_pixels": fg_pixels,
                "n_masks_combined": len(mask_paths),
            })
            continue

        out_name = f"{patient_id}_{breast}_{view}_{abnormality_id}.png"
        out_path = COMBINED_MASK_DIR / out_name
        Image.fromarray((combined_mask * 255).astype(np.uint8)).save(out_path)

        first = g.iloc[0]
        out_rows.append({
            "patient_id": patient_id,
            "left_or_right_breast": breast,
            "image_view": view,
            "abnormality_id": abnormality_id,
            "pathology": first["pathology"],
            "label": first["label"],
            "pathology_original": first["pathology"],  # Actually it should be pathology_original from step3
            "cropped_image_abs_path": first["cropped_image_abs_path"],
            "combined_mask_abs_path": str(out_path),
            "mask_fg_pixels": fg_pixels,
            "mask_fg_ratio": float(fg_ratio),
            "mask_height": int(shape[0]),
            "mask_width": int(shape[1]),
            "breast_density": first["breast_density"],
            "assessment": first["assessment"],
            "subtlety": first["subtlety"],
            "source_split": first["source_split"],
        })

    out_df = pd.DataFrame(out_rows)

    print(f"\n[report]")
    print(f"  processed crops: {len(out_df)} / {groups.ngroups}")
    print(f"  crops empty after combine: {len(empty_after_combine)}")
    print(f"  pathology distribution:")
    print(out_df["pathology"].value_counts())

    if empty_after_combine:
        log_path = LOG_DIR / "step6_empty_after_combine.csv"
        pd.DataFrame(empty_after_combine).to_csv(log_path, index=False)
        print(f"\n[log] empty after combine -> {log_path}")

    out = OUTPUT_DIR / "step6_masks_combined.csv"
    out_df.to_csv(out, index=False)
    print(f"\n[ok] Saved: {out}")
    print(f"[ok] Combined masks: {COMBINED_MASK_DIR}")

if __name__ == "__main__":
    main()

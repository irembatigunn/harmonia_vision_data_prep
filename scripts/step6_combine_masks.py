"""
Step 6: Ayni (patient_id, left_or_right_breast, image_view) grubundaki
maskeleri mantiksal OR ile birlestir.

- Full mammogram ayni (step 5'te dogrulandi).
- Maske boyutlari grup-ici ayni (step 5 bulgusu) - yine de savunmaci kod.
- Karisik pathology'li gruplarda (8 adet) MALIGNANT precedence:
  bir lezyon malign ise kombinasyon MALIGNANT sayilir.
- pathology_original: tum orijinal etiketler '|' ile ayrilarak saklanir.
- Birlestirilmis maske PNG olarak outputs/combined_masks/ altina kaydedilir.
"""
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import OUTPUT_DIR, LOG_DIR


COMBINED_MASK_DIR = OUTPUT_DIR / "combined_masks"


def or_combine_masks(paths: list[str]) -> tuple[np.ndarray, tuple[int, int]]:
    """Maske jpeg'lerini acip binarize edip OR ile birlestir."""
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
                # Savunmaci: farkli boyutta ise target_shape'e resize
                im_r = Image.fromarray(binar * 255).resize(
                    (target_shape[1], target_shape[0]),
                    resample=Image.NEAREST
                )
                binar = (np.asarray(im_r) > 127).astype(np.uint8)
            combined = np.logical_or(combined, binar).astype(np.uint8)
    return combined, target_shape


def aggregate_labels(group: pd.DataFrame) -> dict:
    """MALIGNANT precedence + orijinal etiketleri koru."""
    pathologies = group["pathology"].tolist()
    pathology = "MALIGNANT" if "MALIGNANT" in pathologies else "BENIGN"
    label = 1 if pathology == "MALIGNANT" else 0
    return {
        "pathology": pathology,
        "label": label,
        "pathology_original_combined": "|".join(group["pathology_original"].astype(str).tolist()),
        "abnormality_ids": "|".join(group["abnormality_id"].astype(str).tolist()),
        "n_abnormalities": len(group),
    }


def main():
    COMBINED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step5_masks_validated.csv"
    df = pd.read_csv(in_path)
    print(f"[in] {in_path.name} shape: {df.shape}")

    group_cols = ["patient_id", "left_or_right_breast", "image_view"]
    groups = df.groupby(group_cols)
    print(f"[group] toplam grup sayisi: {groups.ngroups}")

    out_rows = []
    empty_after_combine = []

    for key, g in tqdm(groups, total=groups.ngroups):
        patient_id, breast, view = key
        mask_paths = g["selected_mask_path"].tolist()

        combined_mask, shape = or_combine_masks(mask_paths)
        fg_pixels = int(combined_mask.sum())
        fg_ratio = fg_pixels / combined_mask.size

        # OR sonrasi bos maske kontrolu (beklenmeyen; ama savunmaci)
        if fg_pixels < 100:
            empty_after_combine.append({
                "patient_id": patient_id,
                "view": f"{breast}_{view}",
                "fg_pixels": fg_pixels,
                "n_masks_combined": len(mask_paths),
            })
            continue  # atla

        # Birlestirilmis maskeyi PNG olarak kaydet (lossless)
        out_name = f"{patient_id}_{breast}_{view}.png"
        out_path = COMBINED_MASK_DIR / out_name
        Image.fromarray((combined_mask * 255).astype(np.uint8)).save(out_path)

        labels = aggregate_labels(g)

        # Gruptan degismeyen sutunlari al
        first = g.iloc[0]
        out_rows.append({
            "patient_id": patient_id,
            "left_or_right_breast": breast,
            "image_view": view,
            **labels,
            "full_image_abs_path": first["full_image_abs_path"],
            "combined_mask_abs_path": str(out_path),
            "mask_fg_pixels": fg_pixels,
            "mask_fg_ratio": float(fg_ratio),
            "mask_height": int(shape[0]),
            "mask_width": int(shape[1]),
            "breast_density": first["breast_density"],
            "assessment": first["assessment"],
            "subtlety": first["subtlety"],
            "collection": first["collection"],
            "source_split": first["source_split"],
        })

    out_df = pd.DataFrame(out_rows)

    print(f"\n[report]")
    print(f"  islenen grup: {len(out_df)} / {groups.ngroups}")
    print(f"  OR sonrasi bos kalan grup: {len(empty_after_combine)}")
    print(f"  birlestirilmis maske sayisi (>=2 maske ile): "
          f"{(out_df['n_abnormalities'] >= 2).sum()}")
    print(f"  pathology dagilimi:")
    print(out_df["pathology"].value_counts())
    print(f"  n_abnormalities dagilimi:")
    print(out_df["n_abnormalities"].value_counts().sort_index())
    print(f"  mask_fg_ratio ozet:")
    print(out_df["mask_fg_ratio"].describe())

    # Log
    if empty_after_combine:
        log_path = LOG_DIR / "step6_empty_after_combine.csv"
        pd.DataFrame(empty_after_combine).to_csv(log_path, index=False)
        print(f"\n[log] bos kalanlar -> {log_path}")

    out = OUTPUT_DIR / "step6_masks_combined.csv"
    out_df.to_csv(out, index=False)
    print(f"\n[ok] Kaydedildi: {out}")
    print(f"[ok] Birlestirilmis maskeler: {COMBINED_MASK_DIR}")


if __name__ == "__main__":
    main()

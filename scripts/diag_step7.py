"""
Diagnostic: Step 7 görsel doğrulama.

8 rastgele örnek seçip her biri için 4'lü görsel oluşturur:
  1. CLAHE uygulanmış görüntü (background removed)
  2. Piksel histogramı
  3. ROI mask
  4. Overlay (CLAHE + ROI mask konturu)

Çıktı: logs/step7_diag/samples.png
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, LOG_DIR, IMAGE_SIZE


DIAG_DIR = LOG_DIR / "step7_diag"


def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "step8_processed.csv"
    if not csv_path.exists():
        print(f"[error] {csv_path} bulunamadi. Once step7'yi calistirin.")
        return

    df = pd.read_csv(csv_path)

    # 8 rastgele örnek (seed ile tekrarlanabilir)
    rng = np.random.RandomState(42)
    n_samples = min(8, len(df))
    indices = rng.choice(len(df), n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row_idx, df_idx in enumerate(indices):
        row = df.iloc[df_idx]

        # İşlenmiş görüntü ve maske
        proc_img = np.asarray(Image.open(row["processed_image_path"]).convert("L"))
        proc_mask = np.asarray(Image.open(row["processed_mask_path"]).convert("L"))

        title = f"{row['patient_id']}_{row['left_or_right_breast']}_{row['image_view']}"
        label = row["pathology"]
        fg_ratio = row.get("mask_fg_ratio", 0)

        # 1. İşlenmiş görüntü (Otsu+CLAHE)
        axes[row_idx, 0].imshow(proc_img, cmap="gray")
        axes[row_idx, 0].set_title(f"CLAHE Image\n{title}", fontsize=9)
        axes[row_idx, 0].axis("off")

        # 2. Histogram
        axes[row_idx, 1].hist(proc_img[proc_img > 0].ravel(), bins=50,
                              color="steelblue", alpha=0.7)
        axes[row_idx, 1].set_title(f"Pixel Histogram\n(non-zero only)", fontsize=9)
        axes[row_idx, 1].set_xlabel("Pixel value")

        # 3. ROI maske
        axes[row_idx, 2].imshow(proc_mask, cmap="gray")
        axes[row_idx, 2].set_title(f"ROI Mask\nfg={fg_ratio:.4f}", fontsize=9)
        axes[row_idx, 2].axis("off")

        # 4. Overlay
        overlay = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2RGB)
        mask_bin = proc_mask > 127
        overlay[mask_bin, 0] = np.minimum(
            overlay[mask_bin, 0].astype(int) + 80, 255).astype(np.uint8)
        overlay[mask_bin, 1] = (overlay[mask_bin, 1] * 0.5).astype(np.uint8)
        overlay[mask_bin, 2] = (overlay[mask_bin, 2] * 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].set_title(f"Overlay ({label})", fontsize=9)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    out_path = DIAG_DIR / "samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] Diagnostik gorsel: {out_path}")


if __name__ == "__main__":
    main()

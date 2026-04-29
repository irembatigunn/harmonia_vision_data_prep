"""256x256 cikti ornekleri gorsellestir."""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from config import OUTPUT_DIR, LOG_DIR

DIAG = LOG_DIR / "step8_diag"
DIAG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "step8_processed.csv")
samples = df.sample(8, random_state=42)

fig, axes = plt.subplots(2, 8, figsize=(24, 6))
for i, (_, row) in enumerate(samples.iterrows()):
    img = np.asarray(Image.open(row["processed_image_path"]))
    mask = np.asarray(Image.open(row["processed_mask_path"]))
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f"{row['patient_id']}\n{row['pathology']}", fontsize=8)
    axes[0, i].axis('off')
    # overlay
    overlay = np.stack([img, img, img], axis=-1).copy()
    overlay[mask > 127] = [255, 0, 0]
    axes[1, i].imshow(overlay)
    axes[1, i].axis('off')
plt.tight_layout()
plt.savefig(DIAG / "samples.png", dpi=80)
print(f"saved: {DIAG/'samples.png'}")

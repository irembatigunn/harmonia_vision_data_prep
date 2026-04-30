"""
Step 5: Mask validation (pixel value analysis).

CBIS-DDSM jpeg masks can sometimes be mislabeled or corrupted. Because JPEG is lossy,
we cannot expect exact 0/255 values; instead we use a bimodal distribution test:
more than 98% of pixels must be in [0..BIN_LOW] or [BIN_HIGH..255] ranges.

We also detect empty masks (very low foreground pixels).
This ensures we only pass valid binary masks to Step 6 for crop-level mask resolution.
"""
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import OUTPUT_DIR, LOG_DIR


BIN_LOW = 30      # 0 civari: [0..30]
BIN_HIGH = 225    # 255 civari: [225..255]
BINARY_RATIO_THR = 0.98   # bimodal kabul esigi
MIN_FG_PIXELS = 100       # bos maske esigi (foreground piksel sayisi)


def analyze_mask(path: str) -> dict:
    """Maske jpeg'ini acip istatistikleri cikarir."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))  # grayscale

    h, w = arr.shape
    total = arr.size

    low = int(np.sum(arr <= BIN_LOW))
    high = int(np.sum(arr >= BIN_HIGH))
    mid = total - low - high
    bimodal_ratio = (low + high) / total

    # Foreground: esikle (> 127) binarize et, piksel say
    fg_pixels = int(np.sum(arr > 127))

    unique_count = int(len(np.unique(arr)))

    return {
        "height": h,
        "width": w,
        "bimodal_ratio": float(bimodal_ratio),
        "fg_pixels": fg_pixels,
        "fg_ratio": float(fg_pixels / total),
        "unique_values": unique_count,
        "is_binary_mask": bool(bimodal_ratio >= BINARY_RATIO_THR),
        "is_empty": bool(fg_pixels < MIN_FG_PIXELS),
    }


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step4_files_verified.csv"
    df = pd.read_csv(in_path)
    print(f"[in] {in_path.name} shape: {df.shape}")

    # valid mask paths'i listeye geri ac
    df["mask_candidates"] = df["valid_mask_paths_str"].fillna("").apply(
        lambda s: [p for p in s.split("|") if p]
    )

    selected_paths = []
    selected_stats = []
    rejected_log = []

    for i, row in tqdm(list(df.iterrows()), total=len(df)):
        candidates = row["mask_candidates"]
        # Her aday icin istatistik topla
        per_candidate = []
        for path in candidates:
            try:
                st = analyze_mask(path)
                st["path"] = path
                per_candidate.append(st)
            except Exception as e:
                rejected_log.append({
                    "patient_id": row["patient_id"],
                    "view": f"{row['left_or_right_breast']}_{row['image_view']}",
                    "abnormality_id": row["abnormality_id"],
                    "path": path,
                    "reason": f"analyze_fail:{type(e).__name__}",
                })

        # Gercek binary maske olanlari sec
        binary_candidates = [c for c in per_candidate if c["is_binary_mask"]]

        if not binary_candidates:
            # Hicbir aday binary degil -> muhtemelen yanlis etiketlenmis (cropped?)
            for c in per_candidate:
                rejected_log.append({
                    "patient_id": row["patient_id"],
                    "view": f"{row['left_or_right_breast']}_{row['image_view']}",
                    "abnormality_id": row["abnormality_id"],
                    "path": c["path"],
                    "reason": (f"not_binary_mask(bimodal={c['bimodal_ratio']:.3f},"
                               f"unique={c['unique_values']})"),
                })
            selected_paths.append(None)
            selected_stats.append(None)
            continue

        # Birden fazla binary aday varsa: en cok fg_pixels olani sec
        # (genelde gercek maske dolu; yanlislikla eslesen duz siyah/tam beyaz
        # bir seyin bimodal_ratio'su yuksek ama fg_ratio'su uc degerde olabilir)
        best = max(binary_candidates, key=lambda c: c["fg_pixels"])
        selected_paths.append(best["path"])
        selected_stats.append(best)

        # Secilmeyen binary adaylari da logla (bilgi amacli)
        for c in binary_candidates:
            if c["path"] != best["path"]:
                rejected_log.append({
                    "patient_id": row["patient_id"],
                    "view": f"{row['left_or_right_breast']}_{row['image_view']}",
                    "abnormality_id": row["abnormality_id"],
                    "path": c["path"],
                    "reason": f"not_selected(lower_fg={c['fg_pixels']})",
                })

    df["selected_mask_path"] = selected_paths
    df["mask_bimodal_ratio"] = [s["bimodal_ratio"] if s else None
                                 for s in selected_stats]
    df["mask_fg_pixels"] = [s["fg_pixels"] if s else None for s in selected_stats]
    df["mask_fg_ratio"] = [s["fg_ratio"] if s else None for s in selected_stats]
    df["mask_is_empty"] = [s["is_empty"] if s else None for s in selected_stats]
    df["mask_height"] = [s["height"] if s else None for s in selected_stats]
    df["mask_width"] = [s["width"] if s else None for s in selected_stats]

    # --- Raporlar ---
    n_no_binary = df["selected_mask_path"].isna().sum()
    n_empty = df["mask_is_empty"].fillna(False).sum()
    print(f"\n[report]")
    print(f"  binary maske bulunamadi : {n_no_binary} / {len(df)}")
    print(f"  secilenlerden bos maske : {int(n_empty)} / {len(df)}")
    print(f"  bimodal_ratio ozet (secilenler):")
    print(df["mask_bimodal_ratio"].describe())
    print(f"  fg_ratio ozet (secilenler):")
    print(df["mask_fg_ratio"].describe())

    # Log kaydet
    log_df = pd.DataFrame(rejected_log)
    log_path = LOG_DIR / "step5_mask_validation.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[log] reddedilen aday kayitlari: {len(log_df)} -> {log_path}")
    if len(log_df):
        print("  reason dagilimi:")
        print(log_df["reason"].apply(lambda s: s.split("(")[0]).value_counts())

    # Satir eleme: binary maskesi olmayan satirlar atilir.
    # Bos maskeler Step 6'da atilacak (oradaki OR-birlestirme sonrasi
    # bazi satirlar yine de dolu maskeyle eslesebilir).
    before = len(df)
    df = df[df["selected_mask_path"].notna()].copy()
    print(f"\n[drop] binary maske bulunmadigi icin atilan: "
          f"{before - len(df)} ({before} -> {len(df)})")

    df = df.drop(columns=["mask_candidates", "valid_mask_paths_str",
                          "roi_mask_candidates", "roi_mask_abs_paths"])

    out = OUTPUT_DIR / "step5_masks_validated.csv"
    df.to_csv(out, index=False)
    print(f"\n[ok] Kaydedildi: {out}")


if __name__ == "__main__":
    main()

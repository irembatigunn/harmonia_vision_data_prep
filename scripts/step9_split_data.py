"""
Step 9: Patient-based 70/15/15 train/val/test split.

- Aynı hasta ASLA birden fazla sete düşmez (data leakage önlemi).
- Patient-level etiket: hastanın herhangi bir görüntüsünde MALIGNANT varsa
  patient 'MALIGNANT' kabul edilir (clinical precedence).
- Split patient bazlı yapılır; stratify = patient_level_label.
- İki aşamalı split: 70% train vs 30% temp → 15/15 val/test.

Girdi : outputs/step8_processed.csv
Çıktı : outputs/step9_splits.csv
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


def patient_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Her hasta için: herhangi bir satırda MALIGNANT varsa MALIGNANT."""
    has_malignant = df.groupby("patient_id")["label"].max().rename("patient_label")
    n_samples = df.groupby("patient_id").size().rename("n_samples_per_patient")
    return pd.concat([has_malignant, n_samples], axis=1).reset_index()


def main():
    in_path = OUTPUT_DIR / "step8_processed.csv"
    df = pd.read_csv(in_path)
    print(f"[in] {in_path.name} shape: {df.shape}")

    # Patient-level etiket
    pdf = patient_level_labels(df)
    print(f"\n[patients]")
    print(f"  toplam hasta: {len(pdf)}")
    print(f"  patient-level dagilim:")
    print(pdf["patient_label"].value_counts())

    # Split patient-bazlı
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

    # 70% train, 30% temp
    train_pids, temp_pids, _, _ = train_test_split(
        pdf["patient_id"], pdf["patient_label"],
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=pdf["patient_label"],
        random_state=RANDOM_SEED,
    )

    # temp → 15/15 val/test
    temp_labels = pdf.set_index("patient_id").loc[temp_pids, "patient_label"]
    val_pids, test_pids, _, _ = train_test_split(
        temp_pids, temp_labels,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    split_map = (
        {pid: "train" for pid in train_pids}
        | {pid: "val" for pid in val_pids}
        | {pid: "test" for pid in test_pids}
    )
    df["split"] = df["patient_id"].map(split_map)

    # --- Doğrulama ---
    check = df.groupby("patient_id")["split"].nunique()
    assert (check == 1).all(), "HATA: Bir hasta birden fazla split'te!"
    print("\n[check] Hasta bazli izolasyon: OK")

    print("\n[sizes]")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        n_patients = sub["patient_id"].nunique()
        print(f"  {s:5s}: {len(sub):4d} sample ({len(sub)/len(df):.1%}) | "
              f"{n_patients:3d} patient ({n_patients/len(pdf):.1%})")

    print("\n[class balance]")
    ct = pd.crosstab(df["split"], df["pathology"], margins=True, normalize=False)
    print(ct)
    print("\n[class balance - normalized]")
    ct_norm = pd.crosstab(df["split"], df["pathology"], normalize="index")
    print(ct_norm.round(3))

    # FG ratio istatistikleri (split bazlı)
    if "mask_fg_ratio" in df.columns:
        print("\n[mask foreground ratio by split]")
        for s in ["train", "val", "test"]:
            sub = df[df["split"] == s]
            fr = sub["mask_fg_ratio"]
            print(f"  {s:5s}: mean={fr.mean():.4f}, median={fr.median():.4f}, "
                  f"min={fr.min():.4f}, max={fr.max():.4f}")

    # Kaydet
    out = OUTPUT_DIR / "step9_splits.csv"
    df.to_csv(out, index=False)
    print(f"\n[ok] Kaydedildi: {out}")


if __name__ == "__main__":
    main()

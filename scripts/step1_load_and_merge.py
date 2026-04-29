"""
Step 1: CBIS-DDSM mass CSV'lerini yukle, sutun isimlerini normalize et,
train/test'i tek bir DataFrame'de birlestir, ilk kesif (EDA).
"""
import pandas as pd
from config import MASS_TRAIN_CSV, MASS_TEST_CSV, OUTPUT_DIR


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def load_mass_data() -> pd.DataFrame:
    train = pd.read_csv(MASS_TRAIN_CSV)
    test = pd.read_csv(MASS_TEST_CSV)

    print(f"[raw] mass_train shape: {train.shape}")
    print(f"[raw] mass_test  shape: {test.shape}")
    print(f"[raw] mass_train columns: {list(train.columns)}")

    train = normalize_columns(train)
    test = normalize_columns(test)

    train["source_split"] = "train"
    test["source_split"] = "test"

    df = pd.concat([train, test], axis=0, ignore_index=True)
    print(f"[merged] combined shape: {df.shape}")
    return df


def eda(df: pd.DataFrame) -> None:
    print("\n=== Normalize edilmis sutunlar ===")
    for c in df.columns:
        print(f"  - {c}")

    print("\n=== Dtypes ===")
    print(df.dtypes)

    print("\n=== Null sayilari (sifirdan buyuk olanlar) ===")
    nulls = df.isna().sum()
    print(nulls[nulls > 0].sort_values(ascending=False))

    print("\n=== Unique degerler (kritik kategorik sutunlar) ===")
    for col in ["pathology", "assessment", "subtlety", "breast_density",
                "left_or_right_breast", "image_view", "mass_shape", "mass_margins"]:
        if col in df.columns:
            vals = df[col].value_counts(dropna=False)
            print(f"\n[{col}] ({df[col].nunique(dropna=False)} unique)")
            print(vals.head(15))

    print("\n=== Hasta sayisi ===")
    if "patient_id" in df.columns:
        print(f"  Toplam satir: {len(df)}")
        print(f"  Unique patient_id: {df['patient_id'].nunique()}")
        print(f"  Unique (patient_id, left_or_right_breast, image_view): "
              f"{df.groupby(['patient_id','left_or_right_breast','image_view']).ngroups}")

    print("\n=== Pathology dagilimi ===")
    print(df["pathology"].value_counts(dropna=False))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_mass_data()
    eda(df)

    out_path = OUTPUT_DIR / "step1_mass_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[ok] Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

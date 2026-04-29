"""
Step 2: dicom_info.csv ve meta.csv ile JOIN, 'cropped images' satirlarini temizle.

Mass CSV path formati:
  Mass-Training_P_00001_LEFT_CC/<StudyUID>/<SeriesUID>/<file>.dcm
Yani path'in ikinci UID'si SeriesInstanceUID. dicom_info ile bu anahtarla JOIN
yapacagiz. Ayni SeriesUID icin dicom_info'da birden fazla kayit olabilir
(maske klasorunde birden fazla jpeg). Step 5'te piksel analizi ile gercek
maske secilecek.
"""
import pandas as pd
from config import DICOM_INFO_CSV, META_CSV, OUTPUT_DIR, JPEG_DIR


def extract_series_uid(path: str) -> str | None:
    """Mass CSV path'inden SeriesInstanceUID (ikinci segment) cikarir."""
    if not isinstance(path, str):
        return None
    parts = path.split("/")
    # Mass-Training_P_xxx_YYY_ZZZ/<StudyUID>/<SeriesUID>/<file>.dcm
    if len(parts) >= 3:
        return parts[2]
    return None


def normalize_dicom_info(di: pd.DataFrame) -> pd.DataFrame:
    """dicom_info kolonlarini snake_case'e cevir (mass_df ile ayni stil)."""
    # Sadece ihtiyacimiz olan kolonlari tut
    keep = ["SeriesInstanceUID", "SeriesDescription", "image_path",
            "file_path", "PatientID", "Laterality", "PatientOrientation",
            "Rows", "Columns"]
    di = di[keep].copy()
    rename_map = {
        "SeriesInstanceUID": "series_instance_uid",
        "SeriesDescription": "series_description",
        "image_path": "jpeg_path",
        "file_path": "dcm_path",
        "PatientID": "dicom_patient_id",
        "Laterality": "dicom_laterality",
        "PatientOrientation": "dicom_patient_orientation",
        "Rows": "dicom_rows",
        "Columns": "dicom_columns",
    }
    return di.rename(columns=rename_map)


def main():
    # --- 1. Mass df yukle (step 1 ciktisi) ---
    mass = pd.read_csv(OUTPUT_DIR / "step1_mass_merged.csv")
    print(f"[mass] baslangic satir sayisi: {len(mass)}")

    # --- 2. dicom_info yukle + normalize + 'cropped images' temizle ---
    di = pd.read_csv(DICOM_INFO_CSV)
    print(f"[dicom_info] raw shape: {di.shape}")
    print(f"[dicom_info] SeriesDescription dagilimi:")
    print(di["SeriesDescription"].value_counts(dropna=False))

    di = normalize_dicom_info(di)

    before = len(di)
    di = di[di["series_description"] != "cropped images"].copy()
    print(f"\n[filter] 'cropped images' atildi: {before} -> {len(di)} "
          f"({before - len(di)} satir silindi)")

    # NaN series_description'lari da at (ne oldugu belirsiz)
    before = len(di)
    di = di.dropna(subset=["series_description"])
    print(f"[filter] NaN series_description atildi: {before} -> {len(di)}")

    print(f"[dicom_info] kalan SeriesDescription dagilimi:")
    print(di["series_description"].value_counts())

    # --- 3. Mass path'lerden SeriesInstanceUID cikar ---
    mass["full_series_uid"] = mass["image_file_path"].apply(extract_series_uid)
    mass["roi_series_uid"] = mass["roi_mask_file_path"].apply(extract_series_uid)

    # --- 4. Full mammogram JOIN ---
    full_df = di[di["series_description"] == "full mammogram images"][
        ["series_instance_uid", "jpeg_path", "dicom_rows", "dicom_columns"]
    ].rename(columns={
        "series_instance_uid": "full_series_uid",
        "jpeg_path": "full_jpeg_path",
        "dicom_rows": "full_rows",
        "dicom_columns": "full_cols",
    })
    # Full mammogram series'te tek jpeg bekleniyor; duplicate varsa ilkini al
    dup = full_df["full_series_uid"].duplicated().sum()
    if dup:
        print(f"[warn] full mammogram series'te {dup} duplicate, ilki tutuldu.")
        full_df = full_df.drop_duplicates(subset="full_series_uid", keep="first")

    mass = mass.merge(full_df, on="full_series_uid", how="left")

    missing_full = mass["full_jpeg_path"].isna().sum()
    print(f"\n[join:full] eslesemeyen full mammogram: {missing_full} / {len(mass)}")

    # --- 5. ROI mask JOIN (birden fazla aday olabilir -> listele) ---
    roi_df = di[di["series_description"] == "ROI mask images"][
        ["series_instance_uid", "jpeg_path"]
    ].rename(columns={
        "series_instance_uid": "roi_series_uid",
        "jpeg_path": "roi_jpeg_path",
    })
    # Her series_uid icin tum maske jpeg'lerini listele (Step 5 icin adaylar)
    roi_agg = (roi_df.groupby("roi_series_uid")["roi_jpeg_path"]
               .apply(list).reset_index()
               .rename(columns={"roi_jpeg_path": "roi_mask_candidates"}))
    roi_agg["n_mask_candidates"] = roi_agg["roi_mask_candidates"].apply(len)

    mass = mass.merge(roi_agg, on="roi_series_uid", how="left")

    missing_roi = mass["roi_mask_candidates"].isna().sum()
    print(f"[join:roi] eslesemeyen ROI mask: {missing_roi} / {len(mass)}")

    # --- 6. Mutlak jpeg path'lere cevir ---
    # dicom_info'daki jpeg_path formati: 'CBIS-DDSM/jpeg/<uid>/x.jpg'
    # Bizdeki yerel yerlesim: <JPEG_DIR>/<uid>/x.jpg
    # Bu yuzden 'CBIS-DDSM/jpeg/' onekini soyup JPEG_DIR ile birlestirecegiz.
    PREFIX = "CBIS-DDSM/jpeg/"

    def to_abs(rel):
        if not isinstance(rel, str):
            return None
        rel = rel[len(PREFIX):] if rel.startswith(PREFIX) else rel
        return str(JPEG_DIR / rel)

    mass["full_image_abs_path"] = mass["full_jpeg_path"].apply(to_abs)

    def to_abs_list(lst):
        if not isinstance(lst, list):
            return None
        return [to_abs(p) for p in lst]

    mass["roi_mask_abs_paths"] = mass["roi_mask_candidates"].apply(to_abs_list)

    # --- 7. meta.csv JOIN (sadece referans bilgi - Collection vs.) ---
    me = pd.read_csv(META_CSV)
    me = me[["SeriesInstanceUID", "Collection"]].rename(
        columns={"SeriesInstanceUID": "full_series_uid",
                 "Collection": "collection"}
    ).drop_duplicates(subset="full_series_uid")
    mass = mass.merge(me, on="full_series_uid", how="left")

    # --- 8. Ozet & kaydet ---
    print("\n=== Ozet ===")
    print(f"  toplam satir: {len(mass)}")
    print(f"  full mammogram eslesen: {mass['full_jpeg_path'].notna().sum()}")
    print(f"  ROI mask eslesen       : {mass['roi_mask_candidates'].notna().sum()}")
    print(f"  n_mask_candidates dagilimi:")
    print(mass["n_mask_candidates"].value_counts(dropna=False).sort_index())

    # Eksik eslesmeleri logla
    miss = mass[mass["full_jpeg_path"].isna() | mass["roi_mask_candidates"].isna()]
    if len(miss) > 0:
        miss_out = OUTPUT_DIR / "step2_missing_joins.csv"
        miss[["patient_id", "left_or_right_breast", "image_view",
              "abnormality_id", "full_series_uid", "roi_series_uid"]].to_csv(
            miss_out, index=False)
        print(f"  eksik JOIN'ler: {len(miss)} -> {miss_out}")

    # Listeleri CSV'ye kaydetmek icin string'e cevir (pipe ile ayirt)
    mass_save = mass.copy()
    mass_save["roi_mask_abs_paths"] = mass_save["roi_mask_abs_paths"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    mass_save["roi_mask_candidates"] = mass_save["roi_mask_candidates"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )

    out = OUTPUT_DIR / "step2_mass_joined.csv"
    mass_save.to_csv(out, index=False)
    print(f"\n[ok] Kaydedildi: {out}")


if __name__ == "__main__":
    main()

"""
Step 2: Join with dicom_info.csv and meta.csv to resolve absolute paths for cropped images and ROI masks.

Mass CSV paths format:
  Mass-Training_P_00001_LEFT_CC_1/<StudyUID>/<SeriesUID>/<file>.dcm
The second UID is the SeriesInstanceUID, used to join with dicom_info.
"""
import pandas as pd
from config import DICOM_INFO_CSV, META_CSV, OUTPUT_DIR, JPEG_DIR


def extract_series_uid(path: str) -> str | None:
    """Extract SeriesInstanceUID from Mass CSV path."""
    if pd.isna(path) or not isinstance(path, str):
        return None
    parts = path.split("/")
    if len(parts) >= 3:
        return parts[2]
    return None


def normalize_dicom_info(di: pd.DataFrame) -> pd.DataFrame:
    """Normalize dicom_info columns to snake_case."""
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
    # --- 1. Load mass df (step 1 output) ---
    mass = pd.read_csv(OUTPUT_DIR / "step1_mass_merged.csv")
    print(f"[mass] starting row count: {len(mass)}")

    # --- 2. Load dicom_info and normalize ---
    di = pd.read_csv(DICOM_INFO_CSV)
    print(f"[dicom_info] raw shape: {di.shape}")
    di = normalize_dicom_info(di)
    
    # Drop rows with NaN series_description
    di = di.dropna(subset=["series_description"])

    # --- 3. Extract SeriesInstanceUID from Mass paths ---
    mass["cropped_series_uid"] = mass["cropped_image_file_path"].apply(extract_series_uid)
    mass["roi_series_uid"] = mass["roi_mask_file_path"].apply(extract_series_uid)

    # --- 4. Join Cropped Images ---
    crop_df = di[di["series_description"] == "cropped images"][
        ["series_instance_uid", "jpeg_path", "dicom_rows", "dicom_columns"]
    ].rename(columns={
        "series_instance_uid": "cropped_series_uid",
        "jpeg_path": "cropped_jpeg_path",
        "dicom_rows": "crop_rows",
        "dicom_columns": "crop_cols",
    })
    
    # For a given crop series, we expect one image. If multiple, keep first.
    dup = crop_df["cropped_series_uid"].duplicated().sum()
    if dup > 0:
        print(f"[warn] {dup} duplicates in cropped image series, keeping first.")
        crop_df = crop_df.drop_duplicates(subset="cropped_series_uid", keep="first")

    mass = mass.merge(crop_df, on="cropped_series_uid", how="left")
    missing_crop = mass["cropped_jpeg_path"].isna().sum()
    print(f"[join:crop] missing cropped image match: {missing_crop} / {len(mass)}")

    # --- 5. Join ROI masks (could be multiple candidates) ---
    roi_df = di[di["series_description"] == "ROI mask images"][
        ["series_instance_uid", "jpeg_path"]
    ].rename(columns={
        "series_instance_uid": "roi_series_uid",
        "jpeg_path": "roi_jpeg_path",
    })
    
    roi_agg = (roi_df.groupby("roi_series_uid")["roi_jpeg_path"]
               .apply(list).reset_index()
               .rename(columns={"roi_jpeg_path": "roi_mask_candidates"}))
    roi_agg["n_mask_candidates"] = roi_agg["roi_mask_candidates"].apply(len)

    mass = mass.merge(roi_agg, on="roi_series_uid", how="left")
    missing_roi = mass["roi_mask_candidates"].isna().sum()
    print(f"[join:roi] missing ROI mask match: {missing_roi} / {len(mass)}")

    # --- 6. Convert to absolute paths ---
    PREFIX = "CBIS-DDSM/jpeg/"

    def to_abs(rel):
        if not isinstance(rel, str):
            return None
        rel = rel[len(PREFIX):] if rel.startswith(PREFIX) else rel
        return str(JPEG_DIR / rel)

    mass["cropped_image_abs_path"] = mass["cropped_jpeg_path"].apply(to_abs)

    def to_abs_list(lst):
        if not isinstance(lst, list):
            return None
        return [to_abs(p) for p in lst]

    mass["roi_mask_abs_paths"] = mass["roi_mask_candidates"].apply(to_abs_list)

    # --- 7. Join meta.csv (optional info) ---
    # We can join on either the original full series or the crop series. 
    # The full_image_file_path is named image_file_path in mass CSV.
    mass["full_series_uid"] = mass["image_file_path"].apply(extract_series_uid)
    me = pd.read_csv(META_CSV)
    me = me[["SeriesInstanceUID", "Collection"]].rename(
        columns={"SeriesInstanceUID": "full_series_uid",
                 "Collection": "collection"}
    ).drop_duplicates(subset="full_series_uid")
    mass = mass.merge(me, on="full_series_uid", how="left")

    # --- 8. Summary & Save ---
    print("\n=== Summary ===")
    print(f"  total rows: {len(mass)}")
    print(f"  cropped matched: {mass['cropped_jpeg_path'].notna().sum()}")
    print(f"  ROI matched    : {mass['roi_mask_candidates'].notna().sum()}")
    
    # Log missing joins
    miss = mass[mass["cropped_jpeg_path"].isna() | mass["roi_mask_candidates"].isna()]
    if len(miss) > 0:
        miss_out = OUTPUT_DIR / "step2_missing_joins.csv"
        miss[["patient_id", "left_or_right_breast", "image_view",
              "abnormality_id", "cropped_series_uid", "roi_series_uid"]].to_csv(
            miss_out, index=False)
        print(f"  missing joins logged: {len(miss)} -> {miss_out}")

    # Format lists as pipe-separated strings for CSV
    mass_save = mass.copy()
    mass_save["roi_mask_abs_paths"] = mass_save["roi_mask_abs_paths"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )
    mass_save["roi_mask_candidates"] = mass_save["roi_mask_candidates"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else ""
    )

    out = OUTPUT_DIR / "step2_mass_joined.csv"
    mass_save.to_csv(out, index=False)
    print(f"\n[ok] Saved: {out}")

if __name__ == "__main__":
    main()

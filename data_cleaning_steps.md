# CBIS-DDSM Veri Temizleme Pipeline'ı — Dokümantasyon & Model Kullanım Rehberi

## Context

Bu belge, CBIS-DDSM mamografi veri setinin BENIGN / MALIGNANT sınıflandırması için hazırlanan 11 aşamalı veri temizleme pipeline'ını baştan sona belgeler. Amaç: (1) her adımın **ne yaptığını ve neden yaptığını** kayıt altına almak, (2) her dosyanın içeriğini açıklamak, (3) nihai `.npy` çıktılarının **model eğitiminde nasıl kullanılacağını** göstermek.

Ana yapı, eskiden kullanılan full-mammogram mantığından **cropped lesion-centered mammogram** mantığına geçirilmiştir. Artık ağ, tüm memeyi taramak yerine doğrudan lezyona odaklanan cropped görüntüleri alıp segmentasyon yapacaktır.

---

## 1. Proje Dizin Yapısı

```
/Users/berilcitil/Desktop/data_cleaning/
├── CBIS DDSM Breast Cancer Dataset/        # Ham veri (dokunulmadı)
│   ├── csv/                                 # 6 CSV: mass_*, calc_*, dicom_info, meta
│   └── jpeg/                                # 6775 UID klasörü, her birinde jpeg(ler)
├── scripts/                                 # Pipeline kodu (tüm step'ler)
│   ├── config.py
│   ├── step1_load_and_merge.py
│   ├── step2_join_metadata.py
│   ├── step3_consolidate_labels.py
│   ├── step4_file_integrity.py
│   ├── step5_mask_validation.py
│   ├── step6_combine_masks.py
│   ├── step7_crop_mask_validation.py
│   ├── step8_crop_resize.py
│   ├── step9_split_data.py
│   ├── step10_save_npy.py
│   ├── step11_create_fl_partitions.py
├── outputs/                                 # Tüm ara ve nihai çıktılar
│   ├── step1_mass_merged.csv  …  step9_splits.csv   # Her step'in CSV çıktısı
│   ├── processed/
│   │   ├── images/                          # PNG 256×256 (Step 8 çıktısı)
│   │   └── masks/                           # PNG 256×256 (Step 8 çıktısı)
│   └── final_npy/                           # ⭐ MODELDE KULLANACAĞIN KLASÖR
│       ├── X_train.npy, X_val.npy, X_test.npy
│       ├── masks_train.npy, masks_val.npy, masks_test.npy
│       ├── y_train.npy, y_val.npy, y_test.npy
│       ├── metadata_train.csv, metadata_val.csv, metadata_test.csv
│       ├── pipeline_summary.json
│       └── dataset_info.txt
```

---

## 2. Her Step'in Açıklaması

### `config.py` — Sabitler
Tüm scriptlerin paylaştığı yollar ve parametreler: `IMAGE_SIZE=256`, `RANDOM_SEED=42`, split oranları (70/15/15). 

### Step 1 — `step1_load_and_merge.py`
**Ne yapar:** `mass_case_description_train_set.csv` + `mass_case_description_test_set.csv`'yi tek DataFrame'e birleştirir. Sütun isimlerini `snake_case`'e çevirir.

### Step 2 — `step2_join_metadata.py`
**Ne yapar:** `dicom_info.csv` ile JOIN yapar. Ana input olarak artık "cropped images" dosyalarının yolunu (`cropped_image_abs_path`) alır.
**Neden:** Full mammogram yerine doğrudan crop'lar hedeflendiği için dicom_info üzerindeki cropped image yolları kullanılır.

### Step 3 — `step3_consolidate_labels.py`
**Ne yapar:** `BENIGN_WITHOUT_CALLBACK` → `BENIGN`. Orijinal etiket `pathology_original` sütununda saklanır. Binary `label` sütunu eklenir (0=BENIGN, 1=MALIGNANT).

### Step 4 — `step4_file_integrity.py`
**Ne yapar:** Her crop ve ROI maskesi için `os.path.exists()` kontrolü ve okuma doğrulaması. Bozuk dosyalar atılır.

### Step 5 — `step5_mask_validation.py`
**Ne yapar:** Bimodal test. Piksellerin ≥%98'i [0..30] veya [225..255] olmalı. Boş maske tespit edilir.

### Step 6 — `step6_combine_masks.py`
**Ne yapar:** `(patient_id, left_or_right_breast, image_view, abnormality_id)` için maske çözücü (resolver). Bir crop için birden fazla ROI adayı varsa doğru olanı seçer/birleştirir.

### Step 7 — `step7_crop_mask_validation.py`
**Ne yapar:** Crop ve ROI maskesi için spatial validation uygular.
- Çözünürlük uyumu kontrolü.
- Boş maske kontrolü (foreground).
- Mask area ratio (maske çok büyük/çok küçük mü).
- Edge-touching (maske kenarlara sıfır mı dayanmış).
- Genel misalignment.

### Step 8 — `step8_crop_resize.py`
**Ne yapar:** Sadece aspect ratio'yu bozmadan square pad uygular ve `256×256` resize yapar. Image için `INTER_AREA`, maske için `INTER_NEAREST` kullanılır.

### Step 9 — `step9_split_data.py`
**Ne yapar:** Patient-level stratified split (70/15/15). Seed = 42. Hasta izolasyonu veri sızıntısını önler.

### Step 10 — `step10_save_npy.py`
**Ne yapar:** `.npy` formatına dönüştürür. `metadata_*.csv` içine `abnormality_id`, `patient_id` ve `pathology_clean` gibi bilgileri gömer.

### Step 11 — `step11_create_fl_partitions.py`
**Ne yapar:** Federated Learning (FL) için client partition mapping oluşturur. IID (Independent and Identically Distributed) ve Non-IID dağıtımları oluşturur.

---

## 3. Alınan Kritik Kararlar

1. **Cropped Image Yaklaşımı:** Modeli tüm görüntüyü aramaktan kurtarıp sadece lezyonu anlamaya odaklar.
2. **Kayıplı JPEG Toleransı:** Maskeler tam 0/255 olmadığı için bimodal eşik kullanılmıştır.
3. **Square Padding:** Aspect ratio korunduğu için lezyonların morfolojik özellikleri bozulmaz.

---

## 4. Pipeline'ı Çalıştırma

```bash
cd scripts
python step1_load_and_merge.py
python step2_join_metadata.py
# ... tüm stepleri sırayla çalıştır
python step11_create_fl_partitions.py
```
Her step idempotent'tir.

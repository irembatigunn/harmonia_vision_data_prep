# CBIS-DDSM Veri Temizleme Pipeline'ı — Dokümantasyon & Model Kullanım Rehberi

## Context

Bu belge, CBIS-DDSM mamografi veri setinin BENIGN / MALIGNANT sınıflandırması için hazırlanan 10 aşamalı veri temizleme pipeline'ını baştan sona belgeler. Amaç: (1) her adımın **ne yaptığını ve neden yaptığını** kayıt altına almak, (2) her dosyanın içeriğini açıklamak, (3) nihai `.npy` çıktılarının **model eğitiminde nasıl kullanılacağını** göstermek.

Başlangıç: 1696 ham satır (1318 mass_train + 378 mass_test CSV) → Sonuç: **1582 işlenmiş 256×256 mamogram + binary maske**, 886 hasta, patient-based 70/15/15 train/val/test split ile veri sızıntısına karşı korumalı.

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
│   ├── step7_breast_extraction.py
│   ├── step8_crop_resize.py
│   ├── step9_split_data.py
│   ├── step10_save_npy.py
│   ├── diag_step7.py                        # Görsel doğrulama
│   └── diag_step8.py                        # Görsel doğrulama
├── outputs/                                 # Tüm ara ve nihai çıktılar
│   ├── step1_mass_merged.csv  …  step9_splits.csv   # Her step'in CSV çıktısı
│   ├── combined_masks/                      # 1592 PNG (Step 6 çıktısı)
│   ├── processed/
│   │   ├── images/                          # 1582 PNG 256×256 (Step 8 çıktısı)
│   │   └── masks/                           # 1582 PNG 256×256 (Step 8 çıktısı)
│   └── final_npy/                           # ⭐ MODELDE KULLANACAĞIN KLASÖR
│       ├── X_train.npy, X_val.npy, X_test.npy
│       ├── masks_train.npy, masks_val.npy, masks_test.npy
│       ├── y_train.npy, y_val.npy, y_test.npy
│       ├── metadata_train.csv, metadata_val.csv, metadata_test.csv
│       ├── pipeline_summary.json
│       └── dataset_info.txt
└── logs/                                    # Atılan dosyalar, uyarılar, görseller
    ├── step4_file_integrity.csv             # (boş — hiç bozuk dosya yok)
    ├── step5_mask_validation.csv            # (boş — tüm maskeler binary)
    ├── step7_breast_extraction.csv          # 10 kritik hizalama hatası (atıldı)
    ├── step7_diag/                          # Problemli bbox görselleri
    └── step8_diag/                          # 256×256 çıktı örnek görselleri
```

---

## 2. Her Step'in Açıklaması

### `config.py` — Sabitler
Tüm scriptlerin paylaştığı yollar ve parametreler: `IMAGE_SIZE=256`, `RANDOM_SEED=42`, split oranları (70/15/15). Tek bir yerden değiştirilebilir.

### Step 1 — [step1_load_and_merge.py](scripts/step1_load_and_merge.py)
**Ne yapar:** `mass_case_description_train_set.csv` + `mass_case_description_test_set.csv`'yi tek DataFrame'e birleştirir. Sütun isimlerini `snake_case`'e çevirir (örn: `left or right breast` → `left_or_right_breast`). `calc_*.csv` dosyaları hariç tutulur.
**Neden:** CBIS-DDSM'in kendi train/test split'i **kullanılmıyor** (Step 9'da patient-based split kendimiz yapılıyor). Sütun normalleştirmesi sonraki step'lerde sütun erişimini standartlaştırır.
**Çıktı:** `outputs/step1_mass_merged.csv` (1696 satır, 15 sütun) — `source_split` sütunu orijinal CSV'yi hatırlatır.

### Step 2 — [step2_join_metadata.py](scripts/step2_join_metadata.py)
**Ne yapar:** `dicom_info.csv` ve `meta.csv` ile JOIN. Mass CSV'deki path'lerden `SeriesInstanceUID` çıkarılıp dicom_info ile eşleştirilir. `SeriesDescription == "cropped images"` olan 3567 satır dicom_info'dan atılır, 566 NaN satır da temizlenir. Her mass satırına `full_image_abs_path` ve `roi_mask_abs_paths` (mutlak jpeg yolları) eklenir.
**Neden:** Mass CSV'deki path'ler `.dcm` uzantılı; gerçekte jpeg'e ihtiyaç var. Dicom_info bu eşleştirmeyi sağlıyor. "Cropped images" satırları kullanılmıyor (user spec).
**Çıktı:** `outputs/step2_mass_joined.csv` (1696 satır) — **%100 JOIN oranı**.

### Step 3 — [step3_consolidate_labels.py](scripts/step3_consolidate_labels.py)
**Ne yapar:** `BENIGN_WITHOUT_CALLBACK` (141 satır) → `BENIGN`. Orijinal etiket `pathology_original` sütununda saklanır. Binary `label` sütunu eklenir (0=BENIGN, 1=MALIGNANT).
**Neden:** BENIGN_WITHOUT_CALLBACK klinik olarak BENIGN'dir ama ileride detaylı analiz için orijinal etiket saklanmalı.
**Çıktı:** `outputs/step3_labels_consolidated.csv` — BENIGN 912, MALIGNANT 784.

### Step 4 — [step4_file_integrity.py](scripts/step4_file_integrity.py)
**Ne yapar:** Her full mammogram ve ROI mask adayı için `os.path.exists()` kontrolü + PIL ile açmayı dener (corrupt/empty tespiti).
**Neden:** Sonraki step'lerde okunamayan dosya ile karşılaşmak pipeline'ı kırar; önden tespit edilmeli. Kullanıcının planındaki kritik gereksinimlerden biri.
**Çıktı:** `outputs/step4_files_verified.csv` — **0 eksik, 0 corrupt**. Log: `logs/step4_file_integrity.csv` (boş).

### Step 5 — [step5_mask_validation.py](scripts/step5_mask_validation.py)
**Ne yapar:** Her maske adayını açıp piksel dağılımını analiz eder. **Bimodal test**: piksellerin ≥%98'i [0..30] veya [225..255] aralığında olmalı (JPEG kayıplılığı nedeniyle tam 0/255 değil). Ayrıca boş maske tespiti (<100 foreground piksel).
**Neden:** CBIS-DDSM'de `SeriesDescription` etiketleri bazen yanlış — "ROI mask" etiketli bir dosya aslında cropped görüntü olabilir. Piksel analizi gerçek binary maskeyi doğrular.
**Çıktı:** `outputs/step5_masks_validated.csv` — **1696/1696 binary maske doğrulandı** (bimodal_ratio = 1.0 tamı tamına), 0 boş maske.

### Step 6 — [step6_combine_masks.py](scripts/step6_combine_masks.py)
**Ne yapar:** `(patient_id, left_or_right_breast, image_view)` gruplarına göre gruplanır. Aynı meme/açı için birden fazla abnormality (maske) varsa hepsi **logical OR** ile tek maskeye birleştirilir. **MALIGNANT precedence:** karışık sınıflı gruplarda (8 grup vardı) biri malign ise grup malign sayılır. Birleşik maskeler PNG olarak kaydedilir.
**Neden:** Bir mamografide birden fazla lezyon aynı görüntüde görünebiliyor — ayrı ayrı sample olarak kullanmak yanlış; fiziksel olarak tek bir görüntüdeki tüm lezyonlar birleşik maskeye dahil olmalı. Klinik olarak malign bir lezyon varsa görüntü "malign" olarak sınıflandırılır.
**Çıktı:** `outputs/step6_masks_combined.csv` (**1696 → 1592 satır**, 104 satır birleşti). Birleşik maskeler: `outputs/combined_masks/<patient>_<breast>_<view>.png`.

### Step 7 — [step7_breast_extraction.py](scripts/step7_breast_extraction.py)
**Ne yapar:** Her full mammogram için:
1. Median blur + Otsu threshold
2. **30px kenar temizliği** (scanner marker/etiket gürültüsü)
3. Morfolojik opening (ellipse kernel 15×15)
4. En büyük connected component = meme bölgesi
5. Bounding box hesaplanır
6. **Mask-bbox union:** Eğer maske bbox'a taşıyorsa, bbox genişletilir ki maske tamamen içinde kalsın
7. **Kritik hizalama tespiti:** Maske'nin %50'sinden azı bbox içindeyse CBIS-DDSM veri hatası (maske başka bir görüntüye ait) → satır atılır
**Neden:** Mamogramlarda siyah arkaplan + scanner etiketleri var. Basit bir Otsu + largest CC yetersiz kalıyor (P_00001 örneğinde bbox tüm görüntüyü kaplıyordu). Kenar temizliği ve union stratejisi bunu çözer.
**Çıktı:** `outputs/step7_breast_bbox.csv` (**1592 → 1582 satır**, 10 hizalama hatası atıldı). Log: `logs/step7_breast_extraction.csv`.

### Step 8 — [step8_crop_resize.py](scripts/step8_crop_resize.py)
**Ne yapar:** Her satır için:
1. Bbox ile crop
2. **Square pad** (en-boy oranı korunur, dar kenara siyah padding)
3. 256×256 resize — görüntü için `INTER_AREA`, maske için `INTER_NEAREST`
4. Maske kesin binarize edilir (0/255)
**Neden:** Aspect ratio korunmazsa mamogramlar yatay yönde stretch olur (model için bozuk sinyal). Square pad + resize standart bir yaklaşım.
**Çıktı:** `outputs/step8_processed.csv`, `outputs/processed/images/` (1582 PNG), `outputs/processed/masks/` (1582 PNG).

### Step 9 — [step9_split_data.py](scripts/step9_split_data.py)
**Ne yapar:** **Patient-level stratified split:** her hasta için patient-level etiket (malign sample varsa patient=MALIGNANT). `sklearn.train_test_split` iki aşamada: %70 train vs %30 temp → temp'i 15/15 val/test. **Seed = 42** (tekrarlanabilir).
**Neden:** Aynı hastanın farklı açıları aynı sette olmazsa **data leakage** olur (model hastayı tanır, lezyonu değil). Patient-level stratify ile hem izolasyon hem sınıf dengesi sağlanır.
**Çıktı:** `outputs/step9_splits.csv` — `split` sütunu eklendi. Test edildi: hiçbir hasta birden fazla split'te yok. Sınıf dengesi tüm setlerde ~53/47 korundu.

### Step 10 — [step10_save_npy.py](scripts/step10_save_npy.py)
**Ne yapar:** Her split için PNG'leri tek numpy array'e stack eder ve kaydeder. Metadata CSV'leri de split bazlı ayrılır.
**Neden:** `.npy` formatı tek dosyadan hızlı yükleme sağlar, training sırasında 1582 PNG'yi ayrı ayrı okumaktan çok daha hızlı.
**Çıktı:** `outputs/final_npy/` klasörü — **model eğitimi için tek ihtiyaç duyacağın klasör**.

---

## 3. Nihai Çıktılar — Model Eğitimi İçin

### Dosya Yapısı (`outputs/final_npy/`)

| Dosya | Shape | Dtype | Açıklama |
|-------|-------|-------|----------|
| `X_train.npy` | (1102, 256, 256, 1) | uint8 (0-255) | Grayscale mamografi görüntüleri |
| `X_val.npy` | (237, 256, 256, 1) | uint8 | |
| `X_test.npy` | (243, 256, 256, 1) | uint8 | |
| `masks_train.npy` | (1102, 256, 256, 1) | uint8 (0/255) | Binary ROI maskeleri |
| `masks_val.npy` | (237, 256, 256, 1) | uint8 | |
| `masks_test.npy` | (243, 256, 256, 1) | uint8 | |
| `y_train.npy` | (1102,) | uint8 (0/1) | 0=BENIGN, 1=MALIGNANT |
| `y_val.npy` | (237,) | uint8 | |
| `y_test.npy` | (243,) | uint8 | |
| `metadata_*.csv` | — | — | patient_id, view, orijinal etiketler, density, assessment vb. |
| `pipeline_summary.json` | — | — | Tüm istatistikler (programatik erişim için) |
| `dataset_info.txt` | — | — | İnsan okunur özet |

### Sınıf Dağılımı (patient-based stratified)

| Split | N sample | N patient | BENIGN | MALIGNANT |
|-------|----------|-----------|--------|-----------|
| train | 1102 | 620 | 581 (%52.7) | 521 (%47.3) |
| val   |  237 | 133 | 128 (%54.0) | 109 (%46.0) |
| test  |  243 | 133 | 130 (%53.5) | 113 (%46.5) |

### Model Kullanım Kodu (PyTorch örneği)

```python
import numpy as np

# Yükle
X_train = np.load("outputs/final_npy/X_train.npy")   # (1102, 256, 256, 1) uint8
y_train = np.load("outputs/final_npy/y_train.npy")   # (1102,) uint8
masks_train = np.load("outputs/final_npy/masks_train.npy")  # binary maskeler

# Normalleştirme (training sırasında yapılır, pipeline'da yapılmadı)
X_train_f = X_train.astype(np.float32) / 255.0        # [0,1]
masks_train_b = (masks_train > 127).astype(np.float32)  # [0,1] binary

# Sınıflandırma için: X + y kullan
# Segmentasyon için: X + masks kullan
# Multi-task için: X + masks + y
```

### Keras/TensorFlow örneği

```python
import numpy as np
import tensorflow as tf

X = np.load("outputs/final_npy/X_train.npy").astype("float32") / 255.0
y = np.load("outputs/final_npy/y_train.npy")

ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(1024).batch(32)
```

### Önemli Notlar

1. **Kanal düzeni:** `(N, H, W, 1)` — TensorFlow/Keras default (channels-last). PyTorch için `np.transpose(X, (0, 3, 1, 2))` ile `(N, 1, H, W)` yap.
2. **Normalleştirme pipeline'da yapılmadı:** uint8 olarak [0-255] aralığında. Training sırasında `/255.0` (ya da CLAHE/Z-score) uygula.
3. **Grayscale → 3 kanal** gerekirse (ImageNet pretrained ağlar için): `np.repeat(X, 3, axis=-1)`.
4. **metadata_*.csv**: ileri analizler için — subtlety, breast_density, pathology_original (BENIGN_WITHOUT_CALLBACK) gibi sütunlarda ek bilgi var.
5. **Augmentation:** Yatay flip (LEFT/RIGHT simetrisi), hafif rotasyon, elastic deformation tavsiye edilir. Maskeyi de beraber transform etmeyi unutma.

---

## 4. Alınan Kritik Kararlar (Bilinmesi Gerekenler)

1. **Sadece mass_*.csv kullanıldı** — calc_*.csv pipeline'a dahil edilmedi (user spec).
2. **BENIGN_WITHOUT_CALLBACK → BENIGN** birleştirildi. Orijinal: `metadata_*.csv`'de `pathology_original_combined` sütunu.
3. **MALIGNANT precedence**: Hem Step 6'da (aynı memede karışık sınıf) hem Step 9'da (patient-level etiket) "herhangi bir malign varsa malign" kuralı.
4. **JPEG kayıplılığına toleranslı maske doğrulama:** tam 0/255 değil, bimodal ratio ≥0.98.
5. **Patient-based stratified split** — aynı hasta aynı splitte, sınıf dengesi korundu.
6. **10 satır atıldı** (Step 7'de kritik maske-görüntü hizalama hatası) — toplam kayıp %0.6.
7. **Aspect ratio korundu** — square pad + resize (stretch yok).
8. **Normalleştirme yapılmadı** — model eğitimi sırasında ihtiyaca göre (ör: CLAHE ardından [0,1]) uygulanabilir.

---

## 5. Pipeline'ı Yeniden Çalıştırmak İstersen

```bash
cd /Users/berilcitil/Desktop/data_cleaning/scripts
python3 step1_load_and_merge.py
python3 step2_join_metadata.py
python3 step3_consolidate_labels.py
python3 step4_file_integrity.py
python3 step5_mask_validation.py
python3 step6_combine_masks.py
python3 step7_breast_extraction.py
python3 step8_crop_resize.py
python3 step9_split_data.py
python3 step10_save_npy.py
```

Her step **idempotent** — kendi çıktısını üretip sonraki step'in girdisi oluyor.

---

## 6. Verification / Sanity Checks

Nihai çıktıların doğru olduğunu doğrulamak için:

```python
import numpy as np, pandas as pd, json

# 1. Shape & dtype kontrolü
X = np.load("outputs/final_npy/X_train.npy")
assert X.shape == (1102, 256, 256, 1) and X.dtype == np.uint8

# 2. Değer aralığı
assert X.min() >= 0 and X.max() <= 255

# 3. Maske binary mi?
M = np.load("outputs/final_npy/masks_train.npy")
assert set(np.unique(M).tolist()) == {0, 255}

# 4. y etiketleri binary mi?
y = np.load("outputs/final_npy/y_train.npy")
assert set(np.unique(y).tolist()) == {0, 1}

# 5. Patient-based split doğrulaması
tr = pd.read_csv("outputs/final_npy/metadata_train.csv")
va = pd.read_csv("outputs/final_npy/metadata_val.csv")
te = pd.read_csv("outputs/final_npy/metadata_test.csv")
assert len(set(tr.patient_id) & set(va.patient_id)) == 0
assert len(set(tr.patient_id) & set(te.patient_id)) == 0
assert len(set(va.patient_id) & set(te.patient_id)) == 0

# 6. pipeline_summary.json
summary = json.load(open("outputs/final_npy/pipeline_summary.json"))
print(f"Toplam: {summary['total_samples']} sample, {summary['total_patients']} patient")
```

Ayrıca görsel doğrulama: `logs/step8_diag/samples.png` — 8 adet örnek (üst: görüntü, alt: maske overlay).

---

## 7. Bu Bir Dokümantasyon — Uygulama Değil

Bu plan dosyası mevcut (tamamlanmış) pipeline'ı belgeler. Herhangi bir kod değişikliği yapılmaz. Eğer eklemek istediğin bir adım (ör: CLAHE normalleştirme, farklı çözünürlük, veri augmentation preview) varsa ayrı bir plan hazırlayabiliriz.

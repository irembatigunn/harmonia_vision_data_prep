from pathlib import Path
import platform

# --- Platform-adaptive root ---
# Beril'in Mac'indeki orijinal yol (step1-6 ciktilari bu yollarla kaydedildi)
MAC_ROOT = Path("/Users/berilcitil/Desktop/data_cleaning")

# Windows uyumlu yol
WIN_ROOT = Path(r"C:\Users\LENOVO\Desktop\data_cleaning")

# Otomatik secim
if platform.system() == "Windows":
    ROOT = WIN_ROOT
else:
    ROOT = MAC_ROOT

DATASET_DIR = ROOT / "CBIS DDSM Breast Cancer Dataset"
CSV_DIR = DATASET_DIR / "csv"
JPEG_DIR = DATASET_DIR / "jpeg"

OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

MASS_TRAIN_CSV = CSV_DIR / "mass_case_description_train_set.csv"
MASS_TEST_CSV = CSV_DIR / "mass_case_description_test_set.csv"
DICOM_INFO_CSV = CSV_DIR / "dicom_info.csv"
META_CSV = CSV_DIR / "meta.csv"

IMAGE_SIZE = 256
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Mac yollarini Windows yoluna ceviren yardimci fonksiyon
MAC_ROOT_STR = "/Users/berilcitil/Desktop/data_cleaning"

def fix_path(mac_path_str: str) -> str:
    """Step 1-6 CSV'lerdeki Mac yollarini mevcut platforma cevir."""
    if platform.system() != "Windows":
        return mac_path_str
    return mac_path_str.replace(MAC_ROOT_STR, str(WIN_ROOT)).replace("/", "\\")

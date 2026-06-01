"""
config.py — Cau hinh trung tam cho he thong du bao muc nuoc ho Nui Coc
=======================================================================
Tap trung tat ca hang so, duong dan, sieu tham so cua toan bo pipeline
de de dang chinh sua va quan ly.

Su dung:
    from config import FORECAST_DAYS, FEATURE_COLS, MODEL_DIR
"""

import os

# ============================================================
# DUONG DAN
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# Du lieu dau vao
NASA_POWER_PATH = os.path.join(DATA_DIR, "raw", "nasa_power_hourly.csv")
GEE_PATH        = os.path.join(DATA_DIR, "raw", "gee_water_level.csv")

# Du lieu da xu ly
# dataset_train.csv : 2017-2022  — tap huan luyen
# dataset_val.csv   : 2023       — tap EarlyStopping (khong bao cao)
# dataset_test.csv  : 2024+      — tap kiem dinh cuoi (bao cao luan van)
TRAIN_PATH = os.path.join(DATA_DIR, "final", "dataset_train.csv")
VAL_PATH   = os.path.join(DATA_DIR, "final", "dataset_val.csv")
TEST_PATH  = os.path.join(DATA_DIR, "final", "dataset_test.csv")
FULL_PATH  = os.path.join(DATA_DIR, "final", "dataset_full.csv")

# Mo hinh & Scaler
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler_daily.pkl")


# ============================================================
# PHAN CHIA THOI GIAN (Train / Val / Test)
# Quy chuan nghien cuu thuy loi + hoc sau (LSTM/Bi-LSTM):
# ============================================================
#
#  TRAIN  (Hieu chinh - Calibration):
#    2017-01-01 → 2022-12-31  (~6 nam, ~2000 ngay)
#    Muc dich: Fit toan bo tham so Bi-LSTM
#
#  VAL    (Kiem tra noi bo - Internal Validation):
#    2023-01-01 → 2023-12-31  (~365 ngay)
#    Muc dich: EarlyStopping + lua chon sieu tham so (hoc sieu tham so)
#    !! TUYET DOI khong dung de chon mo hinh cuoi / bao cao ket qua !!
#
#  TEST   (Kiem dinh doc lap - Independent Validation):
#    2024-01-01 → hien tai     (~700 ngay, bao gom lu Yagi 9/2024)
#    Muc dich: Danh gia cuoi, bao cao RMSE/MAE/NSE trong luan van
#    !! Du lieu nay KHONG DUOC dung trong bat ky buoc train/val nao !!
#
TRAIN_END = "2022-12-31"   # Ket thuc tap Train
VAL_END   = "2023-12-31"   # Ket thuc tap Val (EarlyStopping)
                            # Test bat dau tu 2024-01-01 den hien tai


# ============================================================
# CHAN TROI DU BAO
# ============================================================
FORECAST_DAYS = [1, 3, 7, 14, 30]  # Du bao t+1d, t+3d, t+7d, t+14d, t+30d


# ============================================================
# BO DAC TRUNG DAU VAO (21 features — v3.0 Daily)
# ============================================================
FEATURE_COLS = [
    # --- Khi tuong ngay ---
    "rain_1d",            # Luong mua ngay (mm)
    "rain_3d",            # Mua tich luy 3 ngay (mm)
    "rain_7d",            # Mua tich luy 7 ngay (mm)
    "rain_14d",           # Mua tich luy 14 ngay (mm)
    "temperature",        # Nhiet do trung binh ngay (°C)
    "humidity",           # Do am trung binh ngay (%)
    # --- Lag muc nuoc ---
    "water_level_lag1",   # Muc nuoc tre 1 ngay (m)
    "water_level_lag3",   # Muc nuoc tre 3 ngay (m)
    "water_level_lag7",   # Muc nuoc tre 7 ngay (m)
    "water_level_lag14",  # Muc nuoc tre 14 ngay (m)
    "water_level_lag30",  # Muc nuoc tre 30 ngay (m)
    # --- Rolling statistics ---
    "water_level_roll7",  # Trung binh truot muc nuoc 7 ngay (m)
    "water_level_roll30", # Trung binh truot muc nuoc 30 ngay (m)
    "water_level_std7",   # Do lech chuan muc nuoc 7 ngay (m)
    # --- Temporal encoding (tuan hoan) ---
    "month_sin",          # Ma hoa tuan hoan thang (sin)
    "month_cos",          # Ma hoa tuan hoan thang (cos)
    "season_wet",         # Mua mua: thang 5-10 (1/0)
    "season_dry",         # Mua kho: thang 11-4 (1/0)
    # --- Q_out daily (phuong trinh can bang nuoc) ---
    "dH_dt_daily",        # Toc do thay doi muc nuoc (m/ngay)
    "Q_out_daily",        # Luu luong xa uoc tinh ngay (m³/s)
    "Q_out_roll7",        # Trung binh truot luu luong xa 7 ngay (m³/s)
]

FEATURE_COUNT = len(FEATURE_COLS)  # 21
TARGET_COL    = "water_level_m"


# ============================================================
# CUA SO THOI GIAN
# ============================================================
WINDOW_SIZE = 30  # Nhin lai 30 ngay (1 thang) de du bao


# ============================================================
# SIEU THAM SO MO HINH BI-LSTM + ATTENTION
# ============================================================
LSTM_UNITS    = [128, 64]   # So unit cua BiLSTM lop 1 va 2
DROPOUT_RATE  = 0.2         # Ti le Dropout (dung ca MC Dropout inference)
LEARNING_RATE = 0.001       # Adam optimizer learning rate
BATCH_SIZE    = 32
MAX_EPOCHS    = 200
PATIENCE      = 15          # EarlyStopping patience
ATTN_HEADS    = 4           # So dau Multi-Head Attention
ATTN_KEY_DIM  = 32          # Chieu key/query cua Attention
MC_SAMPLES    = 50          # So lan chay Monte Carlo Dropout


# ============================================================
# NGUONG CANH BAO LU HO NUI COC (don vi: m)
# Nguon: Quy trinh van hanh ho chua nuoc Nui Coc — Bo NN&PTNT
# ============================================================
MNDBT    = 46.20  # Muc nuoc dang binh thuong (m)
CANH_BAO = 46.80  # Nguong canh bao — theo doi lien tuc, san sang xa
NGUY_HIEM = 47.40 # Nguong nguy hiem — mo toan bo cua xa, so tan ha luu


# ============================================================
# THONG SO DUONG CONG A-H (Dien tich — Muc nuoc)
# Don vi: (ha, m) — tuong ung voi (dien tich mat ho, muc nuoc)
# Nguon: Quy trinh van hanh ho Nui Coc
# ============================================================
AH_CURVE = [
    (  50, 34.00), (150, 35.50), (200, 36.00),
    ( 500, 38.00), (900, 40.00), (1400, 42.00),
    (2000, 44.00), (2500, 46.20), (2700, 46.50),
    (2900, 46.90), (3050, 47.20), (3150, 47.50),
    (3200, 47.80), (3500, 48.25),
]


# ============================================================
# AUGMENTATION DU LIEU
# ============================================================
AUG_START           = "2017-01-01"  # Bat dau tu khi Sentinel-2A phong
AUG_END_EXCLUSIVE   = "2019-04-01"  # Diem GEE dau tien thuc su
MAX_INTERP_GAP_DAYS = 60            # Khong noi suy qua 60 ngay lien tuc


# ============================================================
# API SERVER
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "He thong du bao muc nuoc ho Nui Coc"
API_VERSION = "3.0"

"""
config.py — Cấu hình trung tâm (v5.1 — Bi-LSTM + chống overfitting)
====================================================================
Thay đổi v5.1:
  - Bật USE_BIDIRECTIONAL = True → kiến trúc Bi-LSTM thật sự
  - LSTM_UNITS tăng lên [64]: mỗi chiều (fwd/bwd) 64 units, output 128
  - PATIENCE tăng lên 20: tránh dừng quá sớm khi loss dao động
  - MIN_DELTA_ES giảm xuống 0.001: nhạy hơn với cải thiện nhỏ
Thay đổi v5:
  - Bỏ water_level_m khỏi features (tránh học vẹt mực nước hiện tại)
  - Dự báo ΔH thay vì H tuyệt đối
  - Cửa sổ 21 ngày
  - Trọng số mẫu: quan trắc thật = 1.0, nội suy/synthetic = 0.25
  - Tắt bơm đỉnh lũ synthetic vào train
"""

import os

# ============================================================
# ĐƯỜNG DẪN
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

NASA_POWER_PATH = os.path.join(DATA_DIR, "raw", "nasa_power_hourly.csv")
GEE_PATH        = os.path.join(DATA_DIR, "raw", "gee_water_level.csv")

TRAIN_PATH = os.path.join(DATA_DIR, "final", "dataset_train.csv")
VAL_PATH   = os.path.join(DATA_DIR, "final", "dataset_val.csv")
TEST_PATH  = os.path.join(DATA_DIR, "final", "dataset_test.csv")
FULL_PATH  = os.path.join(DATA_DIR, "final", "dataset_full.csv")

SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler_daily.pkl")
TRAIN_CONFIG_PATH = os.path.join(MODEL_DIR, "train_config.json")

# ============================================================
# PHÂN CHIA THỜI GIAN
# ============================================================
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
TRAIN_START = "2019-04-01"   # GEE Sentinel-2 thực tế — bỏ synthetic 2017-2019 khỏi train

# ============================================================
# DỰ BÁO
# ============================================================
FORECAST_DAYS = [1, 3, 7, 14, 30]
TARGET_COL = "water_level_m"
BASE_LEVEL_COL = "base_level_m"
PREDICT_DELTA_H = True          # Học ΔH = H(t+d) - H(t)
TARGET_DELTA_PREFIX = "target_delta_t"

# ============================================================
# FEATURES (18 — không gồm water_level_m, Q_out, delta_h trùng lag)
# ============================================================
# Không dùng lag1/lag3 — gây học vẹt H(t) ≈ H(t+d) → đỉnh lệch d ngày trên valid_time
FEATURE_COLS = [
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
    "temperature", "humidity",
    "water_level_lag7", "water_level_lag14", "water_level_lag30",
    "water_level_roll7", "water_level_std7",
    "month_sin", "month_cos", "season_wet", "season_dry",
]

FEATURE_COUNT = len(FEATURE_COLS)

# Cột giữ nguyên đơn vị mét (không scale)
META_COLS = [BASE_LEVEL_COL, "sample_weight", "is_observed"]

# ============================================================
# CỬA SỔ & MÔ HÌNH Bi-LSTM (v5.1)
# ============================================================
WINDOW_SIZE = 21
LSTM_UNITS = [64]              # Mỗi chiều (fwd/bwd) 64 units → output 128 chiều
USE_BIDIRECTIONAL = True       # Bi-LSTM: học đặc trưng cả hai chiều quá khứ
RECURRENT_DROPOUT = 0.2
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 20                  # Tăng từ 5→20: tránh dừng sớm khi loss dao động
MIN_DELTA_ES = 0.001           # Giảm từ 0.005→0.001: nhạy với cải thiện nhỏ
L2_REG = 1e-3
MC_SAMPLES = 50

# Không trộn persistence — tránh sao chép H(t) làm đỉnh lệch trên biểu đồ
ENSEMBLE_PERSISTENCE_WEIGHT = 0.0
# EarlyStopping trên 15% cuối timeline train (cùng phân phối), không dùng 2023
ES_VAL_FRACTION = 0.15
# Tăng trọng số mẫu khi |ΔH| lớn (đỉnh lũ / biến động mạnh)
SAMPLE_WEIGHT_FLOOD = 3.0
FLOOD_DELTA_THRESHOLD_M = 0.5

# Căn lệch d ngày sau inference (mô hình ≈ H(t) đặt nhầm tại valid=t+d)
APPLY_LAG_D_ALIGNMENT = True

# ============================================================
# DỮ LIỆU / AUGMENTATION
# ============================================================
AUG_START = "2017-01-01"
AUG_END_EXCLUSIVE = "2019-04-01"
MAX_INTERP_GAP_DAYS = 60
ENABLE_FLOOD_INJECTION = False   # Tắt bơm đỉnh lũ SCS-CN (gây lệch phân phối train)
SAMPLE_WEIGHT_OBSERVED = 1.0
SAMPLE_WEIGHT_OTHER = 0.25

# ============================================================
# NGƯỠNG VẬN HÀNH (m)
# ============================================================
MNDBT = 46.20
CANH_BAO = 46.80
NGUY_HIEM = 47.40

AH_CURVE = [
    (50, 34.00), (150, 35.50), (200, 36.00),
    (500, 38.00), (900, 40.00), (1400, 42.00),
    (2000, 44.00), (2500, 46.20), (2700, 46.50),
    (2900, 46.90), (3050, 47.20), (3150, 47.50),
    (3200, 47.80), (3500, 48.25),
]

# ============================================================
# API
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "He thong du bao muc nuoc ho Nui Coc"
API_VERSION = "5.0"


def target_delta_col(horizon_d: int) -> str:
    return f"{TARGET_DELTA_PREFIX}{horizon_d}d"


def target_abs_col(horizon_d: int) -> str:
    return f"target_t{horizon_d}d"

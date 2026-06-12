"""
config.py — Cấu hình trung tâm (v7 — Bi-LSTM + Phase-Aware Loss + TLCC Rain Lags)
====================================================================
Thay đổi v7 (chống trễ pha):
  - LAMBDA_PHASE  : trọng số penalty gradient trong loss (mặc định 0.1)
  - LAMBDA_NSE    : trọng số thưởng NSE trong loss (mặc định 0.05)
  - RAIN_LAG_EXTRA: lag mưa bổ sung từ phân tích TLCC (t-1..t-5 ngày)
  - FEATURE_COLS  : thêm rain_1d_lag1/2/3/5 (20 features, vs 16 cũ)
  - FEATURE_COLS_T7D: thêm rain_1d_lag1/2/3/5 (25 features, vs 21 cũ)
Thay đổi v6 (cải thiện t+7d):
  - FEATURE_COLS_T7D: bộ features riêng cho t+7d với lag/rolling dài hơn
    Thêm: rain_60d, water_level_lag60, roll30, roll60, delta_h_7d, delta_h_30d
  - WINDOW_SIZE_T7D = 45 ngày (vs 21 cho t+1/3d)
  - LSTM_UNITS_T7D = [96] (vs [64]), Dense(64) (vs Dense(32))
  - DROPOUT_RATE_T7D = 0.3 (giảm regularization cho pattern dài hạn)
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
# QUY ƯỚC ĐẶT TÊN (quan trọng — tránh nhầm lẫn):
#   dataset_train.csv  : Tập TRAIN   (2019-04 → 2022-12) — huấn luyện mô hình
#   dataset_test.csv   : Tập ES-VAL  (2023-01 → 2023-12) — EarlyStopping nội bộ
#                        (KHÔNG phải kết quả báo cáo!)
#   dataset_val.csv    : Tập EVAL    (2024-01 → nay)    — kiểm định độc lập
#                        (ĐÂY là kết quả chính thức báo cáo trong luận văn)
#
# EarlyStopping trong 06_bilstm_model.py thực chất dùng 15% cuối
# của X_train (ES_VAL_FRACTION=0.15), không trực tiếp dùng dataset_test.csv.
# dataset_test.csv (2023) được truyền vào hàm train_and_evaluate() để theo dõi
# loss trên tập này, nhưng ModelCheckpoint dựa trên val_loss nội bộ.
TRAIN_END   = "2022-12-31"
VAL_END     = "2023-12-31"
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
# FEATURES (20 — không gồm water_level_m, lag1/3 — chống học vẹt)
# ============================================================
# Không dùng lag1/lag3 mực nước — gây học vẹt H(t) ≈ H(t+d) → đỉnh lệch d ngày
# [v7] Thêm rain_1d_lag1/2/3/5: lag mưa từ TLCC cho tín hiệu dẫn đường thủy văn.
#      Mưa (t-k) → mực nước (t) có tương quan cao nhất tại k=2-3 ngày (hồ Núi Cốc).
#      Các lag này giúp mô hình học phản ứng trước khi mực nước thực sự dâng/hạ.
FEATURE_COLS = [
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
    "temperature", "humidity",
    "water_level_lag7", "water_level_lag14", "water_level_lag30",
    "water_level_roll7", "water_level_std7",
    "month_sin", "month_cos", "season_wet", "season_dry",
    # [v7] Lag mưa theo TLCC — tín hiệu dẫn đường, chống trễ pha
    "rain_1d_lag1", "rain_1d_lag2", "rain_1d_lag3", "rain_1d_lag5",
]

FEATURE_COUNT = len(FEATURE_COLS)

# ============================================================
# FEATURES MỞ RỘNG CHO t+7d (25 features — thêm lag mưa TLCC + dài hạn)
# ============================================================
# Lý do: t+7d cần nắm xu hướng 60-90 ngày (quán tính thủy văn dài)
# Các features thêm so với FEATURE_COLS:
#   rain_60d      : mưa tích lũy 60 ngày — nhận biết đầu/cuối mùa mưa
#   lag60         : mực nước 2 tháng trước — so sánh xu hướng dài hạn
#   roll30        : rolling mean 30 ngày — xu hướng hồ ổn định
#   delta_h_7d    : H(t)-H(t-7) — momentum tăng/giảm 7 ngày gần nhất
#   delta_h_30d   : H(t)-H(t-30) — momentum tăng/giảm 1 tháng gần nhất
#   [v7] rain_1d_lag1/2/3/5 — lag mưa TLCC (giống FEATURE_COLS)
FEATURE_COLS_T7D = [
    # --- Giữ nguyên 16 features gốc ---
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
    "temperature", "humidity",
    "water_level_lag7", "water_level_lag14", "water_level_lag30",
    "water_level_roll7", "water_level_std7",
    "month_sin", "month_cos", "season_wet", "season_dry",
    # --- Thêm 5 features dài hạn ---
    "rain_60d",
    "water_level_lag60",
    "water_level_roll30",
    "delta_h_7d",
    "delta_h_30d",
    # --- [v7] Lag mưa TLCC — tín hiệu dẫn đường chống trễ pha ---
    "rain_1d_lag1", "rain_1d_lag2", "rain_1d_lag3", "rain_1d_lag5",
]

FEATURE_COUNT_T7D = len(FEATURE_COLS_T7D)

# Cột giữ nguyên đơn vị mét (không scale)
META_COLS = [BASE_LEVEL_COL, "sample_weight", "is_observed"]

# ============================================================
# CỬA SỔ & MÔ HÌNH Bi-LSTM (v5.1 — dùng cho t+1d, t+3d)
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

# ============================================================
# CONFIG RIÊNG CHO t+7d (v6 — Extended features + larger model)
# ============================================================
# Lý do tách config riêng: t+7d cần kiến trúc khác t+1d/t+3d
#   - Cửa sổ dài hơn để nắm quán tính thủy văn 45 ngày
#   - LSTM rộng hơn để học pattern phức tạp hơn
#   - Dropout thấp hơn: pattern dài hạn ít nhiễu hơn ngắn hạn
WINDOW_SIZE_T7D    = 45        # 45 ngày (vs 21) — nắm xu hướng 6 tuần
LSTM_UNITS_T7D     = [96]      # 96 units/chiều → output 192 chiều (vs 128)
DROPOUT_RATE_T7D   = 0.3      # Ít regularize hơn (vs 0.5) — pattern dài hạn ổn định
DENSE_UNITS_T7D    = 64       # Dense 64 (vs 32) — học interaction phức tạp hơn
L2_REG_T7D         = 5e-4     # Ít regularize hơn (vs 1e-3)
PATIENCE_T7D       = 25       # Kiên nhẫn hơn với val loss dao động chậm

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
# [v7] PHASE-AWARE LOSS — chống trễ pha
# ============================================================
# Công thức: L = Huber + λ_phase*mean(|Δy_true−Δy_pred|) − λ_nse*NSE
# LAMBDA_PHASE : trọng số penalty gradient (độ dốc) — giảm trễ pha thời gian
#   Khuyến nghị: 0.05-0.15 | 0.0 để tắt hoàn toàn
# LAMBDA_NSE   : trọng số thưởng NSE — tối đa hoá hiệu quả thủy văn
#   Khuyến nghị: 0.03-0.10 | 0.0 để tắt
LAMBDA_PHASE = 0.1
LAMBDA_NSE   = 0.05

# [v7] TLCC Rain Lag — lag mưa bổ sung từ phân tích tương quan dịch thời gian
# Hồ Núi Cốc: tương quan mưa-ΔH đạt cực đại tại lag 2-3 ngày (thời gian tập trung)
# Các giá trị này tương ứng với rain_1d_lag{k} trong build_daily_features()
RAIN_LAG_EXTRA = [1, 2, 3, 5]

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

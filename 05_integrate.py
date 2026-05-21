"""
Bước 5: Tích hợp bộ dữ liệu hoàn chỉnh
=========================================
Luồng xử lý:
  (a) Load dữ liệu từ NASA POWER, GEE Sentinel-2, Báo chí
  (b) Nội suy tuyến tính + Kalman Filter → chuỗi mực nước giờ đầy đủ
  (c) Suy luận số cửa xả (infer_cua_xa) + phát hiện xả bất thường
  (d) Xây dựng feature engineering (lag, rolling, temporal encoding)
  (e) Suy luận Q_out từ phương trình cân bằng nước (gộp từ Bước 7)
      → Đây là điểm cải tiến so với phiên bản trước:
         Q_out được tính ngay tại bước 5, TRƯỚC khi chia train/val/test,
         đảm bảo rolling window không bị đứt giữa các tập dữ liệu.
  (f) Chia tập train / val / test theo thời gian (không xáo trộn)
  (g) Min-Max normalization — scaler chỉ fit trên tập train (anti data-leakage)
  (h) Lưu ra 4 file CSV cho bước huấn luyện mô hình Bi-LSTM

Phiên bản: 2.0 — Tích hợp Q_out features
"""

import os
import sys
import time
import logging
import joblib

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from importlib import util as _iutil

# Cấu hình logging chuẩn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# DYNAMIC IMPORT — load module từ file .py cùng thư mục
# (tránh phụ thuộc vào cấu trúc package)
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)


def _load_module(module_name: str, filename: str):
    """Import động một file .py theo đường dẫn tuyệt đối."""
    filepath = os.path.join(_THIS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy '{filename}' tại: {_THIS_DIR}\n"
            "Hãy đảm bảo tất cả các file .py nằm cùng thư mục."
        )
    spec = _iutil.spec_from_file_location(module_name, filepath)
    mod = _iutil.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import từ Bước 3-4: cửa xả & báo chí
_cua_xa_mod = _load_module("cua_xa_bao_chi", "03_04_cua_xa_bao_chi.py")
infer_cua_xa = _cua_xa_mod.infer_cua_xa
detect_abnormal_release = _cua_xa_mod.detect_abnormal_release
load_bao_chi_data = _cua_xa_mod.load_bao_chi_data

# Import từ Bước 7: suy luận Q_out
# Gộp vào bước 5 để đảm bảo pipeline liên tục, không đứt gãy giữa các bước
_qout_mod = _load_module("infer_qout", "07_infer_qout_1.py")
infer_qout = _qout_mod.infer_qout
detect_sudden_release = _qout_mod.detect_sudden_release
add_qout_features = _qout_mod.add_qout_features


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN & PHÂN CHIA DỮ LIỆU
# ============================================================
OUTPUT_TRAIN = "data/final/dataset_train.csv"
OUTPUT_VAL   = "data/final/dataset_val.csv"
OUTPUT_TEST  = "data/final/dataset_test.csv"
OUTPUT_FULL  = "data/final/dataset_full.csv"  # Dữ liệu thô (chưa normalize)

# Mốc thời gian phân chia — tuân thủ nguyên tắc không xáo trộn (no shuffle)
# Train: 2020–2023 | Val: 2024-01 → 2024-08 | Test: 2024-09+ (lũ Yagi)
TRAIN_END = "2023-06-30"
VAL_END   = "2024-08-31"

FORECAST_HORIZONS = [1, 3, 6, 12, 24]  # Các khoảng dự báo (giờ)

# ============================================================
# FEATURE COLUMNS — 18 đặc trưng đầu vào cho Bi-LSTM
# ============================================================
# Nhóm 1: Khí tượng (5 features)
# Nhóm 2: Lag mực nước — bắt được "trí nhớ" ngắn hạn (5 features)
# Nhóm 3: Cửa xả — hành vi vận hành hồ chứa (2 features)
# Nhóm 4: Q_out — lưu lượng xả suy luận (6 features, MỚI trong v2.0)
FEATURE_COLS = [
    # --- Khí tượng ---
    "rain_1h", "rain_6h", "rain_24h",
    "temperature", "humidity",
    # --- Lag mực nước ---
    "water_level_lag1", "water_level_lag2", "water_level_lag3",
    "water_level_lag6", "water_level_lag12",
    # --- Cửa xả ---
    "so_cua_xa", "dang_xa_cua",
    # --- Q_out (lưu lượng xả suy luận) ---
    "Q_out_smooth",   # Lưu lượng xả đã làm mịn (m³/s)
    "Q_out_lag1",     # Q_out 1 giờ trước
    "Q_out_lag6",     # Q_out 6 giờ trước (xu hướng ngắn hạn)
    "Q_out_roll24",   # Trung bình Q_out 24 giờ (xu hướng ngày)
    "dQout_dt",       # Tốc độ thay đổi Q_out (gia tốc xả)
    "xa_dot_ngot",    # Flag xả đột ngột (0/1)
]

TARGET_COL = "water_level_m"


# ============================================================
# HÀM LOAD NASA CSV AN TOÀN
# ============================================================
def load_nasa_csv(path: str) -> pd.DataFrame:
    """
    Đọc file NASA POWER CSV với xử lý linh hoạt tên cột timestamp.

    Vấn đề gốc:
        01_nasa_power.py lưu bằng df.to_csv() sau khi set_index("timestamp"),
        nên cột đầu tiên là index (không có header tên "timestamp").
        → parse_dates=["timestamp"] sẽ báo lỗi KeyError.

    Giải pháp:
        Tự động phát hiện tên cột timestamp và đọc đúng cách.

    Parameters
    ----------
    path : str
        Đường dẫn đến file CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame với DatetimeIndex tên "timestamp".
    """
    # Đọc header để kiểm tra tên cột
    header_df = pd.read_csv(path, nrows=0)
    cols = header_df.columns.tolist()

    # Các tên cột timestamp phổ biến
    time_candidates = ["timestamp", "datetime", "date", "time", "DATE", "DateTime"]
    time_col = next((c for c in time_candidates if c in cols), None)

    if time_col:
        # Cột timestamp còn nằm trong header (không phải index)
        df = pd.read_csv(path, parse_dates=[time_col], index_col=time_col)
    else:
        # Cột đầu tiên là index (trường hợp phổ biến với 01_nasa_power.py)
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    df.index.name = "timestamp"  # Chuẩn hóa tên index

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"Không thể parse datetime từ file '{path}'.\n"
            f"3 giá trị đầu của index: {df.index[:3].tolist()}"
        )

    logger.info(
        "load_nasa_csv: Đọc OK — %d bản ghi | %s → %s",
        len(df), df.index.min(), df.index.max(),
    )
    return df


# ============================================================
# KALMAN FILTER 1D
# ============================================================
class SimpleKalmanFilter:
    """
    Kalman Filter một chiều dùng để làm mịn và điền khuyết chuỗi mực nước.

    Phù hợp khi quan trắc thưa (Sentinel-2 ~5 ngày/lần) xen kẽ với
    các điểm báo chí thưa hơn. Kalman filter dự đoán trạng thái ở các
    giờ không có đo đạc và cập nhật ngay khi có quan trắc mới.

    Tham số
    -------
    process_variance : float
        Phương sai quá trình — điều khiển tốc độ thay đổi ước lượng.
        Giá trị nhỏ → ước lượng thay đổi chậm (mượt hơn).
    measurement_variance : float
        Phương sai đo lường — phản ánh độ tin cậy của quan trắc.
    """

    def __init__(self, process_variance: float = 1e-4,
                 measurement_variance: float = 0.01):
        self.Q = process_variance
        self.R = measurement_variance
        self.P = 1.0   # Phương sai sai số ước lượng ban đầu
        self.x = None  # Trạng thái ước lượng (mực nước)

    def update(self, measurement=None) -> float:
        """
        Cập nhật một bước thời gian.

        Parameters
        ----------
        measurement : float or None
            Giá trị quan trắc tại bước này. None nếu không có đo đạc.

        Returns
        -------
        float
            Ước lượng mực nước sau khi cập nhật.
        """
        # Khởi tạo lần đầu
        if self.x is None:
            self.x = measurement if measurement is not None else 0.0
            return self.x

        # Bước dự đoán (predict step): tăng phương sai
        self.P += self.Q

        # Bước cập nhật (update step): chỉ khi có quan trắc
        if measurement is not None:
            K = self.P / (self.P + self.R)      # Kalman gain
            self.x += K * (measurement - self.x)
            self.P = (1 - K) * self.P

        return self.x


# ============================================================
# NỘI SUY MỰC NƯỚC
# ============================================================
def interpolate_water_level(
    df_nasa: pd.DataFrame,
    df_gee: pd.DataFrame,
    df_bao_chi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tạo chuỗi mực nước liên tục theo giờ từ hai nguồn dữ liệu thưa:
      - GEE Sentinel-2: ~5 ngày/lần, độ chính xác ±0.3m
      - Báo chí: sự kiện quan trọng (lũ, xả lớn), độ chính xác ±0.1m

    Thuật toán:
      1. Resample cả hai nguồn về tần số giờ
      2. Ưu tiên báo chí khi có chồng chéo (chính xác hơn)
      3. Nội suy tuyến tính theo thời gian để lấp khoảng trống
      4. Áp dụng Kalman Filter 1D để làm mịn nhiễu đo lường

    Parameters
    ----------
    df_nasa : pd.DataFrame
        Dữ liệu NASA POWER với DatetimeIndex theo giờ (xác định phạm vi thời gian).
    df_gee : pd.DataFrame
        Dữ liệu GEE với cột 'date' và 'water_level_m'.
    df_bao_chi : pd.DataFrame
        Dữ liệu báo chí với cột 'timestamp' và 'water_level_bao_chi'.

    Returns
    -------
    pd.DataFrame
        DataFrame với DatetimeIndex theo giờ, cột 'water_level_m'.
    """
    # Tạo index giờ đầy đủ theo phạm vi NASA POWER
    full_idx = pd.date_range(
        start=df_nasa.index.min(),
        end=df_nasa.index.max(),
        freq="h",
    )
    df = pd.DataFrame(index=full_idx)

    # Ghép dữ liệu GEE (resample về giờ)
    gee_series = (
        df_gee.set_index("date")["water_level_m"]
        .resample("h").first()
    )
    df["level_gee"] = gee_series

    # Ghép dữ liệu báo chí (resample về giờ)
    bc_series = (
        df_bao_chi.set_index("timestamp")["water_level_bao_chi"]
        .resample("h").first()
    )
    df["level_bao_chi"] = bc_series

    # Kết hợp: báo chí ưu tiên vì độ chính xác cao hơn
    df["level_obs"] = df["level_bao_chi"].fillna(df["level_gee"])

    # Nội suy tuyến tính theo trục thời gian (limit=None → lấp toàn bộ)
    df["water_level_interp"] = df["level_obs"].interpolate(
        method="time", limit=None
    )

    # Kalman Filter — làm mịn chuỗi nội suy
    kf = SimpleKalmanFilter(process_variance=1e-4, measurement_variance=0.01)
    level_kalman = [
        kf.update(row["level_obs"] if pd.notna(row["level_obs"]) else None)
        for _, row in df.iterrows()
    ]
    df["water_level_m"] = level_kalman

    # Đánh giá sai số tại các điểm có quan trắc thực
    mask_obs = df["level_obs"].notna()
    if mask_obs.sum() > 0:
        mae_interp = (
            df.loc[mask_obs, "water_level_interp"]
            - df.loc[mask_obs, "level_obs"]
        ).abs().mean()
        mae_kalman = (
            df.loc[mask_obs, "water_level_m"]
            - df.loc[mask_obs, "level_obs"]
        ).abs().mean()
        logger.info(
            "Sai số tại %d điểm quan trắc: Nội suy=%.4fm | Kalman=%.4fm",
            mask_obs.sum(), mae_interp, mae_kalman,
        )

    return df[["water_level_m"]].copy()


# ============================================================
# XÂY DỰNG FEATURES
# ============================================================
def build_features(df_level: pd.DataFrame,
                   df_nasa: pd.DataFrame) -> pd.DataFrame:
    """
    Kết hợp mực nước + khí tượng + lag features + temporal encoding.

    Các nhóm feature được xây dựng:
      - Khí tượng (rain_1h/6h/24h, temperature, humidity): từ NASA POWER
      - Lag mực nước (lag1/2/3/6/12): bắt "trí nhớ" ngắn-trung hạn
      - Temporal encoding (sin/cos hour/month): bắt chu kỳ ngày/mùa

    Parameters
    ----------
    df_level : pd.DataFrame
        DataFrame với cột 'water_level_m' và DatetimeIndex giờ.
    df_nasa : pd.DataFrame
        DataFrame với các cột khí tượng và DatetimeIndex giờ.

    Returns
    -------
    pd.DataFrame
        DataFrame đã bổ sung đầy đủ features.
    """
    df = df_level.copy()

    # Ghép dữ liệu khí tượng
    nasa_cols = ["rain_1h", "rain_6h", "rain_24h", "temperature", "humidity"]
    for col in nasa_cols:
        if col in df_nasa.columns:
            df[col] = df_nasa[col]

    # Lag mực nước — bắt trạng thái trước đó
    for lag in [1, 2, 3, 6, 12]:
        df[f"water_level_lag{lag}"] = df["water_level_m"].shift(lag)

    # Temporal encoding (biến đổi sin/cos để mô hình hiểu tính tuần hoàn)
    df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    return df


def add_forecast_targets(df: pd.DataFrame,
                          horizons: list = FORECAST_HORIZONS) -> pd.DataFrame:
    """
    Tạo cột target cho từng khoảng thời gian dự báo.

    Mực nước tại t+h được lấy bằng shift(-h) so với thời điểm hiện tại.
    Các hàng cuối sẽ bị NaN và sẽ được loại bỏ sau bước này.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã có cột 'water_level_m'.
    horizons : list of int
        Danh sách khoảng dự báo tính bằng giờ.

    Returns
    -------
    pd.DataFrame
        DataFrame bổ sung các cột 'target_t{h}h'.
    """
    for h in horizons:
        df[f"target_t{h}h"] = df["water_level_m"].shift(-h)
    return df


# ============================================================
# CHUẨN HÓA — ANTI DATA LEAKAGE
# ============================================================
def normalize_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list,
):
    """
    Min-Max normalization — scaler chỉ fit trên tập train.

    Nguyên tắc anti data-leakage:
        scaler.fit()       → chỉ trên df_train
        scaler.transform() → áp dụng lên cả val và test
    → Mô hình không "nhìn thấy" thống kê của tập val/test trong quá trình
      chuẩn hóa, đảm bảo đánh giá khách quan.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Các tập dữ liệu đã chia theo thời gian.
    feature_cols : list of str
        Danh sách tên cột cần chuẩn hóa.

    Returns
    -------
    tuple
        (df_train_norm, df_val_norm, df_test_norm, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    # Fit chỉ trên train — transform trên cả 3 tập
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols]   = scaler.transform(df_val[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/feature_scaler.pkl")
    logger.info("Đã lưu scaler: models/feature_scaler.pkl")

    return df_train, df_val, df_test, scaler


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs("data/final", exist_ok=True)

    logger.info("=" * 60)
    logger.info("BƯỚC 5: TÍCH HỢP BỘ DỮ LIỆU (v2.0 — tích hợp Q_out)")
    logger.info("=" * 60)

    # ── 5a: Load dữ liệu đầu vào ──────────────────────────────
    logger.info("[5a] Đọc dữ liệu từ các nguồn...")
    df_nasa    = load_nasa_csv("data/raw/nasa_power_hourly.csv")
    df_gee     = pd.read_csv("data/raw/gee_water_level.csv",
                             parse_dates=["date"])
    df_bao_chi = load_bao_chi_data()

    logger.info("  NASA POWER     : %d bản ghi", len(df_nasa))
    logger.info("  GEE Sentinel-2 : %d quan trắc", len(df_gee))
    logger.info("  Báo chí        : %d điểm", len(df_bao_chi))

    # ── 5b: Nội suy mực nước + Kalman Filter ──────────────────
    logger.info("[5b] Nội suy mực nước giờ...")
    df_level = interpolate_water_level(df_nasa, df_gee, df_bao_chi)

    # ── 5c: Cửa xả ────────────────────────────────────────────
    logger.info("[5c] Suy luận trạng thái cửa xả...")
    df_level = infer_cua_xa(df_level, df_bao_chi)
    df_level = detect_abnormal_release(df_level, rain_col="rain_6h")

    # ── 5d: Feature engineering ───────────────────────────────
    logger.info("[5d] Xây dựng features khí tượng + lag...")
    df = build_features(df_level, df_nasa)

    # ── 5e: Suy luận Q_out (tích hợp từ Bước 7) ───────────────
    # QUAN TRỌNG: Tính Q_out trên toàn bộ chuỗi TRƯỚC khi chia split,
    # đảm bảo rolling window 24h không bị đứt giữa train và val.
    logger.info("[5e] Suy luận Q_out từ phương trình cân bằng nước...")
    df = infer_qout(df)
    df = detect_sudden_release(df)
    df = add_qout_features(df)

    # Lưu dataset_full.csv (dữ liệu thô, chưa normalize) — dùng cho phân tích
    df.to_csv(OUTPUT_FULL)
    logger.info("Đã lưu dataset_full.csv: %d bản ghi × %d cột",
                len(df), len(df.columns))

    # ── 5f: Thêm targets dự báo ────────────────────────────────
    logger.info("[5f] Tạo cột target t+1/3/6/12/24h...")
    df = add_forecast_targets(df)

    # Loại bỏ hàng đầu/cuối do lag và forecast horizon
    # max_lag=24 vì Q_out_roll24 cần 24 bước khởi động
    max_lag     = 24
    max_horizon = max(FORECAST_HORIZONS)  # 24 giờ
    df = df.iloc[max_lag:-max_horizon].copy()

    # Loại bỏ hàng còn NaN trong features hoặc targets
    required_cols = FEATURE_COLS + [f"target_t{h}h" for h in FORECAST_HORIZONS]
    df = df.dropna(subset=required_cols)

    logger.info(
        "[5f] Bộ dữ liệu sau làm sạch: %d bản ghi × %d cột",
        len(df), len(df.columns),
    )
    logger.info("  Từ: %s → Đến: %s", df.index.min(), df.index.max())

    # ── 5g: Chia train / val / test ───────────────────────────
    # Chia theo thời gian, KHÔNG xáo trộn (no shuffle) để tránh data leakage
    df_train = df[df.index <= TRAIN_END].copy()
    df_val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)].copy()
    df_test  = df[df.index > VAL_END].copy()

    logger.info("[5g] Chia dữ liệu:")
    logger.info(
        "  Train : %d bản ghi (%s → %s)",
        len(df_train), df_train.index.min().date(), df_train.index.max().date(),
    )
    logger.info(
        "  Val   : %d bản ghi (%s → %s)",
        len(df_val), df_val.index.min().date(), df_val.index.max().date(),
    )
    logger.info(
        "  Test  : %d bản ghi (%s → %s) ← Lũ Yagi 2024",
        len(df_test), df_test.index.min().date(), df_test.index.max().date(),
    )

    # ── 5h: Chuẩn hóa ─────────────────────────────────────────
    logger.info("[5h] Chuẩn hóa Min-Max (fit only on train)...")
    df_train, df_val, df_test, _ = normalize_features(
        df_train, df_val, df_test, FEATURE_COLS
    )

    # ── Lưu output ────────────────────────────────────────────
    df_train.to_csv(OUTPUT_TRAIN)
    df_val.to_csv(OUTPUT_VAL)
    df_test.to_csv(OUTPUT_TEST)

    logger.info("Đã lưu: %s", OUTPUT_TRAIN)
    logger.info("Đã lưu: %s", OUTPUT_VAL)
    logger.info("Đã lưu: %s", OUTPUT_TEST)

    # Thống kê giai đoạn lũ Yagi
    yagi = df_test["2024-09-08":"2024-09-15"]
    if len(yagi) > 0:
        logger.info(
            "[Test — Lũ Yagi] %d giờ | Mực nước max: %.2f m | Giờ xả cửa: %d",
            len(yagi),
            yagi["water_level_m"].max(),
            int(yagi["dang_xa_cua"].sum()),
        )

    # Tóm tắt bộ features
    logger.info("\n[Tóm tắt] %d FEATURE_COLS được dùng:", len(FEATURE_COLS))
    for i, col in enumerate(FEATURE_COLS, 1):
        logger.info("  %2d. %s", i, col)

    logger.info("\n✓ Bước 5 hoàn thành — Chạy tiếp: python 06_bilstm_model.py")


if __name__ == "__main__":
    main()
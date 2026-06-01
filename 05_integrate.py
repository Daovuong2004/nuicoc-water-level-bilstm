"""
Bước 5: Tích hợp bộ dữ liệu ngày (Daily) cho BiLSTM — Hồ Núi Cốc
=====================================================================
Phiên bản: 3.0 — Daily pipeline (thay thế hourly v2.0)

LÝ DO CHUYỂN SANG DAILY:
  Dữ liệu GEE Sentinel-2 chỉ có ~80 điểm thực (sau làm sạch) giai đoạn
  2019-2025. Xây dựng chuỗi giờ từ 80 điểm -> ~80% là nội suy -> không
  đáng tin cậy để train BiLSTM. Giải pháp: dùng tần số NGÀY + augmentation
  thủy văn để đạt >= 300 điểm training.

LUỒNG XỬ LÝ:
  (a) Làm sạch GEE Sentinel-2 (loại fallback 36m, outlier >3500ha, dedup)
  (b) Augmentation dữ liệu thủy văn tổng hợp (2017-2019 + lấp gap)
  (c) NASA POWER aggregate ngày (sum rain, mean temp/humidity)
  (d) Merge và PCHIP interpolation (gap <= 60 ngày)
  (e) Feature engineering ngày (lag 1/3/7/14/30, rolling 7/30, seasonal)
  (f) Q_out daily estimation từ phương trình cân bằng nước
  (g) Chia train/val/test theo thời gian (không shuffle)
  (h) Min-Max normalization (fit chỉ trên train)
  (i) Lưu 4 file CSV cho bước huấn luyện BiLSTM

Phân chia thời gian:
  Train : 2017-01 -> 2022-12  (bao gồm augmented data)
  Val   : 2023-01 -> 2023-12
  Test  : 2024-01 -> 2025-12  (bao gồm lũ Yagi 9/2024)
"""

import os
import sys
import logging
import warnings
import joblib

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=FutureWarning)

# Cấu hình logging chuẩn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)


# ============================================================
# CẤU HÌNH
# ============================================================
OUTPUT_TRAIN = "data/final/dataset_train.csv"
OUTPUT_VAL   = "data/final/dataset_val.csv"
OUTPUT_TEST  = "data/final/dataset_test.csv"
OUTPUT_FULL  = "data/final/dataset_full.csv"

# Phân chia thời gian theo giai đoạn thủy văn
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
# Test: 2024-01-01 -> hết (bao gồm lũ Yagi tháng 9/2024)

# Thông số augmentation
AUG_START           = "2017-01-01"  # Bắt đầu từ khi Sentinel-2A phóng
AUG_END_EXCLUSIVE   = "2019-04-01"  # Điểm GEE đầu tiên thực sự
MAX_INTERP_GAP_DAYS = 60            # Không nội suy quá 60 ngày liên tục

# Feature columns cho BiLSTM daily
FEATURE_COLS = [
    # --- Khí tượng ngày ---
    "rain_1d",           # Lượng mưa ngày (mm)
    "rain_3d",           # Mưa tích lũy 3 ngày
    "rain_7d",           # Mưa tích lũy 7 ngày
    "rain_14d",          # Mưa tích lũy 14 ngày
    "temperature",       # Nhiệt độ trung bình ngày (°C)
    "humidity",          # Độ ẩm trung bình ngày (%)
    # --- Lag mực nước ngày ---
    "water_level_lag1",  # Mực nước ngày hôm qua
    "water_level_lag3",  # 3 ngày trước
    "water_level_lag7",  # 7 ngày trước (xu hướng tuần)
    "water_level_lag14", # 14 ngày trước
    "water_level_lag30", # 30 ngày trước (xu hướng tháng)
    # --- Rolling statistics ---
    "water_level_roll7",  # Trung bình mực nước 7 ngày
    "water_level_roll30", # Trung bình mực nước 30 ngày
    "water_level_std7",   # Độ lệch chuẩn mực nước 7 ngày (biến động)
    # --- Temporal encoding ---
    "month_sin",          # Mã hóa tuần hoàn tháng (sin)
    "month_cos",          # Mã hóa tuần hoàn tháng (cos)
    "season_wet",         # Mùa mưa (tháng 5-10)
    "season_dry",         # Mùa khô (tháng 11-4)
    # --- Q_out daily ---
    "dH_dt_daily",        # Tốc độ thay đổi mực nước (m/ngày)
    "Q_out_daily",        # Lưu lượng xả ước tính ngày (m3/s)
    "Q_out_roll7",        # Trung bình xả 7 ngày
]

TARGET_COL    = "water_level_m"
FORECAST_DAYS = [1, 3, 7, 14, 30]  # Dự báo 1/3/7/14/30 ngày

# Đường cong A-H (giống 02_gee_colab.py) — dùng cho Q_out daily
AH_CURVE = [
    (  50, 34.00), (150, 35.50), (200, 36.00),
    ( 500, 38.00), (900, 40.00), (1400, 42.00),
    (2000, 44.00), (2500, 46.20), (2700, 46.50),
    (2900, 46.90), (3050, 47.20), (3150, 47.50),
    (3200, 47.80), (3500, 48.25),
]
_ah_levels = [p[1] for p in AH_CURVE]
_ah_areas  = [p[0] * 1e4 for p in AH_CURVE]   # ha -> m2
_level_to_area = interp1d(
    _ah_levels, _ah_areas, kind="linear",
    bounds_error=False,
    fill_value=(_ah_areas[0], _ah_areas[-1])
)


# ============================================================
# 5a: LÀM SẠCH GEE SENTINEL-2
# ============================================================
def clean_gee_data(df_gee: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch chuyên sâu dữ liệu GEE Sentinel-2.

    Xử lý cả hai trường hợp:
      - File GEE mới (postprocess() v3.0): không còn fallback 36m
      - File GEE cũ (postprocess() cũ): vẫn có fallback 36.0m -> sẽ lọc ở đây

    Parameters
    ----------
    df_gee : pd.DataFrame
        DataFrame từ gee_water_level.csv

    Returns
    -------
    pd.DataFrame
        Dataset đã làm sạch với DatetimeIndex ngày.
    """
    logger.info("[5a] Làm sạch GEE Sentinel-2...")
    df = df_gee.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["water_area_ha"] = pd.to_numeric(df["water_area_ha"], errors="coerce")
    df["water_level_m"] = pd.to_numeric(df["water_level_m"], errors="coerce")

    # Đồng bộ tên cột mây: hỗ trợ cả file cũ (cloud_scene) và mới (cloud_cover)
    if "cloud_scene" not in df.columns:
        if "cloud_cover" in df.columns:
            df["cloud_scene"] = pd.to_numeric(df["cloud_cover"], errors="coerce")
        else:
            logger.warning("  Không tìm thấy cột cloud_scene/cloud_cover — bỏ qua lọc mây.")
            df["cloud_scene"] = 0.0
    else:
        df["cloud_scene"] = pd.to_numeric(df["cloud_scene"], errors="coerce")

    n0 = len(df)
    logger.info("  Ban đầu: %d bản ghi", n0)

    # Bước 1: Loại outlier vật lý (diện tích ngoài phạm vi hồ Núi Cốc)
    df = df[df["water_area_ha"].between(150, 3500)].copy()
    logger.info(
        "  Sau lọc vật lý (150-3500 ha): %d bản ghi (loại %d)",
        len(df), n0 - len(df)
    )

    # Bước 2: Loại fallback 36.0m
    # round(6) tránh bỏ nhầm giá trị gần 36m thực (ví dụ 36.001m)
    n1 = len(df)
    fallback_mask = (
        df["water_level_m"].notna()
        & (df["water_level_m"].round(6) == 36.0)
    )
    df = df[~fallback_mask].copy()
    logger.info(
        "  Sau loại fallback 36.0m: %d bản ghi (loại %d)",
        len(df), n1 - len(df)
    )

    # Bước 3: Loại NaN mực nước
    n2 = len(df)
    df = df.dropna(subset=["water_level_m"])
    logger.info(
        "  Sau loại NaN: %d bản ghi (loại %d)", len(df), n2 - len(df)
    )

    # Bước 4: Deduplication theo ngày — giữ ảnh ít mây nhất
    df = df.sort_values(["date", "cloud_scene"])
    n3 = len(df)
    df = df.drop_duplicates(subset="date", keep="first")
    logger.info(
        "  Sau dedup ngày: %d bản ghi (loại %d trùng)", len(df), n3 - len(df)
    )

    # Phân cấp chất lượng
    df["quality"] = np.select(
        [df["cloud_scene"] < 30, df["cloud_scene"] < 60, df["cloud_scene"] <= 80],
        ["good", "fair", "low"],
        default="low",
    )

    # Set index ngày
    df = df.set_index("date").sort_index()
    df.index = pd.to_datetime(df.index).normalize()

    # Thống kê
    logger.info("  === Kết quả làm sạch GEE ===")
    logger.info("  Tổng hợp lệ: %d điểm quan trắc", len(df))
    logger.info(
        "  Giai đoạn  : %s -> %s",
        df.index.min().date(), df.index.max().date()
    )
    for q in ["good", "fair", "low"]:
        sub = df[df["quality"] == q]
        if len(sub):
            logger.info(
                "  %s: %d điểm | mực nước %.2f-%.2fm",
                q, len(sub),
                sub["water_level_m"].min(), sub["water_level_m"].max()
            )

    return df[["water_area_ha", "water_level_m", "cloud_scene", "quality"]]


# ============================================================
# 5b: AUGMENTATION DỮ LIỆU THỦY VĂN TỔNG HỢP
# ============================================================
def synthesize_hydrological_data(
    df_gee_clean: pd.DataFrame,
    df_nasa_daily: pd.DataFrame,
    aug_start: str = AUG_START,
    aug_end: str = AUG_END_EXCLUSIVE,
) -> pd.DataFrame:
    """
    Tổng hợp dữ liệu thủy văn cho giai đoạn thiếu data GEE (2017-2019).

    Phương pháp: Học seasonal pattern từ dữ liệu thực (good+fair) theo
    tháng, sau đó sinh dữ liệu với noise thực tế và làm mịn bằng
    Gaussian filter để mô phỏng biến động mực nước tự nhiên.

    Nguyên tắc thủy văn hồ Núi Cốc:
      - Mùa mưa (tháng 5-10): mực nước 40-46m, xu hướng tăng
      - Mùa khô (tháng 11-4): mực nước 36-42m, xu hướng giảm
      - Đỉnh: tháng 9-10 | Đáy: tháng 3-4

    Parameters
    ----------
    df_gee_clean : pd.DataFrame
        Dữ liệu GEE đã làm sạch (DatetimeIndex ngày).
    df_nasa_daily : pd.DataFrame
        Dữ liệu NASA POWER theo ngày.
    aug_start, aug_end : str
        Phạm vi thời gian cần tổng hợp.

    Returns
    -------
    pd.DataFrame
        DataFrame với water_level_m tổng hợp và is_observed=False.
    """
    logger.info("[5b] Tổng hợp dữ liệu thủy văn 2017-2019...")

    # Học seasonal pattern từ dữ liệu thực chất lượng cao
    df_real = df_gee_clean[df_gee_clean["quality"].isin(["good", "fair"])].copy()
    if len(df_real) < 20:
        logger.warning(
            "  Ít dữ liệu good/fair (%d điểm) -> dùng toàn bộ GEE clean.", len(df_real)
        )
        df_real = df_gee_clean.copy()

    # Thống kê mực nước theo tháng
    monthly_stats = (
        df_real.groupby(df_real.index.month)["water_level_m"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mu", "std": "sigma"})
    )

    # Đảm bảo đủ 12 tháng bằng nội suy
    all_months = pd.DataFrame(index=range(1, 13))
    monthly_stats = all_months.join(monthly_stats).interpolate(method="index")
    monthly_stats["sigma"] = monthly_stats["sigma"].fillna(0.5).clip(lower=0.3)

    # Tạo index ngày cần tổng hợp
    aug_idx = pd.date_range(start=aug_start, end=aug_end, freq="D")[:-1]

    # Sinh mực nước từng ngày với seasonal pattern + noise giới hạn
    np.random.seed(42)
    synth_levels = []
    for date in aug_idx:
        month = date.month
        mu    = monthly_stats.loc[month, "mu"]
        sigma = monthly_stats.loc[month, "sigma"]

        noise = np.random.normal(0, sigma * 0.4)
        noise = np.clip(noise, -1.5 * sigma, 1.5 * sigma)
        level = np.clip(mu + noise, 36.0, 48.25)
        synth_levels.append(level)

    # Gaussian smoothing: sigma=7 ngày -> mô phỏng quán tính thủy văn
    synth_levels_arr = np.array(synth_levels)
    synth_levels_smooth = gaussian_filter1d(synth_levels_arr, sigma=7)

    df_synth = pd.DataFrame({
        "water_level_m": synth_levels_smooth,
        "is_observed":   False,
        "quality":       "synthetic",
    }, index=aug_idx)

    logger.info(
        "  Tổng hợp: %d điểm ngày (%s -> %s) | %.2f-%.2fm",
        len(df_synth),
        df_synth.index.min().date(),
        df_synth.index.max().date(),
        df_synth["water_level_m"].min(),
        df_synth["water_level_m"].max(),
    )

    return df_synth[["water_level_m", "is_observed", "quality"]]


# ============================================================
# 5c: RESAMPLE VÀ INTERPOLATION
# ============================================================
def build_daily_water_level(
    df_gee_clean: pd.DataFrame,
    df_synth: pd.DataFrame,
    max_gap_days: int = MAX_INTERP_GAP_DAYS,
) -> pd.DataFrame:
    """
    Xây dựng chuỗi mực nước ngày đầy đủ từ GEE + dữ liệu tổng hợp.

    Ưu tiên: GEE thực tế > synthetic > PCHIP interpolation
    Gap > max_gap_days: để NaN (không nội suy sai)

    Parameters
    ----------
    df_gee_clean : pd.DataFrame
        Dữ liệu GEE đã làm sạch.
    df_synth : pd.DataFrame
        Dữ liệu tổng hợp 2017-2019.
    max_gap_days : int
        Khoảng cách tối đa cho phép nội suy (ngày).

    Returns
    -------
    pd.DataFrame
        DataFrame ngày với water_level_m và is_observed.
    """
    logger.info("[5c] Xây dựng chuỗi mực nước ngày liên tục...")

    # GEE thực: is_observed = True
    df_gee_merge = df_gee_clean[["water_level_m"]].copy()
    df_gee_merge["is_observed"] = True

    # Synthetic: chỉ lấy ngày chưa có GEE
    df_synth_filt = df_synth[
        ~df_synth.index.isin(df_gee_merge.index)
    ][["water_level_m", "is_observed"]].copy()

    # Ghép: GEE > synthetic
    df_obs = pd.concat([df_gee_merge, df_synth_filt]).sort_index()
    df_obs = df_obs[~df_obs.index.duplicated(keep="first")]

    # Lưới ngày đầy đủ
    full_idx = pd.date_range(
        start=df_obs.index.min(),
        end=df_obs.index.max(),
        freq="D",
    )
    df_daily = pd.DataFrame(index=full_idx)
    df_daily["water_level_m"] = df_obs["water_level_m"]
    df_daily["is_observed"]   = df_obs["is_observed"].reindex(full_idx, fill_value=False)

    # Đánh dấu gap lớn: không nội suy vào đó
    obs_dates = df_obs.index

    def _mark_large_gaps(series, obs_dates, max_gap):
        """Giữ NaN cho vị trí nằm trong gap > max_gap ngày."""
        result = series.copy()
        nan_mask = series.isna()
        for i, date in enumerate(series.index):
            if not nan_mask.iloc[i]:
                continue
            before = obs_dates[obs_dates < date]
            after  = obs_dates[obs_dates > date]
            if len(before) == 0 or len(after) == 0:
                continue   # biên -> sẽ NaN sau PCHIP extrapolate=False
            gap_left  = (date - before[-1]).days
            gap_right = (after[0]  - date).days
            if min(gap_left, gap_right) > max_gap:
                result.iloc[i] = np.nan
        return result

    df_daily["water_level_m"] = _mark_large_gaps(
        df_daily["water_level_m"], obs_dates, max_gap_days
    )

    # PCHIP interpolation (bảo toàn đơn điệu cục bộ)
    mask_valid = df_daily["water_level_m"].notna()
    if mask_valid.sum() >= 4:
        x_valid  = np.where(mask_valid)[0]
        y_valid  = df_daily["water_level_m"].values[mask_valid]
        pchip    = PchipInterpolator(x_valid, y_valid, extrapolate=False)
        x_all    = np.arange(len(df_daily))
        y_interp = pchip(x_all)

        nan_pos = np.where(df_daily["water_level_m"].isna())[0]
        if len(nan_pos):
            df_daily.loc[df_daily.index[nan_pos], "water_level_m"] = y_interp[nan_pos]

    df_daily["water_level_m"] = df_daily["water_level_m"].clip(34.0, 48.25)

    n_obs   = int(df_daily["is_observed"].sum())
    n_total = int(df_daily["water_level_m"].notna().sum())
    n_nan   = int(df_daily["water_level_m"].isna().sum())
    logger.info(
        "  Lưới ngày: %d ngày | %d quan trắc thực | %d nội suy | %d gap NaN",
        len(df_daily), n_obs, n_total - n_obs, n_nan
    )

    return df_daily


# ============================================================
# 5d: NASA POWER AGGREGATE NGÀY
# ============================================================
def aggregate_nasa_daily(path_nasa: str) -> pd.DataFrame:
    """
    Đọc NASA POWER hourly và aggregate về tần số ngày.

    Aggregation:
      rain    : sum  (tổng mưa ngày mm)
      temp    : mean (nhiệt độ TB ngày °C)
      humidity: mean (độ ẩm TB ngày %)

    Parameters
    ----------
    path_nasa : str
        Đường dẫn nasa_power_hourly.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame ngày với rain_1d/3d/7d/14d, temperature, humidity.
    """
    logger.info("[5d] Aggregate NASA POWER -> ngày...")

    header_df = pd.read_csv(path_nasa, nrows=0)
    cols = header_df.columns.tolist()
    time_candidates = ["timestamp", "datetime", "date", "time", "DATE"]
    time_col = next((c for c in time_candidates if c in cols), None)

    if time_col:
        df = pd.read_csv(path_nasa, parse_dates=[time_col], index_col=time_col)
    else:
        df = pd.read_csv(path_nasa, index_col=0, parse_dates=True)
    df.index.name = "timestamp"

    # Xác định cột mưa giờ
    rain_col = next(
        (c for c in ["rain_1h", "rain_hourly", "PRECTOTCORR"] if c in df.columns),
        None
    )

    # Aggregate
    agg_dict = {}
    if rain_col:
        agg_dict["rain_1d"] = (rain_col, "sum")
    if "temperature" in df.columns:
        agg_dict["temperature"] = ("temperature", "mean")
    if "humidity" in df.columns:
        agg_dict["humidity"] = ("humidity", "mean")

    df_daily = df.resample("D").agg(**agg_dict)

    if "rain_1d" in df_daily.columns:
        df_daily["rain_1d"]  = df_daily["rain_1d"].clip(lower=0)
        df_daily["rain_3d"]  = df_daily["rain_1d"].rolling(3,  min_periods=1).sum()
        df_daily["rain_7d"]  = df_daily["rain_1d"].rolling(7,  min_periods=1).sum()
        df_daily["rain_14d"] = df_daily["rain_1d"].rolling(14, min_periods=1).sum()

    df_daily.index = pd.to_datetime(df_daily.index).normalize()

    logger.info(
        "  NASA daily: %d ngày | %s -> %s",
        len(df_daily),
        df_daily.index.min().date(),
        df_daily.index.max().date()
    )
    return df_daily


# ============================================================
# 5e: FEATURE ENGINEERING NGÀY
# ============================================================
def build_daily_features(
    df_level: pd.DataFrame,
    df_nasa_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Xây dựng bộ đặc trưng ngày (21 features) cho BiLSTM.

    Nhóm features:
      1. Khí tượng ngày : rain_1d/3d/7d/14d, temperature, humidity
      2. Lag mực nước   : 1/3/7/14/30 ngày
      3. Rolling stats  : mean 7/30 ngày, std 7 ngày
      4. Temporal       : month_sin/cos, season_wet/dry
      5. Q_out daily    : dH_dt_daily, Q_out_daily, Q_out_roll7

    Parameters
    ----------
    df_level : pd.DataFrame
        DataFrame với water_level_m (DatetimeIndex ngày).
    df_nasa_daily : pd.DataFrame
        NASA POWER theo ngày.

    Returns
    -------
    pd.DataFrame
        DataFrame đầy đủ features.
    """
    logger.info("[5e] Xây dựng features ngày...")

    df = df_level.copy()

    # --- Ghép khí tượng ---
    nasa_cols = ["rain_1d", "rain_3d", "rain_7d", "rain_14d",
                 "temperature", "humidity"]
    for col in nasa_cols:
        if col in df_nasa_daily.columns:
            df[col] = df_nasa_daily[col].reindex(df.index)

    # --- Lag mực nước ---
    for lag in [1, 3, 7, 14, 30]:
        df[f"water_level_lag{lag}"] = df["water_level_m"].shift(lag)

    # --- Rolling statistics ---
    df["water_level_roll7"]  = df["water_level_m"].rolling(7,  min_periods=3).mean()
    df["water_level_roll30"] = df["water_level_m"].rolling(30, min_periods=7).mean()
    df["water_level_std7"]   = df["water_level_m"].rolling(7,  min_periods=3).std()

    # --- Temporal encoding (tuần hoàn) ---
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    df["season_wet"] = df.index.month.isin([5, 6, 7, 8, 9, 10]).astype(int)
    df["season_dry"] = df.index.month.isin([11, 12, 1, 2, 3, 4]).astype(int)

    # --- Q_out daily (phương trình cân bằng nước) ---
    # dH/dt (m/ngày)
    df["dH_dt_daily"] = df["water_level_m"].diff(1)

    # Diện tích mặt hồ tại mực nước H (m2)
    df["_area_m2"] = df["water_level_m"].apply(
        lambda h: float(_level_to_area(h)) if pd.notna(h) else np.nan
    )

    # Q_out = -A(H) * dH/dt / 86400  [m3/s]
    # Dấu âm: mực nước giảm (dH < 0) -> đang xả ra ngoài
    df["Q_out_daily"] = -(df["_area_m2"] * df["dH_dt_daily"]) / 86400.0
    df["Q_out_daily"] = df["Q_out_daily"].clip(lower=0)
    df["Q_out_roll7"] = df["Q_out_daily"].rolling(7, min_periods=1).mean()

    df.drop(columns=["_area_m2"], inplace=True)

    logger.info(
        "  Features: %d cột x %d ngày | mực nước %.2f-%.2fm",
        len(df.columns), len(df),
        df["water_level_m"].min(), df["water_level_m"].max()
    )

    return df


# ============================================================
# 5f: TẠO CỘT TARGET
# ============================================================
def add_forecast_targets(
    df: pd.DataFrame,
    horizons: list = None,
) -> pd.DataFrame:
    """Tạo cột target cho từng khoảng dự báo ngày (shift(-d))."""
    if horizons is None:
        horizons = FORECAST_DAYS
    for d in horizons:
        df[f"target_t{d}d"] = df["water_level_m"].shift(-d)
    return df


# ============================================================
# 5g: CHUẨN HÓA — ANTI DATA LEAKAGE
# ============================================================
def normalize_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list,
):
    """
    Min-Max normalization — scaler fit ONLY trên train.
    Anti data-leakage: val/test thống kê không ảnh hưởng training.
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()

    cols_present = [c for c in feature_cols if c in df_train.columns]
    missing      = [c for c in feature_cols if c not in df_train.columns]
    if missing:
        logger.warning("  Thiếu %d feature cols: %s", len(missing), missing)

    df_train[cols_present] = scaler.fit_transform(df_train[cols_present])
    df_val[cols_present]   = scaler.transform(df_val[cols_present])
    df_test[cols_present]  = scaler.transform(df_test[cols_present])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/feature_scaler_daily.pkl")
    logger.info("  Đã lưu scaler: models/feature_scaler_daily.pkl")

    return df_train, df_val, df_test, scaler, cols_present


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs("data/final", exist_ok=True)

    logger.info("=" * 65)
    logger.info("BUOC 5 (v3.0): PIPELINE NGAY -- Ho Nui Coc BiLSTM")
    logger.info("=" * 65)

    # -- 5a: Load và làm sạch GEE --
    logger.info("\n[Load] Doc du lieu GEE Sentinel-2...")
    df_gee_raw = pd.read_csv(
        "data/raw/gee_water_level.csv",
        parse_dates=["date"],
    )
    logger.info("  GEE raw: %d ban ghi", len(df_gee_raw))
    df_gee = clean_gee_data(df_gee_raw)

    # -- 5d: NASA POWER aggregate ngày --
    logger.info("\n[Load] Aggregate NASA POWER -> ngay...")
    df_nasa_daily = aggregate_nasa_daily("data/raw/nasa_power_hourly.csv")

    # -- 5b: Augmentation 2017-2019 --
    logger.info("\n[Augment] Tong hop du lieu 2017-2019...")
    df_synth = synthesize_hydrological_data(df_gee, df_nasa_daily)

    # -- 5c: Xây dựng chuỗi ngày đầy đủ --
    logger.info("\n[Resample] Xay dung chuoi muc nuoc ngay...")
    df_daily_level = build_daily_water_level(df_gee, df_synth)

    # -- 5e: Feature engineering --
    logger.info("\n[Features] Xay dung features ngay...")
    df = build_daily_features(df_daily_level, df_nasa_daily)

    # Lưu dataset_full (trước normalize, để phân tích)
    df.to_csv(OUTPUT_FULL)
    logger.info(
        "Da luu dataset_full.csv: %d ban ghi x %d cot",
        len(df), len(df.columns)
    )

    # -- 5f: Thêm targets --
    logger.info("\n[Target] Tao cot du bao t+1/3/7/14/30 ngay...")
    df = add_forecast_targets(df)

    # Cắt hàng đầu/cuối do lag và forecast horizon
    max_lag     = 30  # water_level_lag30
    max_horizon = 30  # target_t30d
    df = df.iloc[max_lag:-max_horizon].copy()

    # Loại hàng NaN trong features/targets
    target_cols   = [f"target_t{d}d" for d in FORECAST_DAYS]
    required_cols = FEATURE_COLS + target_cols
    avail_cols    = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=avail_cols)

    logger.info(
        "\n[Dataset] Sau lam sach cuoi: %d ban ghi x %d cot",
        len(df), len(df.columns)
    )
    logger.info(
        "  Giai doan: %s -> %s",
        df.index.min().date(), df.index.max().date()
    )

    # -- Chia train/val/test --
    df_train = df[df.index <= TRAIN_END].copy()
    df_val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)].copy()
    df_test  = df[df.index > VAL_END].copy()

    logger.info("\n[Split] Chia du lieu:")
    logger.info(
        "  Train : %d ngay (%s -> %s)",
        len(df_train),
        df_train.index.min().date(), df_train.index.max().date()
    )
    logger.info(
        "  Val   : %d ngay (%s -> %s)",
        len(df_val),
        df_val.index.min().date(), df_val.index.max().date()
    )
    logger.info(
        "  Test  : %d ngay (%s -> %s) <- bao gom lu Yagi 2024",
        len(df_test),
        df_test.index.min().date(), df_test.index.max().date()
    )

    # Cảnh báo dataset nhỏ
    if len(df_train) < 200:
        logger.warning(
            "CANH BAO: Train chi co %d ngay (nen >= 200 de BiLSTM on dinh).",
            len(df_train)
        )
    if len(df_val) < 50:
        logger.warning("CANH BAO: Val chi co %d ngay (nen >= 50).", len(df_val))
    if len(df_test) < 50:
        logger.warning("CANH BAO: Test chi co %d ngay (nen >= 50).", len(df_test))

    # -- Chuẩn hóa --
    logger.info("\n[Normalize] Min-Max (fit only on train)...")
    df_train, df_val, df_test, _, feature_cols_present = normalize_features(
        df_train, df_val, df_test, FEATURE_COLS
    )

    # -- Lưu --
    df_train.to_csv(OUTPUT_TRAIN)
    df_val.to_csv(OUTPUT_VAL)
    df_test.to_csv(OUTPUT_TEST)

    logger.info("\n[Luu] Da luu:")
    logger.info("  %s (%d dong)", OUTPUT_TRAIN, len(df_train))
    logger.info("  %s (%d dong)", OUTPUT_VAL,   len(df_val))
    logger.info("  %s (%d dong)", OUTPUT_TEST,  len(df_test))

    # Tóm tắt
    logger.info("\n" + "=" * 65)
    logger.info("TOM TAT BO DU LIEU NGAY:")
    logger.info(
        "  Tong: %d ngay | Train: %d | Val: %d | Test: %d",
        len(df), len(df_train), len(df_val), len(df_test)
    )
    logger.info("  Features: %d cot", len(feature_cols_present))
    logger.info("  Horizons: %s ngay", FORECAST_DAYS)
    if "is_observed" in df.columns:
        obs_pct = int(df["is_observed"].mean() * 100)
        logger.info("  Ty le diem thuc / tong: ~%d%%", obs_pct)
    logger.info("=" * 65)
    logger.info("\nBuoc 5 hoan thanh -- Chay tiep: python 06_bilstm_model.py")


if __name__ == "__main__":
    main()
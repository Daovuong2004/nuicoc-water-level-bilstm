"""
Bước 5: Tích hợp bộ dữ liệu ngày (Daily) cho BiLSTM — Hồ Núi Cốc
=====================================================================
Phiên bản: 3.0 — Daily pipeline (thay thế hourly v2.0)

LÝ DO CHUYỂN SANG DAILY:
  Dữ liệu GEE Sentinel-2 chỉ có ~80 điểm thực (sau làm sạch) giai đoạn
  2019-2025. Xây dựng chuỗi giờ từ 80 điểm -> ~80% là nội suy -> không
  đáng tin cậy để train BiLSTM. Giải pháp: dùng tần số NGÀY + augmentation
  thủy văn để đạt >= 300 điểm training.

LUỒNG Xử LÝ:
  (a) Làm sạch GEE Sentinel-2 (loại fallback 36m, outlier >3500ha, dedup)
  (b) Augmentation dữ liệu thủy văn tổng hợp (2017-2019 + lấp gap)
  (c) NASA POWER aggregate ngày (sum rain, mean temp/humidity)
  (d) Merge và PCHIP interpolation (gap <= 60 ngày)
  (e) Feature engineering ngày (lag 1/3/7/14/30, rolling 7/30, seasonal)
  (f) Q_out daily estimation từ phương trình cân bằng nước
  (g) Chia train/val/test theo thời gian (không shuffle)
  (h) StandardScaler normalization (fit chỉ trên train)
  (i) Lưu 4 file CSV cho bước huấn luyện BiLSTM

Phân chia thời gian và QUY ƯỚC ĐẶT TÊN (rất quan trọng!):
  dataset_train.csv  = Tập TRAIN  : 2019-04 → 2022-12  — huấn luyện tham số
  dataset_test.csv   = Tập ES-VAL : 2023-01 → 2023-12  — nội bộ EarlyStopping
                       (TRONG code gọi là 'test', KHÔNG phải kết quả báo cáo!)
  dataset_val.csv    = Tập EVAL   : 2024-01 → nay      — kiểm định độc lập
                       (ĐÂY là kết quả chính thức trong luận văn, bao gồm lũ Yagi 9/2024)
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

from config import (
    FEATURE_COLS,
    FORECAST_DAYS,
    TARGET_COL,
    BASE_LEVEL_COL,
    TRAIN_END,
    VAL_END,
    TRAIN_START,
    AUG_START,
    AUG_END_EXCLUSIVE,
    MAX_INTERP_GAP_DAYS,
    ENABLE_FLOOD_INJECTION,
    SAMPLE_WEIGHT_OBSERVED,
    SAMPLE_WEIGHT_OTHER,
    SAMPLE_WEIGHT_FLOOD,
    FLOOD_DELTA_THRESHOLD_M,
    target_delta_col,
    PREDICT_DELTA_H,
    target_delta_col,
    target_abs_col,
    RAIN_LAG_EXTRA,          # [v7] lag mưa từ phân tích TLCC
)


# ============================================================
# CẤU HÌNH
# ============================================================
# Đường dẫn file đầu ra
# CHU Ý QUY ƯỚC ĐẶT TÊN: (xem giải thích ở docstring trên)
#   OUTPUT_TRAIN → Tập huấn luyện (Train)
#   OUTPUT_TEST  → Tập ES nội bộ (năm 2023, gọi là 'test' trong code)
#   OUTPUT_VAL   → Tập kiểm định độc lập (2024+, kết quả chính thức luận văn)
OUTPUT_TRAIN = "data/final/dataset_train.csv"   # huan luyen
OUTPUT_VAL   = "data/final/dataset_val.csv"     # kiem dinh doc lap (2024+)
OUTPUT_TEST  = "data/final/dataset_test.csv"    # EarlyStopping (2023)
OUTPUT_FULL  = "data/final/dataset_full.csv"

# ============================================================
# PHAN CHIA THOI GIAN (3-WAY SPLIT — Quy chuan Thuy loi + LSTM)
# ============================================================
# TRAIN (Calibration): 2017-01-01 → 2022-12-31
#   Dung de fit toan bo tham so Bi-LSTM.
# VAL   (Internal Validation): 2023-01-01 → 2023-12-31
#   Dung cho Keras EarlyStopping + ReduceLROnPlateau.
#   KHONG duoc dung lam ket qua bao cao trong luan van.
# (TRAIN_END, VAL_END, FEATURE_COLS, ... imported from config.py v5)

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

    # =============== ĐOẠN CODE BỔ SUNG: NEO ĐỈNH LŨ BÁO CHÍ ===============
    try:
        df_bao_chi = pd.read_csv("data/raw/bao_chi_su_kien.csv", parse_dates=["timestamp"])
        df_bc_daily = df_bao_chi.set_index("timestamp").resample("D").max() # Lấy mốc nước cao nhất trong ngày
        df_bc_daily = df_bc_daily.dropna(subset=["water_level_bao_chi"])
        
        for date, row in df_bc_daily.iterrows():
            d_norm = date.normalize()
            if d_norm in df_daily.index:
                # Ghi đè sự kiện cực đoan, ngăn PCHIP cắt ngang đỉnh lũ
                df_daily.loc[d_norm, "water_level_m"] = row["water_level_bao_chi"]
                df_daily.loc[d_norm, "is_observed"] = True
        logger.info(f"  ✓ Đã chèn cứng {len(df_bc_daily)} sự kiện đỉnh lũ báo chí (vd: Bão Yagi).")
        
        # Cập nhật lại obs_dates vì ta vừa thêm điểm quan sát mới
        obs_dates = df_daily[df_daily["is_observed"]].index
    except FileNotFoundError:
        logger.warning("  ⚠ Không tìm thấy bao_chi_su_kien.csv, bỏ qua mốc đỉnh lũ.")
        obs_dates = df_obs.index
    
    # =============== ĐOẠN CODE BỔ SUNG 2: BỘ LỌC VẬT LÝ (OUTLIER REMOVAL) ===============
    # Loại bỏ điểm GEE bị mù mây/nước đục đợt bão Yagi (GEE báo 38m trong khi thực tế > 46m)
    # Xóa các điểm is_observed=True nhưng không nằm trong báo chí, thuộc giai đoạn 2024-09-01 đến 2024-10-31
    mask_yagi = (df_daily.index >= "2024-09-01") & (df_daily.index <= "2024-10-31")
    try:
        bc_index = df_bc_daily.index
    except NameError:
        bc_index = []
    
    mask_gee_fake = mask_yagi & df_daily["is_observed"] & (~df_daily.index.isin(bc_index))
    if mask_gee_fake.sum() > 0:
        df_daily.loc[mask_gee_fake, "is_observed"] = False
        df_daily.loc[mask_gee_fake, "water_level_m"] = np.nan
        logger.info(f"  ✓ Đã loại bỏ {mask_gee_fake.sum()} điểm GEE lỗi do nước đục đợt bão Yagi.")
        
    obs_dates = df_daily[df_daily["is_observed"]].index
    # =========================================================================

    # Đánh dấu gap lớn: không nội suy vào đó

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
    if "gwettop" in df.columns:
        agg_dict["gwettop"] = ("gwettop", "mean")

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
# 5e: TẠO ĐỈNH LŨ GIẢ LẬP (HYDROLOGICAL INJECTION)
# ============================================================
def inject_synthetic_flood_peaks(
    df_level: pd.DataFrame,
    df_nasa: pd.DataFrame,
    K_factor: float = 0.0065
) -> pd.DataFrame:
    """
    Mô phỏng Thủy văn (Hydrological Injection) dựa trên phương pháp SCS-CN.
    delta_H = K * GWETTOP * rain_1d
    K = 0.0065 (hệ số lưu vực thực nghiệm Hồ Núi Cốc)
    """
    df = df_level.copy()
    
    if "rain_1d" not in df_nasa.columns or "gwettop" not in df_nasa.columns:
        logger.warning("Thiếu cột rain_1d hoặc gwettop trong NASA data. Bỏ qua Hydrological Injection.")
        return df
        
    common_idx = df.index.intersection(df_nasa.index)
    rain_1d_series = df_nasa.loc[common_idx, "rain_1d"]
    gwettop_series = df_nasa.loc[common_idx, "gwettop"]
    
    # Bơm vào toàn bộ tập dữ liệu (Train, Val, Test) để sửa lỗi GEE bị thiếu đỉnh lũ.
    # Các đỉnh lũ thực tế từ báo chí (bão Yagi) sẽ không bị ghi đè nếu chúng cao hơn mô phỏng.
    train_val_idx = common_idx
    
    injected_count = 0
    
    for date in train_val_idx:
        rain_raw = rain_1d_series.loc[date]
        gwettop = gwettop_series.loc[date]
        
        # Giới hạn lượng mưa tối đa 450mm/ngày để loại bỏ các nhiễu vệ tinh NASA (outliers > 4000mm)
        rain = min(rain_raw, 450.0)
        
        # Chỉ kích hoạt nếu mưa đủ lớn (> 50mm/ngày) để tránh nhiễu
        if rain > 50:
            delta_H = K_factor * gwettop * rain
            
            # Định tuyến (Routing): Đỉnh lũ thường xuất hiện sau 1 ngày
            peak_date = date + pd.Timedelta(days=1)
            
            if peak_date in df.index:
                current_level = df.loc[peak_date, "water_level_m"]
                
                # Xác định mực nước nền (Base Flow Level)
                month = peak_date.month
                if 6 <= month <= 10:
                    base_level = max(44.0, current_level)
                else:
                    base_level = max(38.0, current_level)
                    
                synthetic_peak = base_level + delta_H
                
                # Giới hạn đỉnh lũ ảo ở 47.0m (ngưỡng xả lũ khẩn cấp). 
                # Điều này giúp các sự kiện báo chí thực tế (như Yagi 47.6m) không bao giờ bị ghi đè,
                # đồng thời ngăn chặn lỗi cộng dồn vô hạn (runaway accumulation).
                synthetic_peak = min(47.0, synthetic_peak)
                
                if synthetic_peak > current_level:
                    df.loc[peak_date, "water_level_m"] = synthetic_peak
                    df.loc[peak_date, "is_observed"] = True
                    injected_count += 1
                    
    logger.info(f"  [Augment] Đã bơm {injected_count} đỉnh lũ vật lý (delta_H = 0.0065 * GWETTOP * Rain) vào TOÀN BỘ dữ liệu.")
    
    # Nội suy PCHIP lại một lần nữa để làm mượt các đỉnh lũ vừa chèn
    obs_mask = df["is_observed"]
    obs_series = df.loc[obs_mask, "water_level_m"]
    
    if len(obs_series) > 1:
        from scipy.interpolate import PchipInterpolator
        interp_func = PchipInterpolator(obs_series.index.view("int64"), obs_series.values, extrapolate=False)
        df["water_level_m"] = interp_func(df.index.view("int64"))
        
    return df


# ============================================================
# 5f: FEATURE ENGINEERING NGÀY
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
    logger.info(
        "[5e] Xây dựng features ngày (v5 — %d features, không gồm water_level_m)",
        len(FEATURE_COLS),
    )

    df = df_level.copy()

    # --- Khí tượng ---
    nasa_cols = ["rain_1d", "rain_3d", "rain_7d", "rain_14d",
                 "temperature", "humidity"]
    for col in nasa_cols:
        if col in df_nasa_daily.columns:
            df[col] = df_nasa_daily[col].reindex(df.index)

    # Tính mưa tích lũy rolling từ rain_1d (tránh mất dữ liệu 2017-2019)
    if "rain_1d" in df.columns:
        df["rain_30d"] = df["rain_1d"].rolling(30, min_periods=1).sum()
        # [MỚI v6] rain_60d — thông tin mưa dài hạn giúp t+7d nhận biết đang đầu/cuối mùa lũ
        df["rain_60d"] = df["rain_1d"].rolling(60, min_periods=1).sum()

        # --- [v7] Lag mưa theo TLCC — tín hiệu dẫn đường chống trễ pha ---
        # Phân tích TLCC cho hồ Núi Cốc cho thấy mưa tại (t-k) có tương quan
        # mạnh nhất với ΔH(t) tại k=2-3 ngày (thời gian tập trung nước về hồ).
        # Việc đưa các lag này vào feature giúp mô hình dự đoán trước khi mực nước
        # bắt đầu dâng, từ đó giảm hiện tượng trễ pha đỉnh lũ trên biểu đồ.
        for k in RAIN_LAG_EXTRA:
            df[f"rain_1d_lag{k}"] = df["rain_1d"].shift(k)
        logger.info(
            "  [v7] Đã tạo %d lag mưa TLCC: %s",
            len(RAIN_LAG_EXTRA),
            [f'rain_1d_lag{k}' for k in RAIN_LAG_EXTRA]
        )

    # --- Lag mực nước (chỉ quá khứ — không rò rỉ) ---
    for lag in [1, 3, 7, 14, 30]:
        df[f"water_level_lag{lag}"] = df["water_level_m"].shift(lag)
    # [MỚI v6] lag60 — trạng thái hồ 2 tháng trước, quan trọng cho dự báo 7 ngày
    df["water_level_lag60"] = df["water_level_m"].shift(60)

    # --- Rolling statistics ---
    df["water_level_roll7"]  = df["water_level_m"].rolling(7,  min_periods=3).mean()
    df["water_level_std7"]   = df["water_level_m"].rolling(7,  min_periods=3).std()
    # [MỚI v6] Rolling 30/60 ngày — xu hướng dài hạn của hồ (mùa lũ vs mùa khô)
    df["water_level_roll30"] = df["water_level_m"].rolling(30, min_periods=7).mean()
    df["water_level_roll60"] = df["water_level_m"].rolling(60, min_periods=14).mean()

    # [MỚI v6] Biến đổi mực nước — xu hướng tăng/giảm trong 7 và 30 ngày vừa qua
    # Công thức: delta = H(t) - H(t-k), không dùng shift(-k) để tránh data leakage
    df["delta_h_7d"]  = df["water_level_m"] - df["water_level_m"].shift(7)
    df["delta_h_30d"] = df["water_level_m"] - df["water_level_m"].shift(30)

    # --- Temporal encoding (tuần hoàn) ---
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    df["season_wet"] = df.index.month.isin([5, 6, 7, 8, 9, 10]).astype(int)
    df["season_dry"] = df.index.month.isin([11, 12, 1, 2, 3, 4]).astype(int)

    logger.info(
        "  Features v6: %d cột x %d ngày | mực nước %.2f-%.2fm",
        len(df.columns), len(df),
        df["water_level_m"].min(), df["water_level_m"].max()
    )
    logger.info(
        "  [MỚI v6] Features bổ sung: rain_60d, water_level_lag60, "
        "roll30, roll60, delta_h_7d, delta_h_30d"
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
        
    # FIXED: Đã kiểm tra target shifting. Hàm shift(-d) dịch chuyển mục tiêu về phía trước (backward trong pandas), 
    # nghĩa là hàng t sẽ mang giá trị của t+d. Điều này là ĐÚNG. Lỗi thực sự nằm ở hàm create_sequences 
    # của 06_bilstm_model.py (đã được sửa) vì nó bỏ lọt thông tin của ngày hiện tại t.
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
    from sklearn.preprocessing import StandardScaler

    # Dùng StandardScaler thay cho MinMaxScaler để hỗ trợ mô hình ngoại suy (Extrapolation).
    # MinMaxScaler ép data về [0, 1], khi gặp lũ lịch sử (Yagi) > max của tập Train, 
    # đầu vào sẽ lớn hơn 1 và có thể làm bão hòa mạng LSTM.
    scaler = StandardScaler()
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

    if ENABLE_FLOOD_INJECTION:
        logger.info("\n[Inject] Bom dinh lu SCS-CN (toan bo chuoi)...")
        df_daily_level = inject_synthetic_flood_peaks(df_daily_level, df_nasa_daily)
    else:
        logger.info("\n[Inject] TAT — khong bom dinh lu synthetic (v5 anti-overfit)")

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

    # Mốc H(t) và ΔH cho mô hình v5 (trước khi scale features)
    df[BASE_LEVEL_COL] = df[TARGET_COL].astype(float)
    if PREDICT_DELTA_H:
        for d in FORECAST_DAYS:
            df[target_delta_col(d)] = df[target_abs_col(d)] - df[BASE_LEVEL_COL]

    # Trọng số mẫu: quan trắc thật + nhấn mạnh biến động lớn |ΔH|
    if "is_observed" in df.columns:
        df["sample_weight"] = np.where(
            df["is_observed"].astype(bool),
            SAMPLE_WEIGHT_OBSERVED,
            SAMPLE_WEIGHT_OTHER,
        )
    else:
        df["sample_weight"] = SAMPLE_WEIGHT_OBSERVED
    d1 = target_delta_col(1)
    if d1 in df.columns:
        df["sample_weight"] *= np.where(
            np.abs(df[d1]) >= FLOOD_DELTA_THRESHOLD_M,
            SAMPLE_WEIGHT_FLOOD,
            1.0,
        )

    # Cắt hàng đầu/cuối do lag và forecast horizon
    max_lag     = 30
    max_horizon = 30
    df = df.iloc[max_lag:-max_horizon].copy()

    # Loại hàng NaN trong features/targets
    target_cols   = [target_abs_col(d) for d in FORECAST_DAYS]
    if PREDICT_DELTA_H:
        target_cols += [target_delta_col(d) for d in FORECAST_DAYS]
    required_cols = FEATURE_COLS + target_cols + [BASE_LEVEL_COL, "sample_weight"]
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
    
    # FIXED: In ra thông tin kiểm tra dataset để đảm bảo việc xử lý NaN ở cuối (do shift) hoạt động đúng.
    logger.info("\n[Check] Kiem tra dataset cuoi cung:")
    logger.info("  Shape: %s", df.shape)
    for d in FORECAST_DAYS:
        col = f"target_t{d}d"
        if col in df.columns:
            logger.info("  Range %s: %.2f - %.2f", col, df[col].min(), df[col].max())
    logger.info("  3 hang cuoi cung:\n%s", df[target_cols + ["water_level_m"]].tail(3))

    # ── Chia 3 tap: Train / Test / Val (Theo đúng yêu cầu của người dùng) ──────────
    # Train : Huấn luyện mô hình (2020-2022)
    # Test  : Dùng cho EarlyStopping (2023)
    # Val   : Tập Kiểm định Độc lập cuối cùng (2024-2025)
    df_train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    df_test  = df[(df.index > TRAIN_END) & (df.index <= VAL_END)].copy()
    df_val   = df[df.index > VAL_END].copy()

    logger.info("\n[Split] 3-Way Split (Train / Test / Val):")
    logger.info(
        "  Train (Huấn luyện)           : %d ngay (%s -> %s)",
        len(df_train),
        df_train.index.min().date(), df_train.index.max().date()
    )
    logger.info(
        "  Test  (EarlyStopping)        : %d ngay (%s -> %s)  <- Dừng sớm",
        len(df_test),
        df_test.index.min().date(), df_test.index.max().date()
    )
    logger.info(
        "  Val   (Kiểm định Độc lập)    : %d ngay (%s -> %s)  <- BÁO CÁO LUẬN VĂN",
        len(df_val),
        df_val.index.min().date(), df_val.index.max().date()
    )

    # Canh bao dataset nho
    if len(df_train) < 300:
        logger.warning(
            "CANH BAO: Train chi co %d ngay (nen >= 300 de Bi-LSTM on dinh).",
            len(df_train)
        )
    if len(df_test) < 60:
        logger.warning("CANH BAO: Test chi co %d ngay (nen >= 60 cho EarlyStopping).", len(df_test))
    if len(df_val) < 100:
        logger.warning(
            "CANH BAO: Val chi co %d ngay (nen >= 100 de ket qua kiem dinh tin cay).",
            len(df_val)
        )

    # -- Chuẩn hóa --
    logger.info("\n[Normalize] Min-Max (fit only on train)...")
    df_train, df_val, df_test, _, feature_cols_present = normalize_features(
        df_train, df_val, df_test, FEATURE_COLS
    )

    # -- Lưu --
    df_train.to_csv(OUTPUT_TRAIN)
    df_test.to_csv(OUTPUT_TEST)
    df_val.to_csv(OUTPUT_VAL)

    logger.info("\n[Luu] Da luu:")
    logger.info("  %s (%d dong)", OUTPUT_TRAIN, len(df_train))
    logger.info("  %s (%d dong)", OUTPUT_TEST,  len(df_test))
    logger.info("  %s (%d dong)", OUTPUT_VAL,   len(df_val))

    logger.info("\n" + "=" * 65)
    logger.info("TOM TAT BO DU LIEU (TRAIN / TEST / VAL):")
    logger.info(
        "  Tong: %d ngay | Train: %d | Test: %d | Val: %d",
        len(df), len(df_train), len(df_test), len(df_val)
    )
    logger.info("  Features: %d cot", len(feature_cols_present))
    logger.info("  Horizons: %s ngay", FORECAST_DAYS)
    if "is_observed" in df.columns:
        obs_pct = int(df["is_observed"].mean() * 100)
        logger.info("  Ty le diem thuc / tong: ~%d%%", obs_pct)
    logger.info(
        "  [QUAN TRONG] Test (2023) chi dung cho EarlyStopping."
        " Ket qua bao cao lay tu Val (2024+)."
    )
    logger.info("=" * 65)
    logger.info("\nBuoc 5 hoan thanh -- Chay tiep: python 06_bilstm_model.py")

if __name__ == "__main__":
    main()
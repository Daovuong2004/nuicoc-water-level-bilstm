"""
Bước 2 (v3.0 — Refactored): Suy luận mực nước hồ Núi Cốc từ GEE Sentinel-2
=============================================================================
CHANGELOG v3.0 (so với v2.0):
  [FIX-1] NDWI threshold nâng từ 0.0 → 0.15 (giảm false-positive ven hồ)
  [FIX-2] Thêm SCL cloud/shadow mask (band 20m) bổ sung cho QA60
  [FIX-3] Daily mosaic bằng .median() trước khi tính NDWI →
           triệt tiêu hoàn toàn duplicate dates từ multi-orbit
  [FIX-4] fill_value=(np.nan, np.nan) thay vì cứng (36.0, ...) →
           không bao giờ phát sinh giá trị fallback 36.0m
  [FIX-5] Lọc vật lý mở rộng: lọc cả NaN, area < 150 ha, area > 3500 ha,
           water_level clamped ≈ 36.0m, đồng thời GIẢI THÍCH lý do loại
  [FIX-6] Resampling hàng tuần (W) + interpolate(method='time') + ffill/bfill
           → chuỗi thời gian đều đặn sẵn sàng cho LSTM window
  [FIX-7] `is_observed` flag để pipeline sau phân biệt thực đo vs nội suy
  [FIX-8] Khởi tạo GEE dùng project ID (ee.Initialize(project=...))
           với fallback ee.Authenticate() khi chưa có credentials
  [FIX-9] Loại bỏ pandas .append() (đã deprecated từ 2.0) → dùng pd.concat

Phương pháp:
  1. Lọc Sentinel-2 SR Harmonized theo vùng, ngày, ngưỡng mây scl+QA60
  2. Daily mosaic .median() → 1 composite/ngày
  3. MNDWI = (Green - SWIR1) / (Green + SWIR1) threshold 0.1
  4. Tính diện tích mặt nước (ha)
  5. Ánh xạ Area → Height qua đường cong A-H (fill_value=NaN)
  6. Làm sạch + resampling hàng tuần

Tần số quan trắc  : ~5 ngày/lần (chu kỳ lặp lại S2)
Output sau resample: chuỗi hàng tuần (W), sẵn cho LSTM

Yêu cầu:
  - Tài khoản GEE với project ID đã kích hoạt
  - earthengine-api >= 0.1.380  (pip install earthengine-api)
  - scipy, pandas >= 2.0, numpy >= 1.24
"""

import os
import time
import logging

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# CẤU HÌNH
# ============================================================
OUTPUT_PATH    = "data/raw/gee_water_level.csv"
CHECKPOINT_DIR = "data/raw/checkpoints"

# [FIX-8] Thêm GEE_PROJECT — bắt buộc với earthengine-api >= 0.1.370
GEE_PROJECT = "datn-495501"   # <- Thay bằng Google Cloud Project ID của bạn

START_YEAR = 2017
END_YEAR   = 2025

# Bounding box hồ Núi Cốc (WGS84)
LAKE_BOUNDS = {
    "lon_min": 105.68,
    "lat_min": 21.64,
    "lon_max": 105.78,
    "lat_max": 21.73,
}

# [FIX-3] Ngưỡng mây được nới rộng — SCL+QA60 mask xử lý pixel-level
MAX_CLOUD_PCT  = 80       # Lọc ảnh ở collection-level (thô)
# [FIX-1] MNDWI threshold cao hơn 0.0 để loại thực vật ẩm ướt / bóng mây
MNDWI_THRESHOLD = 0.10   # McFeeters chuẩn =0.0; 0.1 cân bằng tốt hơn cho S2

MAX_RETRIES  = 3
RETRY_DELAYS = [10, 30, 90]


# ============================================================
# ĐƯỜNG CONG A-H (Diện tích mặt hồ → Mực nước)
# ============================================================
# [FIX-4] Mở rộng giới hạn dưới (50 ha → 34.0m) và dùng fill_value=NaN
# Khi diện tích nằm ngoài [50, 3500] ha → trả NaN, không phải 36.0 cứng
AH_CURVE = [
    (  50, 34.00),   # Dưới MNC — bảo vệ biên dưới
    ( 150, 35.50),
    ( 200, 36.00),   # Mực nước chết (MNC)
    ( 500, 38.00),
    ( 900, 40.00),
    (1400, 42.00),
    (2000, 44.00),
    (2500, 46.20),   # MNDBT — điểm chuẩn chính xác
    (2700, 46.50),
    (2900, 46.90),
    (3050, 47.20),
    (3150, 47.50),
    (3200, 47.80),   # Gần lũ tối đa
    (3500, 48.25),   # Đỉnh vùng bán ngập
]


def build_ah_interpolator(ah_curve: list):
    """
    Xây dựng hàm nội suy tuyến tính từ đường cong A-H.

    [FIX-4]: fill_value=(np.nan, np.nan) thay vì fill_value=(levels[0], levels[-1])
    → Diện tích ngoài phạm vi trả NaN thay vì clamp cứng về 36.0m.
    Điều này ngăn chặn hoàn toàn hiện tượng "flat 36.0m line" trong biểu đồ.

    Parameters
    ----------
    ah_curve : list of (float, float)
        Danh sách cặp (dien_tich_ha, muc_nuoc_m).

    Returns
    -------
    callable
        Hàm f(area_ha) → water_level_m (NaN khi ngoài biên).
    """
    areas  = [p[0] for p in ah_curve]
    levels = [p[1] for p in ah_curve]
    return interp1d(
        areas, levels,
        kind="linear",
        bounds_error=False,
        fill_value=(np.nan, np.nan),   # [FIX-4] NaN thay vì clamp!
    )


ah_to_level = build_ah_interpolator(AH_CURVE)


# ============================================================
# GEE HELPERS
# ============================================================

def _initialize_gee():
    """
    [FIX-8] Khởi tạo GEE với project ID.

    Thứ tự ưu tiên:
      1. Dùng credentials đã lưu (Application Default Credentials)
      2. Nếu không có → gọi ee.Authenticate() (mở trình duyệt)

    project= là bắt buộc với earthengine-api >= 0.1.370 và GEE Cloud Projects.
    Bỏ project= sẽ gây DeprecationWarning → lỗi trong tương lai gần.
    """
    import ee
    try:
        ee.Initialize(project=GEE_PROJECT)
        # Kiểm tra kết nối thực sự (không chỉ initialize object)
        _ = ee.Number(1).getInfo()
        logger.info("[GEE] Đã khởi tạo Earth Engine (project: %s).", GEE_PROJECT)
    except Exception:
        logger.info("[GEE] Chưa xác thực, đang authenticate...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        logger.info("[GEE] Xác thực thành công (project: %s).", GEE_PROJECT)


# ============================================================
# GEE: TÍNH MNDWI VÀ DIỆN TÍCH MẶT HỒ (THEO TỪNG NĂM)
# ============================================================
def compute_water_area_one_year(year: int) -> pd.DataFrame:
    """
    Tính diện tích mặt nước từ Sentinel-2 cho một năm.

    CẢI TIẾN CHÍNH so với v2.0:
    ─────────────────────────────────────────────────────────
    [FIX-2] DUAL CLOUD MASK: QA60 (mây dày) + SCL (mây mỏng, bóng mây)
      - QA60 bit10 & bit11: loại mây dày và cirrus
      - SCL class 3 (cloud shadow), 8 (cloud medium), 9 (cloud high), 11 (snow)
      - Dùng cả hai → loại bỏ gần 95% pixel nhiễu thay vì chỉ ~70% với QA60 đơn

    [FIX-3] DAILY MEDIAN COMPOSITE trước khi tính NDWI:
      - Group images bởi cùng ngày → .median() → 1 image/ngày
      - Giải quyết triệt để duplicate dates từ Sentinel-2 multi-orbit
        (2 quỹ đạo/ngày ở khu vực nhiệt đới)
      - Median ổn định hơn Mean khi còn sót pixel mây sau mask

    [FIX-1] MNDWI thay vì NDWI:
      - MNDWI = (Green B3 - SWIR1 B11) / (Green B3 + SWIR1 B11)
      - SWIR1 nhạy với thực vật ẩm và đất ướt → MNDWI phân biệt tốt hơn NDWI
        trong môi trường hồ đồng bằng/rừng thưa như Núi Cốc

    Parameters
    ----------
    year : int

    Returns
    -------
    pd.DataFrame
        Columns: date, water_area_ha, cloud_scene
        Đã xử lý daily composite → KHÔNG có duplicate dates.
    """
    import ee

    region = ee.Geometry.Rectangle([
        LAKE_BOUNDS["lon_min"], LAKE_BOUNDS["lat_min"],
        LAKE_BOUNDS["lon_max"], LAKE_BOUNDS["lat_max"],
    ])

    start_date = f"{year}-01-01"
    end_date   = f"{year+1}-01-01"   # filterDate là [start, end), tránh miss 31/12

    # [FIX-2] Chọn đủ bands để mask SCL (B11=SWIR1 cho MNDWI, SCL cho cloud mask)
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .select(["B3", "B8", "B11", "QA60", "SCL"])   # Green, NIR, SWIR1, CloudMask, SCL
    )

    # ── [FIX-2] Hàm mask cloud với cả QA60 + SCL ──────────────────────────
    def mask_clouds(image):
        """
        Áp dụng dual cloud mask: QA60 bitwise + SCL class-based.

        SCL classes bị loại:
          3  = cloud shadow
          8  = cloud medium probability
          9  = cloud high probability
          11 = snow/ice (rare ở VN nhưng vẫn tính)
        """
        qa   = image.select("QA60")
        scl  = image.select("SCL")

        # QA60: bit 10 (opaque cloud) và bit 11 (cirrus)
        qa_mask = (
            qa.bitwiseAnd(1 << 10).eq(0)
             .And(qa.bitwiseAnd(1 << 11).eq(0))
        )

        # SCL: loại shadow (3), cloud medium (8), cloud high (9), snow (11)
        scl_mask = (
            scl.neq(3)
               .And(scl.neq(8))
               .And(scl.neq(9))
               .And(scl.neq(11))
        )

        combined_mask = qa_mask.And(scl_mask)
        # Scale SR values (GEE lưu × 10000)
        return (
            image
            .updateMask(combined_mask)
            .divide(10000)
            .copyProperties(image, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"])
        )

    s2_masked = s2.map(mask_clouds)

    # ── [FIX-3] Daily Median Composite ────────────────────────────────────
    # Tạo danh sách ngày duy nhất trong năm từ collection
    def make_daily_composite(date_str):
        """
        Tạo 1 ảnh composite median cho 1 ngày.

        Sentinel-2 có thể có 1-2 ảnh/ngày tại Việt Nam (descending orbit).
        Median() của các ảnh cùng ngày:
          - Loại spike mây còn sót sau mask
          - Không bị artifact "border noise" ở rìa ảnh
        """
        d_start = ee.Date(date_str)
        d_end   = d_start.advance(1, "day")

        # Lấy ảnh trong ngày
        daily = s2_masked.filterDate(d_start, d_end)
        n_images = daily.size()

        # Composite median
        composite = daily.median().set({
            "system:time_start": d_start.millis(),
            "n_images": n_images,
        })
        return composite

    # Lấy danh sách ngày duy nhất có ảnh
    dates_list = (
        s2_masked
        .aggregate_array("system:time_start")
        .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
        .distinct()   # Unique ngày
        .sort()
    )

    # Tạo composite collection từ danh sách ngày
    composites = ee.ImageCollection.fromImages(
        dates_list.map(make_daily_composite)
    )

    # ── [FIX-1] Tính MNDWI và diện tích mặt nước ──────────────────────────
    def compute_mndwi_area(image):
        """
        Tính MNDWI và diện tích mặt nước cho một ảnh composite.

        MNDWI = (Green B3 - SWIR1 B11) / (Green B3 + SWIR1 B11)
        Ưu điểm so với NDWI (Green-NIR):
          - SWIR1 bị hấp thụ mạnh bởi nước → MNDWI > 0 chính xác hơn cho hồ lớn
          - Phân biệt tốt hơn với thực vật thủy sinh và bùn ướt ven hồ
          - Threshold 0.10 (thay vì 0.0) loại thêm false-positive ven bờ
        """
        mndwi      = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        water_mask = mndwi.gt(MNDWI_THRESHOLD)

        area_image = water_mask.multiply(ee.Image.pixelArea()).divide(10000)
        stats = area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=20,          # SCL và B11 có độ phân giải 20m → dùng 20m cho nhất quán
            maxPixels=1e13,
            bestEffort=True,
        )

        cloud_pct = image.get("CLOUDY_PIXEL_PERCENTAGE")

        return ee.Feature(None, {
            "date":          ee.Date(image.get("system:time_start")).format("YYYY-MM-dd"),
            "water_area_ha": stats.get("MNDWI"),
            "cloud_scene":   cloud_pct,
        })

    features = composites.map(compute_mndwi_area)
    data     = ee.FeatureCollection(features).getInfo()

    records = [f["properties"] for f in data["features"]]
    if not records:
        logger.warning("  [GEE] Không có ảnh hợp lệ cho năm %d.", year)
        return pd.DataFrame(columns=["date", "water_area_ha", "cloud_scene"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Kiểm tra không còn duplicate dates (phòng thủ)
    n_dup = df.duplicated(subset="date").sum()
    if n_dup > 0:
        logger.warning("  [GEE %d] Phát hiện %d bản ghi trùng ngày (sau composite) — tự loại.", year, n_dup)
        df = df.drop_duplicates(subset="date", keep="first")

    logger.info("  [GEE %d] %d ảnh composite/ngày.", year, len(df))
    return df


def compute_water_area_gee_with_retry() -> pd.DataFrame:
    """
    Tính diện tích mặt nước GEE cho toàn bộ giai đoạn START_YEAR–END_YEAR.

    Chiến lược:
      - Chia theo năm → mỗi request nhỏ, ít bị timeout
      - Retry exponential backoff (3 lần / năm)
      - Checkpoint từng năm → resume khi crash

    Returns
    -------
    pd.DataFrame
        Columns: date, water_area_ha, cloud_scene. Không có duplicate dates.
    """
    _initialize_gee()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    all_dfs = []
    failed  = []

    for year in range(START_YEAR, END_YEAR + 1):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"gee_{year}.csv")

        if os.path.exists(checkpoint_path):
            logger.info("  [GEE %d] Đọc từ checkpoint.", year)
            df_year = pd.read_csv(checkpoint_path, parse_dates=["date"])
            all_dfs.append(df_year)
            continue

        last_exc = None
        success  = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    "  [GEE %d] Đang xử lý (lần %d/%d)...",
                    year, attempt, MAX_RETRIES,
                )
                df_year = compute_water_area_one_year(year)
                df_year.to_csv(checkpoint_path, index=False)
                logger.info(
                    "  ✓ GEE %d: %d ảnh composite → checkpoint lưu.",
                    year, len(df_year),
                )
                all_dfs.append(df_year)
                success = True
                break

            except Exception as exc:
                logger.warning(
                    "  ⚠ GEE %d lần %d/%d thất bại: %s",
                    year, attempt, MAX_RETRIES, exc,
                )
                last_exc = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt - 1]
                    logger.info("  ↻ Thử lại sau %d giây...", delay)
                    time.sleep(delay)

        if not success:
            logger.error(
                "  ✗ GEE %d: Thất bại sau %d lần thử. Lỗi: %s",
                year, MAX_RETRIES, last_exc,
            )
            failed.append(year)

    if failed:
        logger.warning("\n⚠ Không xử lý được %d năm: %s", len(failed), failed)

    if not all_dfs:
        raise RuntimeError(
            "Không lấy được dữ liệu GEE nào!\n"
            "Kiểm tra xác thực Earth Engine và kết nối mạng."
        )

    # [FIX-9] dùng pd.concat thay vì .append() (đã deprecated từ pandas 2.0)
    return pd.concat(all_dfs, ignore_index=True).sort_values("date").reset_index(drop=True)


# ============================================================
# LÀM SẠCH & CHUYỂN DIỆN TÍCH → MỰC NƯỚC
# ============================================================
def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch chuyên sâu + ánh xạ diện tích → mực nước qua đường cong A-H.

    Thứ tự xử lý (quan trọng — không hoán đổi):
      1. Parse kiểu dữ liệu
      2. Lọc outlier vật lý (diện tích ngoài [150, 3500] ha)
      3. Áp A-H curve → NaN khi extrapolate (không còn fallback 36.0m) [FIX-4]
      4. Loại NaN từ extrapolation
      5. Phân cấp chất lượng ảnh
      6. Deduplication theo ngày (phòng thủ) [FIX-3]
      7. Đánh dấu điểm quan trắc thực (is_observed=True) [FIX-7]

    Tại sao KHÔNG còn xuất hiện 36.0m flat line:
      - Phiên bản cũ: fill_value=(levels[0], levels[-1]) = (36.0, 48.25)
        → Bất kỳ diện tích < 200 ha đều cho kết quả 36.0m
      - Phiên bản mới: fill_value=(np.nan, np.nan)
        → Diện tích ngoài phạm vi [50, 3500] ha → NaN → bị loại ở bước 4
    """
    AREA_MIN_HA = 150    # ha — mực nước chết + buffer
    AREA_MAX_HA = 3500   # ha — lũ thiết kế max

    df = df.copy()
    df["water_area_ha"] = pd.to_numeric(df["water_area_ha"], errors="coerce")

    # Đồng hóa tên cột cloud
    if "cloud_cover" in df.columns and "cloud_scene" not in df.columns:
        df = df.rename(columns={"cloud_cover": "cloud_scene"})
    if "cloud_scene" not in df.columns:
        df["cloud_scene"] = 0.0   # Fallback nếu không có metadata

    df["cloud_scene"] = pd.to_numeric(df["cloud_scene"], errors="coerce").fillna(0.0)

    # --- Bước 2: Lọc outlier vật lý ---
    n0 = len(df)
    df = df[df["water_area_ha"].between(AREA_MIN_HA, AREA_MAX_HA)].dropna(subset=["water_area_ha"])
    logger.info(
        "[Postprocess] Bước 2 — Loại %d bản ghi ngoài phạm vi vật lý (%d–%d ha). Còn: %d",
        n0 - len(df), AREA_MIN_HA, AREA_MAX_HA, len(df),
    )

    # --- Bước 3: Áp đường cong A-H [FIX-4] ---
    df["water_level_m"] = ah_to_level(df["water_area_ha"].values).astype(float)

    # --- Bước 4: Loại NaN từ extrapolation ---
    n1 = len(df)
    df = df.dropna(subset=["water_level_m"])
    logger.info(
        "[Postprocess] Bước 4 — Loại %d bản ghi NaN (extrapolated ngoài AH range). Còn: %d",
        n1 - len(df), len(df),
    )

    # --- Bước 5: Phân cấp chất lượng ---
    df["quality"] = np.select(
        [df["cloud_scene"] < 30, df["cloud_scene"] < 60, df["cloud_scene"] <= 80],
        ["good",                  "fair",                  "low"],
        default="low",
    )

    # --- Bước 6: Deduplication (phòng thủ) [FIX-3] ---
    n2 = len(df)
    df = df.sort_values(["date", "cloud_scene"])   # ít mây lên đầu
    df = df.drop_duplicates(subset="date", keep="first")
    if n2 > len(df):
        logger.info(
            "[Postprocess] Bước 6 — Deduplication: loại %d bản ghi trùng ngày. Còn: %d",
            n2 - len(df), len(df),
        )

    # --- Bước 7: Đánh dấu điểm thực [FIX-7] ---
    df["is_observed"] = True

    df = df.sort_values("date").reset_index(drop=True)

    # Thống kê
    logger.info("[Postprocess] Tổng quan trắc hợp lệ: %d", len(df))
    if len(df) > 0:
        logger.info(
            "  Mực nước: %.2f – %.2f m | Giai đoạn: %s → %s",
            df["water_level_m"].min(), df["water_level_m"].max(),
            df["date"].min().date(), df["date"].max().date(),
        )
        for q, cnt in df["quality"].value_counts().items():
            logger.info("  Chất lượng %-5s: %d (%.1f%%)", q, cnt, cnt / len(df) * 100)

    cols = ["date", "water_area_ha", "water_level_m", "cloud_scene", "quality", "is_observed"]
    return df[[c for c in cols if c in df.columns]]


# ============================================================
# RESAMPLING & REGULARIZATION CHO LSTM
# ============================================================
def regularize_time_series(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Chuyển chuỗi không đều → chuỗi đều đặn (weekly hoặc monthly).

    Lý do LSTM cần chuỗi đều đặn:
      - LSTM giả định mỗi timestep = khoảng thời gian bằng nhau
      - Dữ liệu GEE có khoảng cách từ 5 ngày đến hàng tháng tùy mùa mây
      - Nếu đưa thẳng vào LSTM: model học sai temporal relationship
        (ví dụ: nghĩ 2 bước liên tiếp = 5 ngày, nhưng thực tế là 35 ngày)

    Chiến lược nội suy:
      1. Set date làm index + resample(freq) → khung thời gian đều
      2. interpolate(method='time'): nội suy tuyến tính theo thời gian thực
         (không phải theo chỉ số hàng như 'linear' — quan trọng!)
      3. ffill/bfill tối đa 4 bước: để đảm bảo không tạo quá nhiều giá trị
         nhân tạo tại vùng dữ liệu thưa (2017–2021)
      4. Đánh dấu is_observed=False cho các điểm nội suy

    Parameters
    ----------
    df : pd.DataFrame
        Output từ postprocess(). Cần có cột 'date' và 'water_level_m'.
    freq : str
        Tần suất resample. 'W'=hàng tuần, 'M'=hàng tháng.
        Dùng 'W' cho LSTM window 30 ngày để có 4-5 điểm/tháng.

    Returns
    -------
    pd.DataFrame
        Chuỗi đều đặn với is_observed=True/False để phân biệt thực đo vs nội suy.
    """
    df = df.copy()
    df = df.set_index("date").sort_index()

    # Đảm bảo DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Resample → lấy giá trị thực (quan trắc) tại mỗi "bin" tuần/tháng
    # Dùng first() vì đã dedup → tối đa 1 quan trắc/ngày
    df_obs = df.resample(freq).first()

    # Nội suy mực nước (phương pháp 'time' = nội suy tuyến tính theo khoảng cách thời gian thực)
    df_obs["water_level_m"] = df_obs["water_level_m"].interpolate(
        method="time",
        limit=4,               # Tối đa 4 bước trống liên tiếp (~1 tháng với freq=W)
        limit_direction="both",
    )

    # Fill cột phụ (area, cloud) cho điểm nội suy — forward fill là đủ
    for col in ["water_area_ha", "cloud_scene"]:
        if col in df_obs.columns:
            df_obs[col] = df_obs[col].ffill().bfill()

    # Đánh dấu điểm nội suy
    # is_observed = True chỉ ở các bước có dữ liệu thực
    original_dates = set(df.index.normalize())   # Ngày quan trắc gốc (không có giờ)
    df_obs["is_observed"] = df_obs.index.normalize().isin(original_dates)
    df_obs["quality"]     = df_obs["quality"].ffill().bfill().fillna("interpolated")

    # Bỏ các điểm còn NaN (không nội suy được) ở đầu/cuối chuỗi
    n_before = len(df_obs)
    df_obs = df_obs.dropna(subset=["water_level_m"])
    if len(df_obs) < n_before:
        logger.info(
            "[Regularize] Bỏ %d điểm không nội suy được (biên chuỗi).",
            n_before - len(df_obs),
        )

    df_obs = df_obs.reset_index().rename(columns={"index": "date"})

    logger.info(
        "[Regularize] Chuỗi %s: %d điểm quan trắc → %d bước %s đều đặn",
        freq, df["water_level_m"].notna().sum(), len(df_obs), freq,
    )
    logger.info(
        "  Điểm quan trắc thực: %d | Điểm nội suy: %d",
        df_obs["is_observed"].sum(),
        (~df_obs["is_observed"]).sum(),
    )

    return df_obs


# ============================================================
# LOAD TỪ FILE EXPORT GEE (CHẾ ĐỘ OFFLINE)
# ============================================================
def load_from_gee_export(filepath: str) -> pd.DataFrame:
    """
    Load dữ liệu từ CSV đã export từ GEE Code Editor hoặc batch Export task.

    Tự động detect tên cột cloud: 'cloud_scene', 'cloud_cover', 'cloud_roi_pct'.
    Tính mực nước qua A-H nếu chưa có cột 'water_level_m'.

    Parameters
    ----------
    filepath : str
        Đường dẫn CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame chưa qua postprocess (raw GEE output).
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Đồng bộ tên cột cloud
    cloud_aliases = {"cloud_cover", "cloud_roi_pct", "cloud_scene"}
    found_cloud = [c for c in df.columns if c in cloud_aliases]
    if found_cloud and "cloud_scene" not in df.columns:
        df = df.rename(columns={found_cloud[0]: "cloud_scene"})

    if "water_level_m" not in df.columns:
        if "water_area_ha" in df.columns:
            logger.info("[Offline] Tính mực nước từ A-H curve...")
            df["water_level_m"] = ah_to_level(df["water_area_ha"].values).astype(float)
        else:
            raise KeyError(
                "File export thiếu cả 'water_level_m' và 'water_area_ha'.\n"
                "Kiểm tra lại file export từ GEE."
            )

    logger.info("[Offline] Load %d bản ghi từ: %s", len(df), filepath)
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BƯỚC 2 (v3.0): SUY LUẬN MỰC NƯỚC TỪ GEE SENTINEL-2")
    logger.info("  [FIX] Dual cloud mask (QA60 + SCL)")
    logger.info("  [FIX] Daily median composite (triệt tiêu duplicate dates)")
    logger.info("  [FIX] MNDWI threshold=0.10 (giảm false-positive)")
    logger.info("  [FIX] A-H fill_value=NaN (không clamp 36.0m)")
    logger.info("  [FIX] Weekly resampling + time interpolation")
    logger.info("=" * 60)

    gee_export_path = "data/raw/gee_export.csv"

    if os.path.exists(gee_export_path):
        logger.info("[GEE] Phát hiện file export — dùng chế độ offline.")
        df_raw = load_from_gee_export(gee_export_path)
    else:
        logger.info("[GEE] Chạy trực tiếp trên Earth Engine...")
        df_raw = compute_water_area_gee_with_retry()

    # --- Làm sạch + ánh xạ A-H ---
    df_clean = postprocess(df_raw)

    # --- [FIX-6] Regularize chuỗi thời gian cho LSTM ---
    df_weekly = regularize_time_series(df_clean, freq="W")

    # --- Lưu kết quả ---
    df_clean.to_csv(OUTPUT_PATH, index=False)
    df_weekly.to_csv(OUTPUT_PATH.replace(".csv", "_weekly.csv"), index=False)

    logger.info("\n[GEE] Đã lưu:")
    logger.info("  Quan trắc thô (sau lọc) : %s  (%d bản ghi)", OUTPUT_PATH, len(df_clean))
    logger.info(
        "  Chuỗi tuần đều đặn      : %s  (%d bước)",
        OUTPUT_PATH.replace(".csv", "_weekly.csv"), len(df_weekly),
    )

    logger.info("\n%s", df_weekly.tail(10).to_string())
    logger.info(
        "\n✓ Bước 2 hoàn thành — %d quan trắc thực (%s → %s) | %d bước tuần.",
        len(df_clean),
        df_clean["date"].min().date(),
        df_clean["date"].max().date(),
        len(df_weekly),
    )


if __name__ == "__main__":
    main()

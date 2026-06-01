"""
Bước 2: Suy luận mực nước hồ Núi Cốc từ GEE Sentinel-2
=========================================================
Phương pháp:
  1. Tính chỉ số NDWI = (Green - NIR) / (Green + NIR) từ Sentinel-2
  2. Phân loại pixel mặt nước (NDWI > 0) → diện tích mặt hồ (ha)
  3. Ánh xạ diện tích → mực nước qua đường cong A-H (nội suy tuyến tính)

Tần số quan trắc : ~5 ngày/lần (chu kỳ lặp lại Sentinel-2)
Số quan trắc ước tính : ~730 điểm (2017–2025)

Yêu cầu:
  - Tài khoản Google Earth Engine đã được xác thực
  - Hoặc file gee_export.csv từ GEE Code Editor (chạy offline)

Cải tiến v2.0:
  - Chia request GEE theo từng năm → tránh timeout với dataset lớn
  - Thêm retry cho từng năm (3 lần, exponential backoff)
  - Lưu checkpoint từng năm → có thể resume nếu crash
  - Thêm progress logging chi tiết
"""

import os
import time
import logging

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Cấu hình logging
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

START_YEAR = 2017  # Sentinel-2 bắt đầu từ 2017
END_YEAR   = 2025

# Bounding box hồ Núi Cốc (WGS84)
LAKE_BOUNDS = {
    "lon_min": 105.68,
    "lat_min": 21.64,
    "lon_max": 105.78,
    "lat_max": 21.73,
}

# Lọc chất lượng ảnh Sentinel-2
MAX_CLOUD_PERCENT = 15    # Loại bỏ ảnh mây > 15%

# Cấu hình retry cho GEE
MAX_RETRIES  = 3
RETRY_DELAYS = [10, 30, 90]   # GEE cần thời gian chờ lâu hơn NASA API

# ============================================================
# ĐƯỜNG CONG A-H (Diện tích mặt hồ → Mực nước)
# ============================================================
# Xây dựng từ DEM và số liệu kỹ thuật hồ Núi Cốc:
#   - Mực nước dâng bình thường (MNDBT) = +46.20m → ~2.500 ha
#   - Dung tích toàn bộ              = 175,5 triệu m³
#   - Lũ tối đa thiết kế             → ~3.200 ha
# Format: (dien_tich_ha, muc_nuoc_m)
# !! Cần thay bằng đường cong A-H đo đạc thực tế từ DEM chính xác !!
AH_CURVE = [
    (200,   36.00),   # Gần mực nước chết (ước lượng từ địa hình)
    (500,   38.00),
    (900,   40.00),
    (1400,  42.00),
    (2000,  44.00),
    (2500,  46.20),   # MNDBT — điểm chuẩn chính xác
    (2700,  46.50),
    (2900,  46.90),
    (3050,  47.20),
    (3150,  47.50),
    (3200,  47.80),   # Gần lũ tối đa
    (3500,  48.25),   # Đỉnh vùng bán ngập
]


# ============================================================
# ĐƯỜNG CONG A-H — XÂY DỰNG HÀM NỘI SUY
# ============================================================
def build_ah_interpolator(ah_curve: list):
    """
    Xây dựng hàm nội suy tuyến tính từ đường cong A-H.

    Sử dụng scipy.interpolate.interp1d với ngoại suy hằng số
    (fill_value) để xử lý giá trị ngoài phạm vi đường cong.

    Parameters
    ----------
    ah_curve : list of (float, float)
        Danh sách cặp (diện_tích_ha, mực_nước_m).

    Returns
    -------
    callable
        Hàm f(area_ha) → water_level_m.
    """
    areas  = [p[0] for p in ah_curve]
    levels = [p[1] for p in ah_curve]
    return interp1d(
        areas, levels,
        kind="linear",
        bounds_error=False,
        fill_value=(levels[0], levels[-1]),   # Ngoại suy hằng số tại biên
    )


# Hàm nội suy toàn cục
ah_to_level = build_ah_interpolator(AH_CURVE)


# ============================================================
# GEE: TÍNH NDWI VÀ DIỆN TÍCH MẶT HỒ (THEO TỪNG NĂM)
# ============================================================
def compute_water_area_one_year(year: int) -> pd.DataFrame:
    """
    Tính diện tích mặt nước từ Sentinel-2 cho một năm cụ thể.

    Chia theo năm để tránh GEE timeout khi xử lý chuỗi 8 năm (2017–2025).
    Mỗi lần gọi `fc.getInfo()` chỉ xử lý ~70 ảnh thay vì ~730 ảnh.

    Parameters
    ----------
    year : int
        Năm cần xử lý.

    Returns
    -------
    pd.DataFrame
        DataFrame với các cột: date, water_area_ha, cloud_cover.
    """
    import ee  # Import ở đây để tránh lỗi nếu không có GEE

    region = ee.Geometry.Rectangle([
        LAKE_BOUNDS["lon_min"], LAKE_BOUNDS["lat_min"],
        LAKE_BOUNDS["lon_max"], LAKE_BOUNDS["lat_max"],
    ])

    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"

    # Load Sentinel-2 SR Harmonized, lọc theo vùng, thời gian, và mây
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PERCENT))
        .select(["B3", "B8", "QA60"])   # Green, NIR, Cloud mask
    )

    def compute_ndwi_area(image):
        """Hàm ánh xạ — tính NDWI và diện tích mặt nước cho một ảnh."""
        # NDWI = (Green - NIR) / (Green + NIR)
        # NDWI > 0 → mặt nước
        ndwi       = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
        water_mask = ndwi.gt(0)

        # Tính diện tích mặt nước (ha) — nhân với pixel area rồi chia 10000
        area = water_mask.multiply(ee.Image.pixelArea()).divide(10000)
        water_area = area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=10,           # Độ phân giải 10m của Sentinel-2
            maxPixels=1e9,
        )

        return ee.Feature(None, {
            "date":          ee.Date(image.get("system:time_start")).format("YYYY-MM-dd"),
            "water_area_ha": water_area.get("NDWI"),
            "cloud_cover":   image.get("CLOUDY_PIXEL_PERCENTAGE"),
        })

    # Ánh xạ hàm lên toàn bộ ảnh trong năm
    features = s2.map(compute_ndwi_area)
    data     = ee.FeatureCollection(features).getInfo()

    records = [f["properties"] for f in data["features"]]
    if not records:
        logger.warning("  [GEE] Không có ảnh hợp lệ cho năm %d.", year)
        return pd.DataFrame(columns=["date", "water_area_ha", "cloud_cover"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_water_area_gee_with_retry() -> pd.DataFrame:
    """
    Tính diện tích mặt nước GEE cho toàn bộ giai đoạn 2017–2025.

    Chiến lược:
      - Chia theo năm → mỗi request nhỏ, ít bị timeout
      - Retry exponential backoff (3 lần / năm)
      - Checkpoint từng năm → resume khi crash

    Returns
    -------
    pd.DataFrame
        DataFrame tổng hợp với các cột: date, water_area_ha, cloud_cover.
    """
    import ee

    # Xác thực GEE
    try:
        ee.Initialize()
        logger.info("[GEE] Đã khởi tạo Earth Engine.")
    except Exception:
        logger.info("[GEE] Chưa xác thực, đang authenticate...")
        ee.Authenticate()
        ee.Initialize()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    all_dfs = []
    failed  = []

    for year in range(START_YEAR, END_YEAR + 1):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"gee_{year}.csv")

        # Đọc checkpoint nếu đã có
        if os.path.exists(checkpoint_path):
            logger.info("  [GEE %d] Đọc từ checkpoint.", year)
            df_year = pd.read_csv(checkpoint_path, parse_dates=["date"])
            all_dfs.append(df_year)
            continue

        # Thử lấy dữ liệu với retry
        last_exc = None
        success  = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    "  [GEE %d] Đang xử lý (lần %d/%d)...",
                    year, attempt, MAX_RETRIES,
                )
                df_year = compute_water_area_one_year(year)

                # Lưu checkpoint
                df_year.to_csv(checkpoint_path, index=False)
                logger.info(
                    "  ✓ GEE %d: %d ảnh → lưu checkpoint.",
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
        logger.warning(
            "\n⚠ Không xử lý được %d năm GEE: %s", len(failed), failed
        )

    if not all_dfs:
        raise RuntimeError(
            "Không lấy được dữ liệu GEE nào!\n"
            "Kiểm tra xác thực Earth Engine và kết nối mạng."
        )

    return pd.concat(all_dfs, ignore_index=True).sort_values("date")


# ============================================================
# CHUYỂN DIỆN TÍCH → MỰC NƯỚC QUA ĐƯỜNG CONG A-H
# ============================================================
def area_to_water_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng đường cong A-H để chuyển diện tích mặt hồ → mực nước.

    Lọc các quan trắc không hợp lệ:
      - Diện tích < 100 ha (nhiễu, không phải hồ Núi Cốc)
      - Mây > MAX_CLOUD_PERCENT (ảnh chất lượng kém)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame với cột 'water_area_ha' và 'cloud_cover'.

    Returns
    -------
    pd.DataFrame
        DataFrame đã lọc với cột 'water_level_m' và 'quality'.
    """
    df = df.copy()
    df["water_area_ha"] = pd.to_numeric(df["water_area_ha"], errors="coerce")

    # Loại bỏ outlier vật lý (diện tích ngoài phạm vi hồ Núi Cốc)
    n_before = len(df)
    df = df[df["water_area_ha"].between(150, 3500)].copy()
    logger.info(
        "[GEE] Loại %d quan trắc diện tích ngoài phạm vi (150-3500 ha).", n_before - len(df)
    )

    # Loại bỏ ảnh mây nhiều
    n_before = len(df)
    df = df[df["cloud_cover"] < MAX_CLOUD_PERCENT].copy()
    logger.info(
        "[GEE] Loại %d quan trắc do mây >= %d%%.", n_before - len(df), MAX_CLOUD_PERCENT
    )

    # Áp dụng đường cong A-H (vectorized)
    df["water_level_m"] = ah_to_level(df["water_area_ha"].values)

    # Loại bỏ mực nước fallback 36.0m (diện tích quá nhỏ, không tin cậy)
    n_before = len(df)
    df = df[df["water_level_m"].round(6) != 36.0].copy()
    logger.info(
        "[GEE] Loại %d quan trắc có mực nước fallback 36.0m.", n_before - len(df)
    )

    # Deduplication theo ngày (giữ ảnh ít mây nhất mỗi ngày)
    n_before = len(df)
    df = df.sort_values(["date", "cloud_cover"])
    df = df.drop_duplicates(subset="date", keep="first").copy()
    logger.info(
        "[GEE] Khử trùng ngày: loại %d bản ghi trùng.", n_before - len(df)
    )

    # Phân cấp chất lượng ảnh
    df["quality"] = np.where(df["cloud_cover"] < 10, "good", "fair")

    logger.info("[GEE] Tổng quan trắc hợp lệ sau lọc: %d", len(df))
    logger.info(
        "  Chất lượng tốt (mây < 10%%): %d", (df["quality"] == "good").sum()
    )
    logger.info(
        "  Chất lượng trung bình (10–15%%): %d",
        (df["quality"] == "fair").sum(),
    )
    if len(df) > 0:
        logger.info(
            "  Mực nước: %.2f – %.2f m",
            df["water_level_m"].min(), df["water_level_m"].max(),
        )

    return df[["date", "water_area_ha", "water_level_m",
               "cloud_cover", "quality"]]


# ============================================================
# LOAD TỪ FILE EXPORT GEE (CHẾ ĐỘ OFFLINE)
# ============================================================
def load_from_gee_export(filepath: str) -> pd.DataFrame:
    """
    Load dữ liệu từ CSV đã export từ GEE Code Editor.

    Dùng khi không muốn chạy GEE trực tiếp từ Python
    (ví dụ: đã export CSV từ GEE JavaScript API trước đó).

    Parameters
    ----------
    filepath : str
        Đường dẫn đến file CSV export từ GEE.

    Returns
    -------
    pd.DataFrame
        DataFrame đã xử lý với đầy đủ các cột.
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Đồng bộ hóa tên cột mây từ file export
    if "cloud_scene" in df.columns and "cloud_cover" not in df.columns:
        df["cloud_cover"] = df["cloud_scene"]
    elif "cloud_roi_pct" in df.columns and "cloud_cover" not in df.columns:
        df["cloud_cover"] = df["cloud_roi_pct"]

    # Tính mực nước nếu chưa có (chỉ có diện tích)
    if "water_level_m" not in df.columns:
        if "water_area_ha" in df.columns:
            df["water_level_m"] = ah_to_level(df["water_area_ha"].values)
        else:
            raise KeyError(
                "File export thiếu cả 'water_level_m' và 'water_area_ha'.\n"
                "Kiểm tra lại file export từ GEE."
            )

    logger.info("[GEE] Load từ file export: %d quan trắc.", len(df))
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BƯỚC 2: SUY LUẬN MỰC NƯỚC TỪ GEE SENTINEL-2")
    logger.info("=" * 60)

    gee_export_path = "data/raw/gee_export.csv"

    if os.path.exists(gee_export_path):
        # Chế độ offline: đọc từ file đã export
        logger.info("[GEE] Phát hiện file export sẵn có — dùng chế độ offline.")
        df_raw = load_from_gee_export(gee_export_path)
        # Áp dụng bộ lọc chất lượng
        df = area_to_water_level(df_raw)
    else:
        # Chế độ online: chạy trực tiếp trên GEE
        logger.info("[GEE] Chạy trực tiếp trên Earth Engine (theo năm)...")
        df_raw = compute_water_area_gee_with_retry()
        df = area_to_water_level(df_raw)

    # Lưu kết quả
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("\n[GEE] Đã lưu: %s", OUTPUT_PATH)
    logger.info("\n%s", df.tail(10).to_string())

    logger.info(
        "\n✓ Bước 2 hoàn thành — %d quan trắc mực nước từ %s đến %s.",
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )


if __name__ == "__main__":
    main()

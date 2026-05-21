"""
Bước 1: Thu thập dữ liệu khí tượng từ NASA POWER API
======================================================
Tham số thu thập:
  - PRECTOTCORR : Lượng mưa hiệu chỉnh (mm/giờ)
  - T2M         : Nhiệt độ không khí 2m (°C)
  - RH2M        : Độ ẩm tương đối 2m (%)
  - WS2M        : Tốc độ gió 2m (m/s)

Giai đoạn : 2020-01-01 → 2025-12-31
Tọa độ    : Hồ Núi Cốc, Thái Nguyên (21.6833°N, 105.7167°E)
Tần số    : Giờ (hourly)

Cải tiến v2.0:
  - Thêm exponential backoff retry (3 lần, delay 5→15→45s)
  - Lưu checkpoint từng năm → có thể resume nếu crash giữa chừng
  - Cải thiện error logging và xử lý partial failure
"""

import os
import time
import logging

import numpy as np
import pandas as pd
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError

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
LAT = 21.6833       # Vĩ độ hồ Núi Cốc
LON = 105.7167      # Kinh độ hồ Núi Cốc
START_YEAR = 2020
END_YEAR   = 2025

OUTPUT_PATH      = "data/raw/nasa_power_hourly.csv"
CHECKPOINT_DIR   = "data/raw/checkpoints"  # Lưu từng năm để resume

NASA_PARAMS = [
    "PRECTOTCORR",   # Lượng mưa hiệu chỉnh (mm/giờ)
    "T2M",           # Nhiệt độ không khí 2m (°C)
    "RH2M",          # Độ ẩm tương đối 2m (%)
    "WS2M",          # Tốc độ gió 2m (m/s)
]

# Cấu hình retry
MAX_RETRIES    = 3          # Số lần thử lại tối đa
RETRY_DELAYS   = [5, 15, 45]  # Thời gian chờ theo lũy thừa (giây)
REQUEST_TIMEOUT = 120       # Timeout mỗi request (giây)


# ============================================================
# HÀM GỌI API VỚI RETRY
# ============================================================
def fetch_nasa_power(
    lat: float,
    lon: float,
    start: str,
    end: str,
    params: list,
    community: str = "RE",
) -> dict:
    """
    Gọi NASA POWER Hourly API với cơ chế retry exponential backoff.

    Xử lý các lỗi phổ biến:
      - Timeout (kết nối chậm / server quá tải)
      - ConnectionError (mạng không ổn định)
      - HTTP 429 / 503 (rate limit hoặc server quá tải)

    Parameters
    ----------
    lat, lon : float
        Tọa độ điểm quan trắc.
    start, end : str
        Ngày bắt đầu / kết thúc theo định dạng "YYYYMMDD".
    params : list of str
        Danh sách tham số NASA POWER cần lấy.
    community : str
        Cộng đồng người dùng NASA POWER ("RE" = Renewable Energy).

    Returns
    -------
    dict
        Dữ liệu JSON từ NASA POWER API.

    Raises
    ------
    RuntimeError
        Nếu tất cả lần thử đều thất bại.
    """
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    payload = {
        "parameters":    ",".join(params),
        "community":     community,
        "longitude":     lon,
        "latitude":      lat,
        "start":         start,
        "end":           end,
        "format":        "JSON",
        "time-standard": "UTC",
    }

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "[NASA POWER] Tải %s → %s (lần %d/%d)...",
                start, end, attempt, MAX_RETRIES,
            )
            response = requests.get(url, params=payload,
                                    timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()

        except Timeout as exc:
            logger.warning("  ⚠ Timeout sau %ds (lần %d/%d).",
                           REQUEST_TIMEOUT, attempt, MAX_RETRIES)
            last_exc = exc

        except ConnectionError as exc:
            logger.warning("  ⚠ Lỗi kết nối mạng (lần %d/%d): %s",
                           attempt, MAX_RETRIES, exc)
            last_exc = exc

        except HTTPError as exc:
            status = exc.response.status_code
            if status in (429, 503):
                logger.warning("  ⚠ HTTP %d — Server quá tải (lần %d/%d).",
                               status, attempt, MAX_RETRIES)
                last_exc = exc
            else:
                # HTTP 4xx khác (ví dụ 400 Bad Request) → không retry
                raise RuntimeError(
                    f"HTTP {status} — Lỗi request, không thử lại: {exc}"
                ) from exc

        # Chờ trước khi thử lại (exponential backoff)
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[attempt - 1]
            logger.info("  ↻ Thử lại sau %d giây...", delay)
            time.sleep(delay)

    raise RuntimeError(
        f"Không lấy được dữ liệu {start}→{end} sau {MAX_RETRIES} lần thử. "
        f"Lỗi cuối: {last_exc}"
    )


def parse_nasa_response(data: dict, params: list) -> pd.DataFrame:
    """
    Chuyển đổi JSON response từ NASA POWER thành DataFrame.

    Định dạng timestamp NASA POWER: "YYYYMMDDTHH" (ví dụ: "2020010100").
    Giá trị -999.0 là mã đặc biệt cho dữ liệu thiếu → thay bằng NaN.

    Parameters
    ----------
    data : dict
        JSON response từ NASA POWER API.
    params : list of str
        Danh sách tham số cần trích xuất.

    Returns
    -------
    pd.DataFrame
        DataFrame với DatetimeIndex UTC theo giờ.
    """
    properties    = data["properties"]["parameter"]
    sample_param  = params[0]
    timestamps_raw = list(properties[sample_param].keys())

    records = []
    for ts_str in timestamps_raw:
        try:
            ts = pd.to_datetime(ts_str, format="%Y%m%d%H")
        except Exception:
            continue  # Bỏ qua timestamp không hợp lệ

        row = {"timestamp": ts}
        for p in params:
            val = properties[p].get(ts_str, np.nan)
            # NASA POWER dùng -999.0 để mã hóa dữ liệu khuyết
            row[p] = np.nan if val == -999.0 else val

        records.append(row)

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return df


# ============================================================
# TIỀN XỬ LÝ DỮ LIỆU KHÍ TƯỢNG
# ============================================================
def preprocess_nasa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch và tính các feature mưa tích lũy từ dữ liệu NASA POWER.

    Các bước xử lý:
      1. Reindex về chuỗi giờ đầy đủ (lấp khoảng trống trong index)
      2. Nội suy tuyến tính cho khoảng thiếu ≤ 6 giờ (limit=6)
      3. Clip mưa âm về 0 (vật lý không hợp lệ)
      4. Tính mưa tích lũy rolling (1h, 6h, 24h)
      5. Đổi tên cột sang tiếng Anh thân thiện

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame raw từ NASA POWER với DatetimeIndex UTC.

    Returns
    -------
    pd.DataFrame
        DataFrame đã xử lý với các cột mưa tích lũy.
    """
    logger.info("[NASA POWER] Tiền xử lý dữ liệu...")

    # 1. Đảm bảo index giờ đầy đủ (không bị lỗ hổng)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)

    # 2. Nội suy tuyến tính — chỉ lấp khoảng < 6 giờ (limit=6)
    #    Khoảng dài hơn giữ nguyên NaN để tránh suy luận sai
    df = df.interpolate(method="linear", limit=6, limit_direction="both")

    # 3. Mưa không được âm
    df["PRECTOTCORR"] = df["PRECTOTCORR"].clip(lower=0)

    # 4. Mưa tích lũy rolling (vectorized — không dùng vòng lặp for)
    df["rain_1h"]  = df["PRECTOTCORR"].rolling(window=1,  min_periods=1).sum()
    df["rain_6h"]  = df["PRECTOTCORR"].rolling(window=6,  min_periods=1).sum()
    df["rain_24h"] = df["PRECTOTCORR"].rolling(window=24, min_periods=1).sum()

    # 5. Đổi tên cột
    df = df.rename(columns={
        "PRECTOTCORR": "rain_hourly",
        "T2M":         "temperature",
        "RH2M":        "humidity",
        "WS2M":        "wind_speed",
    })

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BƯỚC 1: THU THẬP DỮ LIỆU KHÍ TƯỢNG NASA POWER")
    logger.info("=" * 60)

    all_dfs  = []
    failed   = []  # Danh sách năm thất bại

    for year in range(START_YEAR, END_YEAR + 1):
        start = f"{year}0101"
        end   = f"{year}1231"

        # Kiểm tra checkpoint — nếu đã tải năm này thì bỏ qua
        checkpoint_path = os.path.join(CHECKPOINT_DIR,
                                       f"nasa_power_{year}.csv")
        if os.path.exists(checkpoint_path):
            logger.info("  [%d] Đọc từ checkpoint: %s", year, checkpoint_path)
            df_year = pd.read_csv(checkpoint_path,
                                  index_col=0, parse_dates=True)
            df_year.index.name = "timestamp"
            all_dfs.append(df_year)
            continue

        # Gọi API với retry
        try:
            raw     = fetch_nasa_power(LAT, LON, start, end, NASA_PARAMS)
            df_year = parse_nasa_response(raw, NASA_PARAMS)

            # Lưu checkpoint ngay sau khi tải thành công
            df_year.to_csv(checkpoint_path)
            logger.info(
                "  ✓ %d: %d bản ghi → lưu checkpoint.", year, len(df_year)
            )
            all_dfs.append(df_year)

        except RuntimeError as exc:
            # Ghi nhận thất bại nhưng tiếp tục với năm tiếp theo
            logger.error("  ✗ %d: Thất bại — %s", year, exc)
            failed.append(year)

    # Báo cáo năm thất bại
    if failed:
        logger.warning(
            "\n⚠ Không lấy được dữ liệu cho %d năm: %s",
            len(failed), failed,
        )
        logger.warning(
            "  → Hãy chạy lại script sau khi kết nối ổn định.\n"
            "  → Checkpoint đã có sẽ không bị tải lại."
        )

    if not all_dfs:
        raise RuntimeError(
            "Không lấy được dữ liệu nào từ NASA POWER!\n"
            "Kiểm tra kết nối mạng và chạy lại script."
        )

    # Ghép tất cả năm và tiền xử lý
    df_all = pd.concat(all_dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]  # Loại timestamp trùng
    df_all = preprocess_nasa(df_all)

    # Lưu file tổng hợp
    df_all.to_csv(OUTPUT_PATH)
    logger.info("\n[NASA POWER] Đã lưu: %s", OUTPUT_PATH)
    logger.info(
        "  Tổng: %d bản ghi | %d giá trị khuyết | %s → %s",
        len(df_all),
        int(df_all.isnull().sum().sum()),
        df_all.index.min().date(),
        df_all.index.max().date(),
    )
    logger.info("\n%s", df_all.describe().round(2).to_string())

    if failed:
        logger.warning(
            "\n⚠ Cảnh báo: File tổng hợp thiếu dữ liệu năm %s.", failed
        )


if __name__ == "__main__":
    main()

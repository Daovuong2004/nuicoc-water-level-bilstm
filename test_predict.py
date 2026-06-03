"""
test_predict.py — Demo gọi API dự báo mực nước hồ Núi Cốc
===========================================================
Lấy 21 ngày THỰC TẾ gần nhất từ dataset_full.csv (chưa scale)
gửi lên POST http://localhost:8000/predict
in kết quả dự báo 5 chân trời + cảnh báo lũ.

Chạy:
    python test_predict.py
    python test_predict.py --date 2024-09-07   (dự báo từ 1 ngày cụ thể)
"""

import sys
import json
import argparse
import requests
import pandas as pd
import numpy as np

# Windows UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from config import FEATURE_COLS, WINDOW_SIZE


API_URL = "http://localhost:8000/predict"


# ─────────────────────────────────────────────
# Load dữ liệu gốc (chưa scale)
# ─────────────────────────────────────────────
def load_raw_window(date_str: str | None = None) -> tuple[list, float, str]:
    """
    Đọc 21 ngày liên tiếp kết thúc tại `date_str` (hoặc ngày mới nhất).

    Trả về:
        features     : list[list[float]] shape (21, 16) — GIÁ TRỊ THỰC (chưa scale)
        base_level_m : float — mực nước ngày cuối cùng (m)
        end_date_str : str   — ngày kết thúc window
    """
    df = pd.read_csv(
        "data/final/dataset_full.csv",
        index_col=0,
        parse_dates=True,
    )
    df.index = pd.to_datetime(df.index)

    # Kiểm tra đủ features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu features trong dataset_full.csv: {missing}")

    # Xác định ngày kết thúc window
    if date_str:
        end_date = pd.Timestamp(date_str)
        if end_date not in df.index:
            # Lấy ngày gần nhất ≤ date_str
            avail = df.index[df.index <= end_date]
            if len(avail) == 0:
                raise ValueError(f"Không có dữ liệu trước ngày {date_str}")
            end_date = avail[-1]
    else:
        end_date = df.index[-1]

    # Lấy WINDOW_SIZE ngày kết thúc tại end_date
    end_pos = df.index.get_loc(end_date)
    if end_pos < WINDOW_SIZE - 1:
        raise ValueError(
            f"Không đủ {WINDOW_SIZE} ngày trước {end_date.date()}. "
            f"Chỉ có {end_pos + 1} ngày."
        )

    window = df.iloc[end_pos - WINDOW_SIZE + 1 : end_pos + 1]
    features = window[FEATURE_COLS].values.tolist()

    # base_level_m = mực nước tại ngày cuối window (H(t))
    base_level_m = float(df.loc[end_date, "water_level_m"])

    return features, base_level_m, str(end_date.date())


# ─────────────────────────────────────────────
# Gọi API
# ─────────────────────────────────────────────
def call_predict_api(features: list, base_level_m: float, end_date_str: str) -> dict:
    payload = {
        "features": features,
        "base_level_m": base_level_m,
        "timestamp": f"{end_date_str}T12:00:00+07:00",
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("\n❌ Không kết nối được server. Hãy chạy trước:")
        print("   python 08_api_serve.py")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Lỗi HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)


# ─────────────────────────────────────────────
# Hiển thị kết quả
# ─────────────────────────────────────────────
ALERT_COLORS = {
    "BÌNH THƯỜNG": "\033[92m",  # xanh lá
    "CẢNH BÁO":    "\033[93m",  # vàng
    "NGUY HIỂM":   "\033[91m",  # đỏ
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def print_result(result: dict, base_level_m: float, end_date_str: str):
    alert   = result["alert_level"]
    color   = ALERT_COLORS.get(alert, "")
    forecasts = result["forecasts"]

    print()
    print("=" * 62)
    print(f"{BOLD}  DỰ BÁO MỰC NƯỚC HỒ NÚI CỐC — Bi-LSTM v5.1{RESET}")
    print("=" * 62)
    print(f"  Ngày phát hành (issue time) : {end_date_str}")
    print(f"  Mực nước hiện tại H(t)      : {base_level_m:.2f} m")
    print(f"  Thời điểm xử lý             : {result['request_time'][:19]} UTC")
    print("-" * 62)
    print(f"  {'Chân trời':<14} {'Dự báo (m)':>11} {'CI95 dưới':>11} {'CI95 trên':>11}")
    print("-" * 62)

    for f in forecasts:
        d        = f["horizon_d"]
        wl       = f["water_level_m"]
        lo       = f["ci95_lower"]
        hi       = f["ci95_upper"]
        delta    = wl - base_level_m
        sign     = "▲" if delta >= 0 else "▼"
        print(
            f"  t+{d:<3}d ({d:>2} ngày)  {wl:>9.3f}m  {lo:>9.3f}m  {hi:>9.3f}m  "
            f"{sign}{abs(delta):.3f}m"
        )

    print("-" * 62)
    print(f"\n  {color}{BOLD}MỨC CẢNH BÁO: {alert}{RESET}")
    print(f"  {result['alert_message']}")
    print("=" * 62)
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Demo dự báo mực nước hồ Núi Cốc qua API Bi-LSTM"
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Ngày kết thúc cửa sổ 21 ngày (mặc định: ngày mới nhất trong dataset)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="In raw JSON response thay vì bảng đẹp",
    )
    args = parser.parse_args()

    print(f"\n[1] Đọc dữ liệu thực từ dataset_full.csv ...")
    features, base_level_m, end_date_str = load_raw_window(args.date)
    print(f"    Window: 21 ngày kết thúc {end_date_str} | H(t)={base_level_m:.2f}m")
    print(f"    Features: {len(features)} bước × {len(features[0])} đặc trưng")

    print(f"\n[2] Gửi POST {API_URL} ...")
    result = call_predict_api(features, base_level_m, end_date_str)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result, base_level_m, end_date_str)


if __name__ == "__main__":
    main()

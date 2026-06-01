"""
07_infer_qout.py — Suy luận lưu lượng xả Q_out và phát hiện xả đột ngột
từ phương trình cân bằng nước hồ chứa:

    Q_out(t) = Q_in(t) - dS/dt
             = Q_in(t) - A(H) × dH/dt / 3600

Đầu vào : data/final/dataset_full.csv  (đã có water_level_m, q_in_m3s)
Đầu ra  : data/final/dataset_with_qout.csv  (thêm 5 cột mới)
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure standard output and error output use UTF-8 to prevent UnicodeEncodeError on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================================================
# CẤU HÌNH
# ============================================================
INPUT_PATH  = "data/final/dataset_full.csv"
OUTPUT_PATH = "data/final/dataset_with_qout.csv"
os.makedirs("results", exist_ok=True)

# Ngưỡng phát hiện xả đột ngột
QOUT_THRESHOLD  = 200    # m³/s — lưu lượng tối thiểu để coi là "đang xả lớn"
DH_THRESHOLD    = -0.05  # m/h  — mực nước phải đang giảm
RAIN_THRESHOLD  = 5      # mm/6h — mưa nhỏ (loại trừ giảm do không mưa tự nhiên)

# ============================================================
# ĐƯỜNG CONG A-H (Diện tích mặt hồ → Mực nước)
# Thông số hồ Núi Cốc đã xác minh:
#   MNDBT = 46.20m → ~2.500 ha (25 km²)
#   Lũ tối đa      → ~3.200 ha (32 km²)
# ============================================================
AH_CURVE = [
    (36.0,  200e4),    # Gần mực nước chết
    (38.0,  500e4),
    (40.0,  900e4),
    (42.0, 1400e4),
    (44.0, 2000e4),
    (46.2, 2500e4),    # MNDBT — điểm chuẩn chính xác
    (46.5, 2700e4),
    (46.9, 2900e4),
    (47.2, 3050e4),
    (47.5, 3150e4),
    (47.8, 3200e4),
    (48.25,3500e4),    # Đỉnh vùng bán ngập
]
_ah_levels = [p[0] for p in AH_CURVE]
_ah_areas  = [p[1] for p in AH_CURVE]
_ah_interp = interp1d(_ah_levels, _ah_areas,
                      kind="linear", bounds_error=False,
                      fill_value=(_ah_areas[0], _ah_areas[-1]))

def area_from_level(h):
    """Diện tích mặt hồ (m²) tại mực nước h (m)."""
    return float(_ah_interp(h))


# ============================================================
# SỬA LUẬN Q_OUT
# ============================================================
def infer_qout(df):
    """
    Tính Q_out từ phương trình cân bằng nước:
        dS/dt = A(H) × dH/dt          (m³/h)
        Q_out = Q_in - dS/dt / 3600   (m³/s)
    """
    df = df.copy()

    # 1. Tốc độ thay đổi mực nước (m/h)
    df["dH_dt"] = df["water_level_m"].diff(1)

    # 2. Diện tích mặt hồ tại mực nước hiện tại (m²)
    df["area_m2"] = df["water_level_m"].apply(area_from_level)

    # 3. Tốc độ thay đổi dung tích (m³/s)
    df["dS_dt_m3s"] = (df["area_m2"] * df["dH_dt"]) / 3600.0

    # 4. Lưu lượng xả ước tính
    #    Nếu không có q_in_m3s thì dùng 0 (chỉ tính từ biến thiên mực nước)
    if "q_in_m3s" in df.columns:
        df["Q_out_est"] = df["q_in_m3s"] - df["dS_dt_m3s"]
    else:
        df["Q_out_est"] = -df["dS_dt_m3s"]

    # 5. Clip âm (vật lý không thể xả âm)
    df["Q_out_est"] = df["Q_out_est"].clip(lower=0)

    # 6. Làm mịn nhiễu bằng rolling median (cửa sổ 3 giờ)
    df["Q_out_smooth"] = (df["Q_out_est"]
                          .rolling(3, center=True, min_periods=1)
                          .median())

    return df


# ============================================================
# PHÁT HIỆN XẢ ĐỘT NGỘT
# ============================================================
def detect_sudden_release(df):
    """
    Phát hiện giờ có xả đột ngột dựa trên 3 điều kiện đồng thời:
      1. Q_out_smooth > QOUT_THRESHOLD  (đang xả lớn)
      2. dH_dt < DH_THRESHOLD           (mực nước đang giảm)
      3. rain_6h < RAIN_THRESHOLD       (mưa nhỏ → giảm do xả, không phải tự nhiên)
    """
    df = df.copy()

    # Điều kiện xả lớn
    cond_qout = df["Q_out_smooth"] > QOUT_THRESHOLD

    # Điều kiện mực nước giảm
    cond_giam = df["dH_dt"] < DH_THRESHOLD

    # Điều kiện mưa nhỏ (dùng cột rain_6h nếu có)
    if "rain_6h" in df.columns:
        cond_mua  = df["rain_6h"] < RAIN_THRESHOLD
    else:
        cond_mua  = pd.Series(True, index=df.index)

    # Flag xả đột ngột
    df["xa_dot_ngot"] = (cond_qout & cond_giam & cond_mua).astype(int)

    # Phân cấp mức độ xả
    conditions = [
        df["Q_out_smooth"] < 100,
        df["Q_out_smooth"] < 300,
        df["Q_out_smooth"] < 600,
        df["Q_out_smooth"] < 1000,
        df["Q_out_smooth"] >= 1000,
    ]
    choices = ["Không xả / xả nhỏ", "Xả vừa", "Xả lớn", "Xả rất lớn", "Xả tối đa"]
    df["muc_xa"] = np.select(conditions, choices, default="Không xác định")

    # Thống kê
    n_xa = df["xa_dot_ngot"].sum()
    pct  = n_xa / len(df) * 100
    print(f"\n[Phát hiện] Số giờ xả đột ngột: {n_xa:,} / {len(df):,} ({pct:.2f}%)")
    print(f"\n[Phân cấp Q_out] Phân phối:")
    print(df["muc_xa"].value_counts().to_string())

    return df


# ============================================================
# CROSS-CHECK VỚI ĐIỂM BÁO CHÍ
# ============================================================
def crosscheck_bao_chi(df):
    """
    So sánh Q_out suy luận với các điểm báo chí đã biết.
    """
    bao_chi = [
        # (timestamp, Q_out_bao_chi, nguon)
        ("2024-08-01 07:00", 100,  "danviet.vn — xả 100 m³/s"),
        ("2024-08-23 10:30", 125,  "baothainguyen.vn — xả 50-200 m³/s"),
        ("2024-08-30 14:00", 115,  "baotintuc.vn — xả 30-200 m³/s"),
        ("2024-09-05 14:00", 90,   "vtv.vn — xả 30-150 m³/s"),
        ("2024-09-09 13:30", 180,  "thainguyen.gov.vn — xả 60-300 m³/s"),
        ("2025-09-30 12:15", 250,  "baothainguyen.vn — mở 5 cửa 250 m³/s"),
    ]

    print("\n[Cross-check] So sánh Q_out suy luận vs báo chí:")
    print(f"  {'Thời điểm':<22} {'Q_bc (m³/s)':>12} {'Q_est (m³/s)':>14} {'Sai lệch':>10} {'Nguồn'}")
    print("  " + "-" * 85)

    errors = []
    for ts_str, q_bc, nguon in bao_chi:
        ts = pd.to_datetime(ts_str)
        if ts in df.index:
            q_est = df.loc[ts, "Q_out_smooth"]
        else:
            # Tìm timestamp gần nhất
            idx = df.index.get_indexer([ts], method="nearest")[0]
            q_est = df.iloc[idx]["Q_out_smooth"]

        err = abs(q_est - q_bc)
        errors.append(err)
        flag = "✓" if err < 100 else "✗"
        print(f"  {ts_str:<22} {q_bc:>12.0f} {q_est:>14.1f} {err:>9.1f}  {flag}  {nguon}")

    mae = np.mean(errors)
    print(f"\n  MAE cross-check: {mae:.1f} m³/s")
    if mae < 100:
        print("  → Suy luận đạt yêu cầu (sai số < 100 m³/s)")
    else:
        print("  → Cần điều chỉnh đường cong A-H hoặc q_in")


# ============================================================
# VẼ BIỂU ĐỒ
# ============================================================
def plot_results(df):
    """Vẽ biểu đồ mực nước, Q_out và flag xả đột ngột."""
    # Chỉ vẽ giai đoạn lũ Yagi 2024
    mask = (df.index >= "2024-09-06") & (df.index <= "2024-09-15")
    df_plot = df[mask]

    if len(df_plot) == 0:
        print("[Biểu đồ] Không có dữ liệu giai đoạn Yagi 2024 để vẽ.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Suy luận Q_out — Lũ Yagi 2024 | Hồ Núi Cốc",
                 fontsize=13, fontweight="bold")

    # --- Biểu đồ 1: Mực nước ---
    ax1 = axes[0]
    ax1.plot(df_plot.index, df_plot["water_level_m"],
             color="royalblue", linewidth=2, label="Mực nước (m)")
    ax1.axhline(46.20, color="orange", linestyle="--", linewidth=1, label="MNDBT 46.20m")
    ax1.axhline(47.40, color="red",    linestyle="--", linewidth=1, label="LTK 47.40m")

    # Tô màu vùng xả đột ngột
    for i, row in df_plot[df_plot["xa_dot_ngot"] == 1].iterrows():
        ax1.axvspan(i, i + pd.Timedelta(hours=1),
                    alpha=0.2, color="red", linewidth=0)

    ax1.set_ylabel("Mực nước (m)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # --- Biểu đồ 2: Q_out suy luận ---
    ax2 = axes[1]
    ax2.fill_between(df_plot.index, df_plot["Q_out_smooth"],
                     alpha=0.4, color="tomato", label="Q_out suy luận (m³/s)")
    ax2.plot(df_plot.index, df_plot["Q_out_smooth"],
             color="tomato", linewidth=1.5)
    ax2.axhline(QOUT_THRESHOLD, color="gray", linestyle=":",
                linewidth=1, label=f"Ngưỡng xả lớn ({QOUT_THRESHOLD} m³/s)")

    if "q_in_m3s" in df_plot.columns:
        ax2.plot(df_plot.index, df_plot["q_in_m3s"],
                 color="steelblue", linewidth=1, alpha=0.6, label="Q_in (m³/s)")

    ax2.set_ylabel("Lưu lượng (m³/s)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # --- Biểu đồ 3: Flag xả đột ngột ---
    ax3 = axes[2]
    ax3.fill_between(df_plot.index, df_plot["xa_dot_ngot"],
                     step="post", alpha=0.7, color="red", label="Xả đột ngột (1=Có)")
    ax3.set_ylim(-0.1, 1.5)
    ax3.set_ylabel("Flag xả đột ngột")
    ax3.set_xlabel("Thời gian")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig("results/qout_infer_yagi2024.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Biểu đồ] Đã lưu: results/qout_infer_yagi2024.png")


# ============================================================
# THÊM VÀO FEATURE SET CHO BI-LSTM
# ============================================================
def add_qout_features(df):
    """
    Thêm các đặc trưng Q_out vào bộ dữ liệu để đưa vào Bi-LSTM.
    """
    df = df.copy()

    # Lag Q_out (giờ trước)
    df["Q_out_lag1"]  = df["Q_out_smooth"].shift(1)
    df["Q_out_lag3"]  = df["Q_out_smooth"].shift(3)
    df["Q_out_lag6"]  = df["Q_out_smooth"].shift(6)

    # Rolling Q_out (xu hướng xả)
    df["Q_out_roll6"]  = df["Q_out_smooth"].rolling(6,  min_periods=1).mean()
    df["Q_out_roll24"] = df["Q_out_smooth"].rolling(24, min_periods=1).mean()

    # Tốc độ thay đổi Q_out (gia tốc xả)
    df["dQout_dt"] = df["Q_out_smooth"].diff(1)

    print("\n[Features] Đã thêm 7 đặc trưng Q_out vào bộ dữ liệu:")
    new_cols = ["Q_out_smooth","Q_out_lag1","Q_out_lag3","Q_out_lag6",
                "Q_out_roll6","Q_out_roll24","dQout_dt"]
    print(df[new_cols].describe().round(2).to_string())

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("BƯỚC 7: SUY LUẬN Q_OUT VÀ PHÁT HIỆN XẢ ĐỘT NGỘT")
    print("=" * 60)

    # Load dữ liệu
    print("\n[Load] Đọc dataset_full.csv...")
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    print(f"  Tổng: {len(df):,} bản ghi | {df.index.min()} → {df.index.max()}")

    # Bước 1: Suy luận Q_out
    print("\n[1] Tính Q_out từ phương trình cân bằng nước...")
    df = infer_qout(df)

    # Bước 2: Phát hiện xả đột ngột
    print("\n[2] Phát hiện xả đột ngột...")
    df = detect_sudden_release(df)

    # Bước 3: Cross-check báo chí
    print("\n[3] Cross-check với điểm báo chí...")
    crosscheck_bao_chi(df)

    # Bước 4: Thêm features
    print("\n[4] Thêm Q_out features cho Bi-LSTM...")
    df = add_qout_features(df)

    # Bước 5: Vẽ biểu đồ
    print("\n[5] Vẽ biểu đồ...")
    plot_results(df)

    # Lưu kết quả
    df.to_csv(OUTPUT_PATH)
    print(f"\n[Lưu] {OUTPUT_PATH}")

    # Tóm tắt các cột mới
    new_cols = ["dH_dt","area_m2","dS_dt_m3s","Q_out_est","Q_out_smooth",
                "xa_dot_ngot","muc_xa","Q_out_lag1","Q_out_lag6",
                "Q_out_roll24","dQout_dt"]
    existing = [c for c in new_cols if c in df.columns]
    print(f"\n[Tóm tắt] Đã thêm {len(existing)} cột mới vào dataset:")
    for c in existing:
        print(f"  + {c}")

    print("\n✓ Hoàn thành! Tiếp theo:")
    print("  → Thêm các cột Q_out vào FEATURE_COLS trong 06_bilstm_model.py")
    print("  → Chạy lại python 06_bilstm_model.py để retrain với features mới")


if __name__ == "__main__":
    main()

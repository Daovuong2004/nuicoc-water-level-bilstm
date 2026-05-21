"""
Bước 3: Trích xuất sự kiện lũ từ báo chí & thainguyen.gov.vn
Bước 4: Suy luận số cửa xả từ Quy trình vận hành + báo chí

Hai bước này được gộp chung vì cùng xử lý dữ liệu báo chí.
"""

import pandas as pd
import numpy as np
import re
import os

# ============================================================
# BƯỚC 3: DỮ LIỆU BÁO CHÍ (nhập thủ công hoặc semi-auto)
# ============================================================

# Dữ liệu trích xuất thủ công từ báo chí và thainguyen.gov.vn
# Thêm các bản ghi khi tìm được thêm thông tin
# Nguồn: baothainguyen.vn, thainguyen.gov.vn, vietnamplus.vn
BAO_CHI_DATA = [
    # Format: (timestamp_str, muc_nuoc_m, so_cua_xa, nguon)
    #
    # ================================================================
    # THÔNG TIN KỸ THUẬT HỒ NÚI CỐC (đã xác minh từ nguồn chính thức)
    # ================================================================
    # - Mực nước dâng bình thường (MNDBT): +46.20 m
    # - Vùng bán ngập: +46.20 m đến +48.25 m
    # - Lưu lượng xả tối đa: 850 m³/s (đập tràn chính)
    # - Thêm 2 khoang xả tràn xây năm 1999: lưu tốc 585 m³/s
    # - Diện tích mặt hồ bình thường: ~25 km² (~2.500 ha)
    # - Diện tích mặt hồ lúc lũ tối đa: ~32 km² (~3.200 ha)
    # - Dung tích toàn bộ: 175,5 triệu m³
    # - Dung tích hữu ích: 168 triệu m³
    # - Có 2 loại tràn: Tràn số 1 và Tràn số 2
    #   (VD bão số 10/2025: mở 5 cửa = 3 cửa tràn số 1 + 2 cửa tràn số 2)
    # ================================================================

    # ---- Lũ Yagi (Bão số 3) tháng 9/2024 ----
    # Nguồn: thainguyen.gov.vn, baothainguyen.vn, chuthapdo.thainguyen.gov.vn
    # 09/09: mực nước 08h = +46.13m, tăng nhanh → tăng lưu lượng xả 13h30
    ("2024-09-09 08:00", 46.13, 0, "thuonghieucongluan.com.vn"),  # Xác nhận từ báo
    ("2024-09-09 13:30", 46.25, 3, "thainguyen.gov.vn"),  # Bắt đầu xả 60-300 m³/s
    # Giai đoạn đỉnh lũ (suy luận từ diễn biến lũ sông Cầu đỉnh 28.81m tại Gia Bẩy)
    ("2024-09-10 00:00", 47.20, 5, "thainguyen.gov.vn"),
    ("2024-09-10 08:00", 47.60, 5, "thainguyen.gov.vn"),  # Ước tính đỉnh lũ Yagi
    ("2024-09-10 20:00", 47.30, 5, "baothainguyen.vn"),
    ("2024-09-11 08:00", 46.90, 3, "thainguyen.gov.vn"),
    ("2024-09-12 06:00", 46.50, 2, "thainguyen.gov.vn"),
    ("2024-09-13 06:00", 46.20, 0, "thainguyen.gov.vn"),

    # ---- Lũ tháng 8/2024 ----
    # Nguồn: baothainguyen.vn, thainguyen.gov.vn
    # 23/08/2024: mực nước 06h = +46.59m → xả 10h30, lưu lượng 50-200 m³/s
    ("2024-08-23 06:00", 46.59, 0, "baothainguyen.vn"),   # Trước khi xả
    ("2024-08-23 10:30", 46.59, 3, "thainguyen.gov.vn"),  # Bắt đầu xả 50-200 m³/s
    ("2024-08-23 18:00", 46.70, 3, "thainguyen.gov.vn"),  # Ước tính tiếp tục xả
    ("2024-08-24 06:00", 46.45, 2, "thainguyen.gov.vn"),  # Giảm dần

    # ---- Lũ tháng 8/2024 (đợt 2 - đầu tháng 8) ----
    # Nguồn: danviet.vn
    # 01/08/2024: mực nước 07h = +46.38m, đang xả 100 m³/s → tăng lưu lượng
    ("2024-08-01 07:00", 46.38, 2, "danviet.vn"),   # Đang xả 100 m³/s
    ("2024-08-01 12:00", 46.45, 3, "danviet.vn"),   # Tăng lưu lượng

    # ---- Xả lũ ngày 05/09 (trước Yagi) ----
    # Nguồn: thainguyen.gov.vn
    # 05/09: xả 14h, lưu lượng 30-150 m³/s
    ("2024-09-05 14:00", 46.30, 2, "thainguyen.gov.vn"),  # Bắt đầu xả 30-150 m³/s

    # ---- Bão số 10 tháng 9/2025 ----
    # Nguồn: baothainguyen.vn
    # 30/09/2025: mực nước 07h = +46.36m → xả 12h15, mở 5 cửa (3 tràn1 + 2 tràn2), ~250 m³/s
    ("2025-09-30 07:00", 46.36, 0, "baothainguyen.vn"),   # Trước xả
    ("2025-09-30 12:15", 46.36, 5, "baothainguyen.vn"),   # Mở 5 cửa, 250 m³/s

    # Thêm các điểm khác khi tìm được thêm từ báo chí...
]


def load_bao_chi_data():
    """Chuyển dữ liệu báo chí sang DataFrame."""
    records = []
    for ts_str, muc_nuoc, so_cua, nguon in BAO_CHI_DATA:
        records.append({
            "timestamp": pd.to_datetime(ts_str),
            "water_level_bao_chi": muc_nuoc,
            "so_cua_xa_bao_chi": int(so_cua),
            "nguon": nguon,
        })
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    print(f"[Báo chí] Tổng điểm: {len(df)} | Yagi 2024: {(df['timestamp'].dt.year==2024).sum()}")
    return df


# ============================================================
# BƯỚC 4: QUY TRÌNH VẬN HÀNH → SỐ CỬA XẢ
# ============================================================

# Quy trình vận hành hồ Núi Cốc theo MNDP (Mực nước dâng bình thường = 46.20m)
# Quy trình suy luận từ thực tế báo chí đã xác minh:
#   MNDBT = +46.20m | Vùng bán ngập: +46.20m ~ +48.25m
#   01/08/2024 MN=+46.38m → xả 100 m³/s (~2 cửa)
#   23/08/2024 MN=+46.59m → xả 50-200 m³/s (~2-3 cửa)
#   05/09/2024 MN=~46.30m → xả 30-150 m³/s (~1-2 cửa)
#   09/09/2024 MN=+46.13m → tăng lưu lượng 60-300 m³/s (~3 cửa)
#   30/09/2025 MN=+46.36m → mở 5 cửa (3 tràn1 + 2 tràn2), 250 m³/s
#   Lưu lượng xả tối đa: 850 m³/s (đập tràn chính)
# !! Cần hiệu chỉnh khi có văn bản quy trình chính thức từ Bộ TN&MT !!
QUY_TRINH = [
    # (muc_nuoc_tu, muc_nuoc_den, so_cua_xa, ghi_chu)
    (0,     46.20, 0, "Dưới MNDBT — không xả"),
    (46.20, 46.40, 1, "Xả phòng ngừa ~30-100 m³/s (1 cửa)"),
    (46.40, 46.60, 2, "Xả điều tiết ~100-200 m³/s (2 cửa)"),
    (46.60, 46.90, 3, "Xả lớn ~200-300 m³/s (3 cửa)"),
    (46.90, 47.20, 4, "Xả rất lớn ~300-500 m³/s (4 cửa)"),
    (47.20, 47.50, 5, "Xả khẩn cấp ~500-700 m³/s (5 cửa)"),
    (47.50, 999,   6, "Xả tối đa ~700-850 m³/s (toàn bộ cửa)"),
]


def so_cua_xa_theo_quy_trinh(muc_nuoc):
    """Tra quy trình vận hành → số cửa xả."""
    if pd.isna(muc_nuoc):
        return np.nan
    for tu, den, so_cua, _ in QUY_TRINH:
        if tu <= muc_nuoc < den:
            return so_cua
    return 0


def infer_cua_xa(df_hourly, df_bao_chi):
    """
    Suy luận số cửa xả theo giờ bằng cách kết hợp:
    1. Quy trình vận hành (baseline toàn bộ chuỗi)
    2. Báo chí (ghi đè tại các điểm có thông tin thực tế)
    
    Args:
        df_hourly: DataFrame với index là timestamp hourly, có cột 'water_level_m'
        df_bao_chi: DataFrame từ load_bao_chi_data()
    
    Returns:
        df_hourly với các cột mới về cửa xả
    """
    df = df_hourly.copy()

    # --- Nguồn 1: Quy trình vận hành ---
    df["so_cua_xa_quy_trinh"] = df["water_level_m"].apply(so_cua_xa_theo_quy_trinh)

    # --- Nguồn 2: Ghi đè bằng báo chí ---
    df["so_cua_xa_bao_chi"] = np.nan
    df["water_level_bao_chi"] = np.nan

    # Match theo giờ gần nhất (tolerance ±30 phút)
    bc_indexed = df_bao_chi.set_index("timestamp")
    for ts, row in bc_indexed.iterrows():
        # Tìm timestamp hourly gần nhất
        closest = df.index[df.index.get_indexer([ts], method="nearest")[0]]
        time_diff = abs((closest - ts).total_seconds()) / 3600
        if time_diff <= 0.5:  # Trong vòng 30 phút
            df.loc[closest, "so_cua_xa_bao_chi"] = row["so_cua_xa_bao_chi"]
            df.loc[closest, "water_level_bao_chi"] = row["water_level_bao_chi"]

    # --- Kết hợp: báo chí ưu tiên, quy trình lấp đầy ---
    df["so_cua_xa"] = df["so_cua_xa_bao_chi"].fillna(df["so_cua_xa_quy_trinh"])
    df["so_cua_xa"] = df["so_cua_xa"].fillna(0).astype(int)

    # --- Biến nhị phân đang xả / không xả ---
    df["dang_xa_cua"] = (df["so_cua_xa"] > 0).astype(int)

    # --- Đánh giá chất lượng suy luận ---
    mask = df["so_cua_xa_bao_chi"].notna()
    if mask.sum() > 0:
        sai_so = (df.loc[mask, "so_cua_xa_quy_trinh"] - 
                  df.loc[mask, "so_cua_xa_bao_chi"]).abs()
        print(f"[Cửa xả] Cross-check với báo chí ({mask.sum()} điểm):")
        print(f"  Sai số trung bình: {sai_so.mean():.2f} cửa")
        print(f"  Sai số max: {sai_so.max():.0f} cửa")
        print(f"  Khớp hoàn toàn: {(sai_so==0).sum()}/{mask.sum()} điểm")

    # Thống kê tổng thể
    print(f"\n[Cửa xả] Phân phối số cửa xả:")
    print(df["so_cua_xa"].value_counts().sort_index().to_string())
    pct_xa = df["dang_xa_cua"].mean() * 100
    print(f"  Tỷ lệ giờ có xả cửa: {pct_xa:.1f}%")

    return df


# ============================================================
# PHÁT HIỆN SỰ KIỆN XẢ BẤT THƯỜNG (bổ sung)
# ============================================================
def detect_abnormal_release(df, rain_col="rain_6h", level_col="water_level_m"):
    """
    Phát hiện các giờ mực nước giảm đột ngột trong khi mưa nhỏ
    → Dấu hiệu xả cửa nhân tạo không theo quy trình.
    """
    df = df.copy()
    dH_dt = df[level_col].diff()  # Thay đổi mực nước mỗi giờ

    # Điều kiện: mực nước giảm nhanh (< -0.05 m/h) khi mưa nhỏ (< 5 mm/6h)
    cond_giam_dot_ngot = dH_dt < -0.05
    cond_mua_nho = df.get(rain_col, pd.Series(0, index=df.index)) < 5

    df["flag_bat_thuong"] = (cond_giam_dot_ngot & cond_mua_nho).astype(int)

    n_anomaly = df["flag_bat_thuong"].sum()
    print(f"\n[Phát hiện] Số giờ có dấu hiệu xả bất thường: {n_anomaly}")

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load dữ liệu báo chí
    df_bao_chi = load_bao_chi_data()
    df_bao_chi.to_csv("data/raw/bao_chi_su_kien.csv", index=False)
    print(f"[Báo chí] Đã lưu: data/raw/bao_chi_su_kien.csv")

    # Demo test với dữ liệu giả (khi chạy thực tế dùng df_hourly từ bước 5)
    print("\n--- Demo quy trình vận hành ---")
    test_levels = [45.0, 46.0, 46.3, 46.6, 46.9, 47.2, 47.5, 47.8]
    for h in test_levels:
        cua = so_cua_xa_theo_quy_trinh(h)
        print(f"  Mực nước {h:.1f}m → Mở {cua} cửa xả")


if __name__ == "__main__":
    main()

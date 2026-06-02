import json
import pandas as pd
import requests

# Địa chỉ API Server
API_URL = "http://localhost:8000/predict"

# Danh sách 26 đặc trưng ngày theo đúng thứ tự huấn luyện
FEATURE_COLS = [
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
    "temperature", "humidity",
    "water_level_lag1", "water_level_lag3", "water_level_lag7",
    "water_level_lag14", "water_level_lag30", "water_level_lag60",
    "water_level_roll7", "water_level_roll30", "water_level_roll60",
    "water_level_std7",
    "month_sin", "month_cos", "season_wet", "season_dry",
    "delta_h_7d", "delta_h_30d",
    "dH_dt_daily", "Q_out_daily", "Q_out_roll7",
]

def main():
    print("=" * 60)
    print("CHẠY THỬ NGHIỆM API DỰ BÁO MỰC NƯỚC HỒ NÚI CỐC")
    print("=" * 60)

    # 1. Đọc dữ liệu từ dataset_full.csv (chứa dữ liệu thô chưa chuẩn hóa)
    try:
        df = pd.read_csv("data/final/dataset_full.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file data/final/dataset_full.csv!")
        return

    # Lọc lấy khoảng thời gian tập test (từ 2024-01-01) và giới hạn đến 2025-11-10 để khớp với dataset_test.csv
    df = df[(df.index >= "2024-01-01") & (df.index <= "2025-11-10")]

    # Kiểm tra số lượng bản ghi
    if len(df) < 60:
        print(f"Lỗi: Tập test chỉ có {len(df)} bản ghi, cần tối thiểu 60 bản ghi.")
        return

    # Lấy 60 ngày cuối cùng
    last_60_days = df.tail(60)
    last_timestamp = last_60_days.index[-1].strftime("%Y-%m-%dT%H:%M:%S+07:00")

    # Trích xuất 26 cột đặc trưng
    features_data = last_60_days[FEATURE_COLS].values.tolist()

    # 2. Tạo Request Payload
    payload = {
        "features": features_data,
        "timestamp": last_timestamp
    }

    # 3. Gửi Request POST đến API
    print(f"\n[Gửi] Gửi request đến {API_URL}...")
    print(f"  + Thời điểm quan sát cuối cùng: {last_timestamp}")
    print(f"  + Kích thước ma trận features: {len(features_data)} hàng x {len(features_data[0])} cột")

    try:
        response = requests.post(API_URL, json=payload)
    except requests.exceptions.ConnectionError:
        print("Lỗi: Không thể kết nối tới API Server! Hãy đảm bảo 'python 08_api_serve.py' đang chạy.")
        return

    # 4. Hiển thị kết quả
    if response.status_code == 200:
        res_data = response.json()
        print("\n" + "="*50)
        print(" KẾT QUẢ DỰ BÁO TỪ MÔ HÌNH Bi-LSTM")
        print("="*50)
        print(f"Thời điểm xử lý: {res_data['request_time']}")
        print(f"Mức cảnh báo   : {res_data['alert_level']}")
        print(f"Chi tiết       : {res_data['alert_message']}")
        print("-" * 50)
        print(f"{'Khoảng dự báo':<15} | {'Dự báo (m)':>12} | {'Khoảng tin cậy 95% (m)':^25}")
        print("-" * 50)
        for fc in res_data["forecasts"]:
            horizon = f"t+{fc['horizon_d']} ngày"
            val = fc["water_level_m"]
            ci = f"[{fc['ci95_lower']:.2f} - {fc['ci95_upper']:.2f}]"
            print(f"{horizon:<15} | {val:>12.2f} | {ci:^25}")
        print("="*50)
    else:
        print(f"\nLỗi từ API (Mã lỗi {response.status_code}):")
        print(response.text)

if __name__ == "__main__":
    main()

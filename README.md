# Dự báo mực nước hồ Núi Cốc — Bi-LSTM + Self-Attention

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2-brightgreen)](https://earthengine.google.com)

> **Đồ án tốt nghiệp** — Ngành Công nghệ Thông tin (Kỹ thuật Phần mềm)  
> Đề tài: *Hệ thống dự báo mực nước hồ Núi Cốc sử dụng kiến trúc mô hình Bi-LSTM*

---

## 📌 Tổng quan

Pipeline học máy end-to-end dự báo mực nước hồ Núi Cốc (Thái Nguyên) cho các khoảng thời gian **t+1h, t+3h, t+6h, t+12h, t+24h** sử dụng mô hình **Bidirectional LSTM + Multi-Head Self-Attention**.

### Kiến trúc hệ thống

```
Nguồn dữ liệu                   Pipeline                      Đầu ra
─────────────────────────────────────────────────────────────────────
NASA POWER API  ──┐
GEE Sentinel-2  ──┤──► 05_integrate.py ──► 06_bilstm_model.py ──► Dự báo mực nước
Báo chí/Cửa xả ──┘    (Kalman + Q_out)    (Bi-LSTM+Attention)     + Khoảng tin cậy 95%
                                                  │
                                         08_api_serve.py ──► REST API (FastAPI)
```

### Điểm nổi bật kỹ thuật

| Thành phần | Kỹ thuật |
|-----------|---------|
| **Mô hình** | Bi-LSTM 2 lớp + Multi-Head Self-Attention (4 heads) + Residual + LayerNorm |
| **Uncertainty** | Monte Carlo Dropout → khoảng tin cậy 95% cho từng dự báo |
| **XAI** | SHAP Feature Importance — giải thích "mô hình dựa vào gì" |
| **Dữ liệu vệ tinh** | Sentinel-2 SR + QA60 cloud mask → NDWI → đường cong A-H |
| **Thủy văn** | Phương trình cân bằng nước Q_out = Q_in − A(H)·dH/dt |
| **Tín hiệu** | Kalman Filter làm mịn mực nước quan trắc thưa từ GEE |
| **Anti-leakage** | MinMaxScaler chỉ fit trên tập train |
| **Đánh giá** | Nash-Sutcliffe Efficiency (NSE) — chỉ số chuẩn thủy văn |
| **Serving** | FastAPI + MC Dropout inference + cảnh báo lũ tự động |

---

## 📁 Cấu trúc dự án

```
DATN/
├── 01_nasa_power.py          # Bước 1: Thu thập dữ liệu khí tượng NASA POWER
├── 02_gee_sentinel2.py       # Bước 2: Xử lý ảnh vệ tinh GEE (chế độ offline)
├── 02_gee_colab.py           # Bước 2b: Trích xuất GEE trên Google Colab
├── 03_04_cua_xa_bao_chi.py   # Bước 3+4: Cửa xả + sự kiện lũ từ báo chí
├── 05_integrate.py           # Bước 5: Tích hợp + Kalman + Q_out + normalize
├── 06_bilstm_model.py        # Bước 6: Huấn luyện Bi-LSTM + Attention + SHAP
├── 06b_baseline_comparison.py# Bước 6b: Ablation study SARIMA/LSTM/BiLSTM
├── 07_infer_qout_1.py        # Bước 7: Phân tích Q_out (tùy chọn)
├── 08_api_serve.py           # Bước 8: FastAPI inference server
├── run_all.py                # Chạy toàn bộ pipeline một lệnh
│
├── data/                     # (gitignored — dữ liệu quá lớn)
│   ├── raw/                  #   NASA POWER, GEE, cửa xả
│   ├── processed/            #   Dữ liệu trung gian
│   └── final/                #   dataset_train/val/test.csv
│
├── models/                   # (gitignored — model weights)
│   ├── bilstm_t1h.keras
│   ├── bilstm_t*.keras
│   └── feature_scaler.pkl
│
├── results/                  # (gitignored — biểu đồ, metrics)
│   ├── plot_t*h.png
│   ├── shap_importance_*.png
│   └── metrics_summary.json
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Cài đặt & Chạy

### 1. Yêu cầu
- Python 3.10+
- Tài khoản Google Earth Engine (đã đăng ký)
- Google Cloud Project với Earth Engine API đã kích hoạt

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

### 3. Chạy pipeline đầy đủ

```bash
# Bước 1: Thu thập khí tượng (NASA POWER API)
python 01_nasa_power.py

# Bước 2: Ảnh vệ tinh GEE (chạy trên Google Colab)
#   → Mở 02_gee_colab.py trên Colab và chạy từng cell

# Bước 3–8: Pipeline tự động
python run_all.py
```

### 4. Khởi động API

```bash
uvicorn 08_api_serve:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## 🔢 Bộ đặc trưng (18 features)

| Nhóm | Features |
|------|---------|
| **Khí tượng** | `rain_1h`, `rain_6h`, `rain_24h`, `temperature`, `humidity` |
| **Lag mực nước** | `water_level_lag1/2/3/6/12` |
| **Cửa xả** | `so_cua_xa`, `dang_xa_cua` |
| **Q_out (mới v2.0)** | `Q_out_smooth`, `Q_out_lag1`, `Q_out_lag6`, `Q_out_roll24`, `dQout_dt`, `xa_dot_ngot` |

---

## 📊 Phân chia dữ liệu

| Tập | Giai đoạn | Mục đích |
|-----|-----------|---------|
| **Train** | 2020–06/2023 | Huấn luyện mô hình |
| **Validation** | 07/2023–08/2024 | Tuning siêu tham số |
| **Test** | 09/2024–2025 | Đánh giá — bao gồm lũ Yagi tháng 9/2024 |

---

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/predict` | Dự báo mực nước t+1/3/6/12/24h + khoảng tin cậy 95% |
| `GET` | `/health` | Trạng thái hệ thống |
| `GET` | `/features` | Danh sách 18 features |
| `GET` | `/thresholds` | Ngưỡng cảnh báo hồ Núi Cốc |

---

## 📈 Chỉ số đánh giá

Dự báo tốt khi **NSE ≥ 0.75** (thang đánh giá thủy văn chuẩn quốc tế):

| Chỉ số | Ý nghĩa |
|--------|---------|
| **NSE** (Nash-Sutcliffe) | NSE=1: hoàn hảo, NSE=0: ngang bằng trung bình |
| **RMSE** | Sai số căn bậc hai trung bình (m) |
| **MAE** | Sai số tuyệt đối trung bình (m) |

---

## 🔗 Nguồn dữ liệu

- **NASA POWER**: [power.larc.nasa.gov](https://power.larc.nasa.gov) — Khí tượng giờ
- **Copernicus Sentinel-2**: [GEE — COPERNICUS/S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) — Ảnh vệ tinh 10m
- **Hồ Núi Cốc**: Tỉnh Thái Nguyên, Việt Nam (21.64°N–21.73°N, 105.68°E–105.78°E)

---

## 📄 License

MIT License — xem [LICENSE](LICENSE)

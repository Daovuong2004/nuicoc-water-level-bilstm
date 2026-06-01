# Du bao muc nuoc ho Nui Coc — Bi-LSTM + Self-Attention

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2-brightgreen)](https://earthengine.google.com)

> **Do an tot nghiep** — Nganh Cong nghe Thong tin (Ky thuat Phan mem)
> De tai: *He thong du bao muc nuoc ho Nui Coc su dung kien truc mo hinh Bi-LSTM + Self-Attention*

---

## Tong quan

Pipeline hoc may end-to-end du bao muc nuoc ho Nui Coc (Thai Nguyen) cho cac chan troi thoi gian **t+1d, t+3d, t+7d, t+14d, t+30d** su dung mo hinh **Bidirectional LSTM + Multi-Head Self-Attention** ket hop **Monte Carlo Dropout** de uoc luong do khong chac chan.

### Kien truc he thong

```
Nguon du lieu                    Pipeline                       Dau ra
───────────────────────────────────────────────────────────────────────────
NASA POWER API   ─┐
GEE Sentinel-2   ─┤─► 05_integrate.py ──► 06_bilstm_model.py ──► Du bao muc nuoc
Bao chi/Cua xa   ─┘   (Kalman + Q_out)   (Bi-LSTM+Attention)    + Khoang tin cay 95%
                                                │
                                       08_api_serve.py ──► REST API (FastAPI)
```

### Diem noi bat ky thuat

| Thanh phan | Ky thuat |
|-----------|---------|
| **Mo hinh** | Bi-LSTM 2 lop + Multi-Head Self-Attention (4 heads, key_dim=32) + Residual + LayerNorm |
| **Uncertainty** | Monte Carlo Dropout (50 samples) → khoang tin cay 95% cho tung du bao |
| **XAI** | SHAP Feature Importance — giai thich "mo hinh dua vao gi" |
| **Du lieu ve tinh** | Sentinel-2 SR + QA60 cloud mask → NDWI → duong cong A-H |
| **Thuy van** | Phuong trinh can bang nuoc: Q_out = -A(H) × dH/dt / 86400 |
| **Noi suy** | PCHIP interpolation (bao toan don dieu cuc bo) cho gap ≤ 60 ngay |
| **Anti-leakage** | MinMaxScaler chi fit tren tap train |
| **Danh gia** | Nash-Sutcliffe Efficiency (NSE) — chi so chuan thuy van |
| **Serving** | FastAPI + MC Dropout inference + canh bao lu tu dong |

---

## Cau truc du an

```
DATN/
├── 01_nasa_power.py            # Buoc 1: Thu thap du lieu khi tuong NASA POWER
├── 02_gee_sentinel2.py         # Buoc 2: Xu ly anh ve tinh GEE (offline)
├── 02_gee_sentinel2_v2.py      # Buoc 2 v2: Pipeline GEE cai tien
├── 02_gee_colab.py             # Buoc 2b: Trich xuat GEE tren Google Colab
├── 03_04_cua_xa_bao_chi.py     # Buoc 3+4: Cua xa + su kien lu tu bao chi
├── 05_integrate.py             # Buoc 5: Tich hop + Kalman + Q_out + normalize
├── 06_bilstm_model.py          # Buoc 6: Huan luyen Bi-LSTM + Attention + SHAP
├── 06b_baseline_comparison.py  # Buoc 6b: Ablation study SARIMA/LSTM/BiLSTM
├── 07_infer_qout_1.py          # Buoc 7: Phan tich Q_out (tuy chon)
├── 08_api_serve.py             # Buoc 8: FastAPI inference server
├── run_all.py                  # Chay toan bo pipeline mot lenh
│
├── data/                       # (gitignored)
│   ├── raw/                    #   NASA POWER, GEE, cua xa
│   ├── processed/              #   Du lieu trung gian
│   └── final/                  #   dataset_train/val/test/full.csv
│
├── models/                     # (gitignored)
│   ├── bilstm_t1d.keras        #   Mo hinh du bao t+1 ngay
│   ├── bilstm_t3d.keras
│   ├── bilstm_t7d.keras
│   ├── bilstm_t14d.keras
│   ├── bilstm_t30d.keras
│   └── feature_scaler_daily.pkl
│
├── results/                    # (gitignored)
│   ├── plot_t*d.png            #   Bieu do du bao + CI95%
│   ├── shap_importance_*.png   #   SHAP Feature Importance
│   ├── predictions_t*d.csv     #   Ket qua du bao
│   └── metrics_summary.json    #   RMSE / MAE / NSE
│
├── .gitignore
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## Cai dat & Chay

### 1. Yeu cau he thong
- Python 3.10+
- RAM >= 8GB (de huan luyen BiLSTM)
- GPU (tuy chon, TensorFlow tu dong dung GPU neu co)
- Tai khoan Google Earth Engine (cho buoc 2 tren Colab)

### 2. Cai thu vien

```bash
pip install -r requirements.txt
```

Cac thu vien chinh can cai:
- `tensorflow>=2.13`, `keras>=3.0`
- `fastapi>=0.110`, `uvicorn[standard]>=0.29`, `pydantic>=2.0`
- `scikit-learn`, `pandas`, `numpy`, `scipy`, `joblib`
- `shap>=0.44`, `statsmodels>=0.14`, `matplotlib`, `seaborn`

### 3. Chay pipeline day du

```bash
# Option A: Chay tat ca cac buoc tu dong
python run_all.py

# Option B: Chay tung buoc
python 01_nasa_power.py          # Thu thap khi tuong
# Buoc 2: Chay 02_gee_colab.py tren Google Colab
python 03_04_cua_xa_bao_chi.py   # Xu ly bao chi
python 05_integrate.py           # Tich hop du lieu
python 06_bilstm_model.py        # Huan luyen Bi-LSTM
python 06b_baseline_comparison.py # Ablation study (tuy chon)
```

### 4. Khoi dong API Server

```bash
# Su dung uvicorn (khuyen nghi)
uvicorn 08_api_serve:app --reload --host 0.0.0.0 --port 8000

# Hoac chay truc tiep
python 08_api_serve.py
```

Sau khi khoi dong, truy cap:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

---

## Bo dac trung dau vao (21 features)

| Nhom | Features | Mo ta |
|------|---------|-------|
| **Khi tuong** | `rain_1d`, `rain_3d`, `rain_7d`, `rain_14d` | Luong mua tich luy (mm) |
| **Khi tuong** | `temperature`, `humidity` | Nhiet do (°C), Do am (%) |
| **Lag muc nuoc** | `water_level_lag1/3/7/14/30` | Muc nuoc tre 1/3/7/14/30 ngay (m) |
| **Rolling stats** | `water_level_roll7`, `water_level_roll30` | Trung binh truot 7/30 ngay (m) |
| **Rolling stats** | `water_level_std7` | Do lech chuan muc nuoc 7 ngay (m) |
| **Temporal** | `month_sin`, `month_cos` | Ma hoa tuan hoan thang |
| **Temporal** | `season_wet`, `season_dry` | Mua mua/kho (0/1) |
| **Q_out** | `dH_dt_daily` | Toc do thay doi muc nuoc (m/ngay) |
| **Q_out** | `Q_out_daily`, `Q_out_roll7` | Luu luong xa uoc tinh (m³/s) |

**Cua so thoi gian**: 30 ngay | **Chan troi du bao**: 1, 3, 7, 14, 30 ngay

---

## Phan chia du lieu

| Tap | Giai doan | So ngay | Muc dich |
|-----|-----------|---------|---------|
| **Train** | 2017-01 → 2022-12 | ~2190 | Huan luyen (co data tong hop 2017-2019) |
| **Validation** | 2023-01 → 2023-12 | ~365 | Tuning sieu tham so |
| **Test** | 2024-01 → 2025-12 | ~730 | Danh gia — bao gom lu Yagi thang 9/2024 |

> **Ghi chu**: Du lieu 2017-2019 duoc tong hop tu seasonal pattern cua GEE thuc (Gaussian smoothing, sigma=7 ngay) vi GEE Sentinel-2 chi co ~80 diem quan trac thuc giai doan 2019-2025.

---

## Kien truc mo hinh Bi-LSTM + Self-Attention

```
Input (30, 21)
    │
    ▼
BiLSTM(128, return_sequences=True) ── Dropout(0.2) ── BatchNorm
    │
    ▼
Multi-Head Self-Attention(heads=4, key_dim=32, dropout=0.2)
    │
    ▼
Residual Add + LayerNorm           ◄── Skip connection tu BiLSTM(128)
    │
    ▼
BiLSTM(64, return_sequences=False) ── Dropout(0.2) ── BatchNorm
    │
    ▼
Dense(32, relu) → Dense(1, linear)
    │
    ▼
Output: Muc nuoc du bao (m)
```

**Ly do dung Self-Attention trong thuy van:**
Muc nuoc luc 8h sang bi anh huong boi con mua luc 20h hom qua (lag xa). LSTM xu ly kem khi "buoc thoi gian quan trong" nam xa trong sequence. Self-Attention giai quyet dieu nay: moi timestep "nhin" tat ca cac timestep khac va hoc trong so quan trong.

---

## API Endpoints

| Method | Endpoint | Mo ta |
|--------|----------|-------|
| `POST` | `/predict` | Du bao muc nuoc t+1/3/7/14/30d + khoang tin cay 95% |
| `GET` | `/health` | Trang thai server + cac model da tai |
| `GET` | `/features` | Danh sach 21 features theo dung thu tu |
| `GET` | `/thresholds` | Nguong canh bao lu ho Nui Coc |

### Vi du goi API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[0.0]*21]*30,
    "timestamp": "2026-06-01T00:00:00+07:00"
  }'
```

### Nguong canh bao lu ho Nui Coc

| Muc canh bao | Nguong muc nuoc | Hanh dong |
|-------------|----------------|---------|
| BINH THUONG | ≤ 46.80m | Van hanh binh thuong |
| CANH BAO | 46.80m – 47.40m | Theo doi lien tuc, san sang xa |
| NGUY HIEM | > 47.40m | Mo toan bo cua xa, so tan ha luu |

---

## Chi so danh gia

Danh gia theo tieu chuan thuy van quoc te (NSE ≥ 0.75 = Tot):

| Chi so | Y nghia |
|--------|---------|
| **NSE** (Nash-Sutcliffe) | 1.0: hoan hao | 0.0: ngang trung binh | <0: kem |
| **RMSE** | Sai so can bac hai trung binh (m) |
| **MAE** | Sai so tuyet doi trung binh (m) |

---

## Nguon du lieu

- **NASA POWER**: [power.larc.nasa.gov](https://power.larc.nasa.gov) — Khi tuong theo gio
- **Copernicus Sentinel-2**: [GEE — COPERNICUS/S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) — Anh ve tinh 10m
- **Ho Nui Coc**: Tinh Thai Nguyen, Viet Nam (21.64°N–21.73°N, 105.68°E–105.78°E)

---

## Troubleshooting

**Loi `ModuleNotFoundError: No module named 'fastapi'`**
```bash
pip install fastapi uvicorn[standard] pydantic
```

**VS Code bao loi import nhung da cai roi**
→ Kiem tra dung Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → chon `Python310`

**Loi encoding Unicode khi chay tren Windows**
```bash
$env:PYTHONIOENCODING="utf-8"; python run_all.py
```

---

## License

MIT License — xem [LICENSE](LICENSE)

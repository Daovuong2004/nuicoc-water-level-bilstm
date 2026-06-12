# Du bao muc nuoc ho Nui Coc — Bi-LSTM

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2-brightgreen)](https://earthengine.google.com)

> **Do an tot nghiep** — Nganh Cong nghe Thong tin (Ky thuat Phan mem)
> De tai: *He thong du bao muc nuoc ho Nui Coc su dung kien truc mo hinh Bi-LSTM*

---

## Tong quan

Pipeline hoc may end-to-end du bao muc nuoc ho Nui Coc (Thai Nguyen) cho cac chan troi thoi gian **t+1d, t+3d, t+7d, t+14d, t+30d** su dung mo hinh **Bidirectional LSTM** ket hop **Monte Carlo Dropout** de uoc luong do khong chac chan.

### Kien truc he thong

```
Nguon du lieu                    Pipeline                       Dau ra
───────────────────────────────────────────────────────────────────────────
NASA POWER API   ─┐
GEE Sentinel-2   ─┤─► 05_integrate.py ──► 06_bilstm_model.py ──► Du bao muc nuoc
Bao chi/Cua xa   ─┘   (Kalman + Q_out)   (Bi-LSTM)              + Khoang tin cay 95%
                                                │
                                       08_api_serve.py ──► REST API (FastAPI)
```

### Diem noi bat ky thuat

| Thanh phan | Ky thuat |
|-----------|---------|
| **Mo hinh** | Bidirectional LSTM (64 units/chieu → 128 output) + Dense(32) + L2(1e-3) |
| **Uncertainty** | Monte Carlo Dropout (50 samples) → khoang tin cay 95% cho tung du bao |
| **XAI** | SHAP Feature Importance (GradientExplainer) — giai thich "mo hinh dua vao gi" |
| **Du lieu ve tinh** | Sentinel-2 SR + QA60 cloud mask → NDWI → duong cong A-H |
| **Thuy van** | Phuong trinh can bang nuoc: Q_out = -A(H) × dH/dt / 86400 |
| **Noi suy** | PCHIP interpolation (bao toan don dieu cuc bo) cho gap ≤ 60 ngay |
| **Anti-leakage** | StandardScaler chi fit tren tap train; EarlyStopping dung 15% cuoi train |
| **Danh gia** | NSE (Nash-Sutcliffe) + RMSE + MAE + PBIAS + F1-Score nguong van hanh |
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
├── 06_bilstm_model.py          # Buoc 6: Huan luyen Bi-LSTM + SHAP
├── 06b_baseline_comparison.py  # Buoc 6b: Ablation study SARIMA/LSTM/GRU/BiLSTM
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

## Bo dac trung dau vao

### Bo features chuan (16 features — dung cho t+1d, t+3d, t+14d, t+30d)

| Nhom | Features | Mo ta |
|------|---------|-------|
| **Khi tuong** | `rain_1d`, `rain_3d`, `rain_7d`, `rain_14d`, `rain_30d` | Luong mua tich luy (mm) |
| **Khi tuong** | `temperature`, `humidity` | Nhiet do (°C), Do am (%) |
| **Lag muc nuoc** | `water_level_lag7`, `water_level_lag14`, `water_level_lag30` | Muc nuoc tre 7/14/30 ngay (m) |
| **Rolling stats** | `water_level_roll7`, `water_level_std7` | TB truot 7 ngay & Do lech chuan (m) |
| **Temporal** | `month_sin`, `month_cos` | Ma hoa tuan hoan thang (cyclical encoding) |
| **Temporal** | `season_wet`, `season_dry` | Mua mua (thang 5-10) / kho (thang 11-4) |

### Bo features mo rong (21 features — dung rieng cho t+7d)

Giu nguyen 16 features tren, bo sung them:

| Feature | Mo ta |
|---------|-------|
| `rain_60d` | Mua tich luy 60 ngay — nhan biet dau/cuoi mua lu |
| `water_level_lag60` | Muc nuoc 60 ngay truoc — so sanh xu huong dai han |
| `water_level_roll30` | Trung binh truot 30 ngay — xu huong ho on dinh |
| `delta_h_7d` | H(t) - H(t-7) — momentum tang/giam 7 ngay gan nhat |
| `delta_h_30d` | H(t) - H(t-30) — momentum tang/giam 1 thang gan nhat |

**Cua so thoi gian**: 21 ngay (t+1/3/14/30d) | 45 ngay (t+7d) | **Chan troi du bao**: 1, 3, 7, 14, 30 ngay

---

## Phan chia du lieu

| Ten tap (file) | Ten trong bao cao | Giai doan | So ngay | Muc dich |
|---------------|------------------|-----------|---------|----------|
| `dataset_train.csv` | **Train** | 2019-04 → 2022-12 | ~1340 | Huan luyen toan bo tham so Bi-LSTM |
| `dataset_test.csv` | **EarlyStopping-Val** | 2023-01 → 2023-12 | ~365 | Dieu chinh EarlyStopping & ReduceLROnPlateau |
| `dataset_val.csv` | **Evaluation Set** | 2024-01 → 2025-12 | ~730 | **Ket qua chinh thuc trong luan van** (bao gom lu Yagi 9/2024) |

> **Luu y quan trong ve dat ten**: File `dataset_test.csv` (nam 2023) duoc dung cho EarlyStopping noi bo — **khong phai ket qua bao cao**. File `dataset_val.csv` (nam 2024+) la tap **kiem dinh doc lap cuoi cung** duoc bao cao trong luan van.
>
> **Du lieu tong hop**: Giai doan 2017-2019 duoc bo sung bang du lieu thuy van tong hop tu seasonal pattern cua GEE thuc (Gaussian smoothing, sigma=7 ngay), vi GEE Sentinel-2 chi co ~80 diem quan trac thuc giai doan 2019-2025. Trong so mau: quan trac thuc=1.0, noi suy/tong hop=0.25.

---

## Kien truc mo hinh Bi-LSTM

### Kien truc chuan (t+1d, t+3d, t+14d, t+30d)

```
Input (window=21 ngay, 16 features)
    │
    ▼
Bidirectional(LSTM(64 units, recurrent_dropout=0.2))
  ├─ Forward LSTM(64)  →┐
  └─ Backward LSTM(64) →┘ Concat → output 128 chieu
    │
    ▼
Dropout(0.5)
    │
    ▼
Dense(32, activation='relu') + L2(1e-3)
    │
    ▼
Dense(1, activation='linear')   ← Du bao ΔH (scaled)
    │
    ▼
Hau xu ly: ΔH → H(t+d) = H(t) + ΔH (inverse_transform)
```

### Kien truc mo rong rieng cho t+7d

```
Input (window=45 ngay, 21 features)
    │
    ▼
Bidirectional(LSTM(96 units, recurrent_dropout=0.2))
  ├─ Forward LSTM(96)  →┐
  └─ Backward LSTM(96) →┘ Concat → output 192 chieu
    │
    ▼
Dropout(0.3)   ← Giam regularization vi pattern dai han on dinh hon
    │
    ▼
Dense(64, activation='relu') + L2(5e-4)
    │
    ▼
Dense(1, activation='linear')   ← Du bao ΔH (scaled)
```

**Ly do dung Bi-LSTM trong thuy van:**
Bi-LSTM xu ly chuoi thoi gian theo ca hai chieu (xuoi va nguoc). Chieu xuoi giup hoc xu huong tich luy mua va tang muc nuoc. Chieu nguoc giup nam bat cac quy luat chu ky mua kho. Ket qua ablation study: Bi-LSTM (RMSE=0.374m, NSE=0.988) vuot troi LSTM don huong (RMSE=0.482m, NSE=0.981) va GRU (RMSE=0.480m, NSE=0.981) tai t+1d.

**Tham so huan luyen:**
| Tham so | Gia tri |
|---------|--------|
| Loss function | Huber(delta=1.0) — it nhay voi outlier dinh lu |
| Optimizer | Adam(lr=0.001) |
| Batch size | 32 |
| Max epochs | 150 (EarlyStopping patience=20) |
| Target scaling | StandardScaler (ho tro ngoai suy dinh lu) |

---

## API Endpoints

| Method | Endpoint | Mo ta |
|--------|----------|-------|
| `POST` | `/predict` | Du bao muc nuoc t+1/3/7/14/30d + khoang tin cay 95% |
| `GET` | `/health` | Trang thai server + cac model da tai |
| `GET` | `/features` | Danh sach 26 features theo dung thu tu |
| `GET` | `/thresholds` | Nguong canh bao lu ho Nui Coc |

### Vi du goi API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[0.0]*26]*60,
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

Danh gia theo tieu chuan thuy van quoc te (Moriasi et al., 2007 & WMO):

| Chi so | Cong thuc | Nguong Tot |
|--------|-----------|------------|
| **NSE** (Nash-Sutcliffe Efficiency) | `1 - Σ(obs-sim)²/Σ(obs-mean)²` | NSE ≥ 0.75 |
| **RMSE** (Root Mean Squared Error) | `√(mean((obs-sim)²))` | cang nho cang tot (m) |
| **MAE** (Mean Absolute Error) | `mean(|obs-sim|)` | cang nho cang tot (m) |
| **PBIAS** (Percent Bias) | `100×Σ(sim-obs)/Σ(obs)` | \|PBIAS\| < 10% = Tot |
| **F1-Score** (phan loai vuot nguong) | Precision-Recall tren nhi phan | cang cao cang tot |

### Ket qua tren tap Evaluation (2024-2025, bao gom lu Yagi 9/2024)

| Horizon | RMSE (m) | MAE (m) | NSE | PBIAS (%) | Danh gia |
|---------|---------|--------|-----|----------|----------|
| **t+1d** | 0.358 | 0.219 | **0.989** | 0.032 | ✅ Xuat sac |
| **t+3d** | 0.965 | 0.570 | **0.922** | 0.093 | ✅ Tot |
| **t+7d** | 2.061 | 1.205 | 0.643 | 0.443 | 🟡 Kha |
| t+14d | 3.147 | 2.388 | 0.163 | 0.308 | 🔴 Yeu |
| t+30d | 4.556 | 3.893 | -0.702 | 3.618 | 🔴 Rat yeu |

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

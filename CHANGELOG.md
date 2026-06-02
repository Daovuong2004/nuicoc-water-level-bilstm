# Changelog — He thong du bao muc nuoc ho Nui Coc

Tat ca thay doi dang ke cua du an duoc ghi lai tai day.
Dinh dang: [Semantic Versioning](https://semver.org/)

---

## [3.0.0] — 2026-05-28

### Added
- Pipeline tuyen so **Ngay (Daily)** thay the pipeline Gio (Hourly) cua v2.0
  - Ly do: GEE Sentinel-2 chi co ~80 diem quan trac thuc (2019-2025); xay chuoi gio tu 80 diem → ~80% la noi suy, khong dang tin cay
  - Giai phap: tan so NGAY + augmentation thuy van (2017-2019) de dat >=300 diem training
- **21 features** moi (thay 18 features cu):
  - Them `dH_dt_daily` — toc do thay doi muc nuoc (m/ngay)
  - Them `Q_out_daily` — luu luong xa uoc tinh tu phuong trinh can bang nuoc
  - Them `Q_out_roll7` — trung binh truot luu luong xa 7 ngay
- **Chan troi du bao moi**: t+1d, t+3d, t+7d, t+14d, t+30d (thay t+1h/3h/6h/12h/24h)
- **PCHIP interpolation** thay Kalman Filter cho noi suy muc nuoc (bao toan don dieu cuc bo)
- **Augmentation du lieu thuy van** 2017-2019 bang phuong phap seasonal + Gaussian smoothing
- **02_gee_sentinel2_v2.py**: pipeline GEE v2 voi duong cong A-H chinh xac hon
- **CHANGELOG.md**: file nay

### Changed
- `05_integrate.py` (v3.0): viet lai hoan toan — daily pipeline thay hourly
  - Augmentation 2017-2019 bang seasonal pattern tu GEE thuc (good/fair quality)
  - PCHIP interpolation cho gap <= 60 ngay
  - Q_out daily tu phuong trinh can bang nuoc: Q_out = -A(H) * dH/dt / 86400
  - Phan chia: Train 2017-2022 | Val 2023 | Test 2024-2025
- `06_bilstm_model.py` (v3.0): cap nhat phu hop pipeline daily
  - Window size: 30 ngay (thay 48 gio)
  - Cap nhat docstring va hyperparameters
- `08_api_serve.py`: sua loi `NameError: FORECAST_HOURS` trong endpoint `/health`
  - Thay the `FORECAST_HOURS` → `FORECAST_DAYS` (bien dung)
  - Cap nhat mo ta 21 features trong FEATURE_COLS
- `README.md`: cap nhat chinh xac theo code thuc te v3.0

### Fixed
- `08_api_serve.py` dòng 537: `NameError: name 'FORECAST_HOURS' is not defined`
  - Endpoint `/health` se crash khi duoc goi neu khong sua
- Loi encoding Unicode khi chay tren Windows (cp1252 khong ho tro tieng Viet)

---

## [2.0.0] — 2026-05-10

### Added
- **Bi-LSTM nâng cao** với siêu tham số tối ưu (dropout, L2 regularization) thay thế LSTM đơn giản
- **Monte Carlo Dropout** (50 samples) để ước lượng khoảng tin cậy 95%
  - Thay vì nói "mực nước t+7d là 46.5m", mô hình nói "46.5m ± 0.3m (95% CI)"
- **SHAP Feature Importance** (GradientExplainer) — giải thích mô hình (XAI)
  - Tính cho t+1d và t+30d để tiết kiệm thời gian
- **06b_baseline_comparison.py**: ablation study so sánh 4 mô hình
  - SARIMA | LSTM | GRU | Bi-LSTM
- **08_api_serve.py**: FastAPI server với endpoint /predict, /health, /features, /thresholds
- **run_all.py**: chạy toàn bộ pipeline một lệnh
- Q_out features mới: `Q_out_smooth`, `Q_out_lag1/6`, `Q_out_roll24`, `dQout_dt`, `xa_dot_ngot`

### Changed
- Mô hình nâng cấp từ LSTM đơn giản → Bi-LSTM nâng cao
- Chia dữ liệu: thêm tập Validation riêng biệt (chống data leakage)

---

## [1.0.0] — 2026-04-15

### Added
- **01_nasa_power.py**: Thu thap du lieu khi tuong NASA POWER API (co retry + checkpoint)
- **02_gee_sentinel2.py**: Xu ly anh ve tinh GEE Sentinel-2 → NDWI → duong cong A-H → muc nuoc
- **03_04_cua_xa_bao_chi.py**: Trich xuat su kien lu va van hanh cua xa tu bao chi
- **05_integrate.py** (v1.0): Tich hop du lieu, Kalman Filter, chuan hoa
- **06_bilstm_model.py** (v1.0): Mo hinh Bi-LSTM 2 lop don gian
- **requirements.txt**, **.gitignore**, **README.md** co ban
- **data/**: Cau truc thu muc du lieu
- **models/**, **results/**: Thu muc luu mo hinh va ket qua

---

## Ke hoach tuong lai

- [ ] Dashboard web truc quan hoa du bao theo thoi gian thuc
- [ ] Tich hop du lieu mua tu tram quan trac mat dat (IMHEN)
- [ ] Mo hinh ensemble: Bi-LSTM + XGBoost
- [ ] Docker container de deploy API len cloud
- [ ] Unit tests cho cac module xu ly du lieu

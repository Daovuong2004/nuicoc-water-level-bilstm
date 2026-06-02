"""
================================================================================
08_api_serve.py — FastAPI Inference Server
Hệ thống dự báo mực nước hồ Núi Cốc (Thái Nguyên)
Mô hình: Bi-LSTM với Monte Carlo Dropout
================================================================================

Mô tả:
    Server cung cấp API REST để dự báo mực nước hồ Núi Cốc theo nhiều chân trời
    thời gian (1h, 3h, 6h, 12h, 24h) dựa trên dữ liệu 48 giờ gần nhất.
    Kết quả bao gồm dự báo điểm, khoảng tin cậy 95% (Monte Carlo Dropout),
    và cảnh báo ngưỡng lũ.

Cách chạy:
    uvicorn 08_api_serve:app --reload --port 8000

Các endpoint:
    POST /predict     — Dự báo mực nước cho các chân trời thời gian
    GET  /health      — Kiểm tra trạng thái server và các model đã load
    GET  /features    — Danh sách và thứ tự các đặc trưng đầu vào
    GET  /thresholds  — Ngưỡng cảnh báo lũ hồ Núi Cốc

Tài liệu tương tác (Swagger UI):
    http://localhost:8000/docs

Tác giả: Đồ án tốt nghiệp — Dự báo mực nước hồ Núi Cốc
Ngày:    2026-05-20
================================================================================
"""

# ---------------------------------------------------------------------------
# Thư viện chuẩn
# ---------------------------------------------------------------------------
import os
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Thư viện bên ngoài
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import joblib
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from keras.models import Model, load_model  # type: ignore
from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2

# ---------------------------------------------------------------------------
# Cấu hình logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api_serve")

# ---------------------------------------------------------------------------
# Hằng số: Đặc trưng đầu vào
# ---------------------------------------------------------------------------
# 26 đặc trưng ngày theo đúng thứ tự đã dùng khi huấn luyện model v4.0
FEATURE_COLS = [
    # Khí tượng (7 features)
    "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
    "temperature", "humidity",
    # Lag mực nước (6 features)
    "water_level_lag1", "water_level_lag3", "water_level_lag7",
    "water_level_lag14", "water_level_lag30", "water_level_lag60",
    # Rolling stats (4 features)
    "water_level_roll7", "water_level_roll30", "water_level_roll60",
    "water_level_std7",
    # Temporal (4 features)
    "month_sin", "month_cos", "season_wet", "season_dry",
    # Xu hướng thủy văn (2 features)
    "delta_h_7d", "delta_h_30d",
    # Q_out (3 features)
    "dH_dt_daily", "Q_out_daily", "Q_out_roll7",
]

# Số đặc trưng đầu vào
FEATURE_COUNT = len(FEATURE_COLS)  # 26

# Kích thước cửa sổ thời gian (ngày)
WINDOW_SIZE = 60

# Siêu tham số L2 regularization
L2_REG = 1e-4

# Các chân trời dự báo (ngày)
FORECAST_DAYS = [1, 3, 7, 14, 30]

# ---------------------------------------------------------------------------
# Ngưỡng cảnh báo hồ Núi Cốc (đơn vị: mét)
# ---------------------------------------------------------------------------
MNDBT = 46.20       # Mực nước dâng bình thường (m)
CANH_BAO = 46.80    # Ngưỡng cảnh báo — theo dõi liên tục, sẵn sàng xả (m)
NGUY_HIEM = 47.40   # Ngưỡng nguy hiểm — mở toàn bộ cửa xả, sơ tán hạ lưu (m)

# ---------------------------------------------------------------------------
# Đường dẫn model và scaler
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler_daily.pkl")


# ---------------------------------------------------------------------------
# Kiến trúc mô hình Bi-LSTM để rebuild cho MC Dropout
# ---------------------------------------------------------------------------
def build_bilstm(input_shape: tuple, lstm_units: list = [256, 128],
                  dropout_rate: float = 0.25, mc_dropout: bool = False) -> Model:
    inputs = Input(shape=input_shape, name="input_sequence")

    # ── Lớp BiLSTM 1 ──
    x = Bidirectional(
        LSTM(lstm_units[0], return_sequences=True, name="bilstm_1"),
        name="bidirectional_1",
    )(inputs)
    if mc_dropout:
        x = Dropout(dropout_rate, name="dropout_1")(x, training=True)
    else:
        x = Dropout(dropout_rate, name="dropout_1")(x)

    # ── Lớp BiLSTM 2 ──
    x = Bidirectional(
        LSTM(lstm_units[1], return_sequences=False, name="bilstm_2"),
        name="bidirectional_2",
    )(x)
    if mc_dropout:
        x = Dropout(dropout_rate, name="dropout_2")(x, training=True)
    else:
        x = Dropout(dropout_rate, name="dropout_2")(x)

    # ── Lớp Dense ──
    x       = Dense(64, activation="relu", kernel_regularizer=l2(L2_REG),
                    name="dense_1")(x)
    x       = Dense(32, activation="relu", name="dense_2")(x)
    outputs = Dense(1,  activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="BiLSTM_v4")


# ---------------------------------------------------------------------------
# Hàm tải model và scaler khi khởi động
# ---------------------------------------------------------------------------
def load_models_and_scaler() -> tuple[dict, object, dict]:
    """
    Tải toàn bộ model Bi-LSTM, feature scaler và các target scalers từ thư mục models/.

    Quy trình:
        1. Load scaler (MinMaxScaler) từ models/feature_scaler_daily.pkl
        2. Với mỗi chân trời d trong FORECAST_DAYS, load
           models/bilstm_t{d}d.keras (nếu tồn tại)
        3. Load target scaler tương ứng từ models/target_scaler_t{d}d.pkl
        4. Model hoặc scaler bị thiếu sẽ bị bỏ qua và ghi log warning
        5. Rebuild model với mc_dropout=True để sửa lỗi Dropout trong Keras

    Returns:
        models_dict (dict): {horizon_d: keras_model}
        scaler: fitted MinMaxScaler hoặc None nếu không tìm thấy file
        target_scalers (dict): {horizon_d: target_scaler}
    """
    models_dict = {}
    target_scalers = {}

    # ---- Tải feature scaler ----
    scaler = None
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            logger.info("Đã tải scaler thành công: %s", SCALER_PATH)
        except Exception as exc:
            logger.warning("Không thể tải scaler từ %s: %s", SCALER_PATH, exc)
    else:
        logger.warning(
            "Không tìm thấy file scaler tại %s — endpoint /predict sẽ không hoạt động.",
            SCALER_PATH,
        )

    # ---- Tải từng model và target scaler theo chân trời dự báo ----
    for d in FORECAST_DAYS:
        model_path = os.path.join(MODEL_DIR, f"bilstm_t{d}d.keras")
        if os.path.exists(model_path):
            try:
                trained_model = load_model(model_path)
                models_dict[d] = trained_model
                logger.info("Đã tải model t%dd thành công: %s", d, model_path)
            except Exception as exc:
                logger.warning(
                    "Không thể tải model t%dd từ %s: %s", d, model_path, exc
                )
        else:
            logger.warning(
                "Không tìm thấy file model t%dd tại %s — bỏ qua chân trời này.",
                d,
                model_path,
            )

        # Tải target scaler
        target_scaler_path = os.path.join(MODEL_DIR, f"target_scaler_t{d}d.pkl")
        if os.path.exists(target_scaler_path):
            try:
                target_scalers[d] = joblib.load(target_scaler_path)
                logger.info("Đã tải target scaler t%dd thành công: %s", d, target_scaler_path)
            except Exception as exc:
                logger.warning("Không thể tải target scaler t%dd từ %s: %s", d, target_scaler_path, exc)
        else:
            logger.warning("Không tìm thấy target scaler t%dd tại %s", d, target_scaler_path)

    logger.info(
        "Khởi động hoàn tất — %d model và %d target scaler sẵn sàng: %s",
        len(models_dict),
        len(target_scalers),
        sorted(models_dict.keys()),
    )
    return models_dict, scaler, target_scalers


# ---------------------------------------------------------------------------
# Biến toàn cục — được điền khi app khởi động
# ---------------------------------------------------------------------------
# Sử dụng dict để chứa state, tránh vấn đề với global trong FastAPI
app_state: dict = {
    "models": {},          # {horizon_d (int): keras model}
    "scaler": None,        # fitted MinMaxScaler cho features
    "target_scalers": {},  # {horizon_d (int): MinMaxScaler cho target}
}


# ---------------------------------------------------------------------------
# Pydantic models — Request / Response
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    """
    Payload đầu vào cho endpoint POST /predict.

    Attributes:
        features:   Ma trận đặc trưng shape (60, 26) — dữ liệu 60 ngày gần nhất.
                    Thứ tự cột phải khớp với FEATURE_COLS.
        timestamp:  Thời điểm quan sát cuối cùng (ISO 8601), tùy chọn.
                    Ví dụ: "2026-05-20T14:00:00+07:00"
    """
    features: list[list[float]] = Field(
        ...,
        description=(
            "Ma trận đặc trưng shape (60, 26). "
            "Hàng = bước thời gian (ngày), Cột = đặc trưng theo thứ tự FEATURE_COLS."
        ),
        example=[[0.0] * 26] * 60,
    )
    timestamp: str | None = Field(
        default=None,
        description="Thời điểm quan sát cuối cùng (ISO 8601), tùy chọn.",
        example="2026-05-20T14:00:00+07:00",
    )


class HorizonForecast(BaseModel):
    """Dự báo mực nước cho một chân trời thời gian cụ thể."""
    horizon_d: int = Field(..., description="Chân trời dự báo (ngày)")
    water_level_m: float = Field(..., description="Mực nước dự báo trung bình (m)")
    ci95_lower: float = Field(..., description="Cận dưới khoảng tin cậy 95% (m)")
    ci95_upper: float = Field(..., description="Cận trên khoảng tin cậy 95% (m)")


class ForecastResponse(BaseModel):
    """Phản hồi đầy đủ từ endpoint POST /predict."""
    request_time: str = Field(..., description="Thời điểm xử lý request (ISO 8601 UTC)")
    forecasts: list[HorizonForecast] = Field(..., description="Danh sách dự báo theo chân trời")
    alert_level: str = Field(
        ...,
        description="Mức cảnh báo: 'BÌNH THƯỜNG' | 'CẢNH BÁO' | 'NGUY HIỂM'",
    )
    alert_message: str = Field(..., description="Thông điệp cảnh báo chi tiết")
    models_used: list[int] = Field(..., description="Danh sách chân trời đã có model dự báo (ngày)")


class HealthResponse(BaseModel):
    """Phản hồi từ endpoint GET /health."""
    status: str = Field(..., description="Trạng thái server: 'ok' hoặc 'degraded'")
    models_loaded: list[int] = Field(..., description="Chân trời (ngày) đã có model sẵn sàng")
    feature_count: int = Field(..., description="Số đặc trưng đầu vào")
    window_size: int = Field(..., description="Kích thước cửa sổ thời gian (ngày)")
    scaler_loaded: bool = Field(..., description="Scaler đã được tải thành công hay chưa")


# ---------------------------------------------------------------------------
# Hàm dự báo MC Dropout
# ---------------------------------------------------------------------------

def predict_with_mc_dropout(
    model: Model,
    X_input: np.ndarray,
    target_scaler: object = None,
    n_samples: int = 50,
) -> tuple[float, float, float]:
    """
    Dự báo mực nước sử dụng kết hợp dự báo tất định và MC Dropout.

    Chiến lược:
        - Điểm dự báo chính (Point Forecast) là dự báo tất định (training=False) để tối ưu NSE.
        - Ước lượng khoảng tin cậy 95% dựa trên độ lệch chuẩn của các mẫu MC Dropout (training=True).
        - Nghịch đảo chuẩn hóa kết quả về đơn vị mét (m).

    Args:
        model:          Keras model đã được load
        X_input:        Mảng numpy shape (1, WINDOW_SIZE, FEATURE_COUNT)
        target_scaler:  MinMaxScaler dùng cho target (nếu có)
        n_samples:      Số lần lấy mẫu Monte Carlo (mặc định 50)

    Returns:
        mean (float):       Mực nước dự báo tất định (m)
        ci95_lower (float): Cận dưới khoảng tin cậy 95% (m)
        ci95_upper (float): Cận trên khoảng tin cậy 95% (m)
    """
    # 1. Dự báo tất định (Point Forecast)
    pred_scaled = float(np.squeeze(model(X_input, training=False).numpy()))

    # 2. Thu thập n_samples dự báo từ MC Dropout (bật training=True) - Sử dụng batching để tối ưu hóa tốc độ
    X_tiled = np.tile(X_input, (n_samples, 1, 1))
    mc_preds_scaled_batch = model(X_tiled, training=True)
    mc_preds_scaled = np.squeeze(mc_preds_scaled_batch.numpy())

    # 3. Nghịch đảo chuẩn hóa về mét
    if target_scaler is not None:
        mean_wl = float(np.squeeze(target_scaler.inverse_transform([[pred_scaled]])))
        mc_preds_meters = target_scaler.inverse_transform(mc_preds_scaled.reshape(-1, 1)).flatten()
        std_pred = float(np.std(mc_preds_meters))
    else:
        mean_wl = pred_scaled
        std_pred = float(np.std(mc_preds_scaled))

    # Khoảng tin cậy 95%: mean ± 1.96 * std
    ci95_lower = mean_wl - 1.96 * std_pred
    ci95_upper = mean_wl + 1.96 * std_pred

    return mean_wl, ci95_lower, ci95_upper


# ---------------------------------------------------------------------------
# Hàm xác định mức cảnh báo
# ---------------------------------------------------------------------------

def determine_alert_level(max_water_level: float) -> tuple[str, str]:
    """
    Xác định mức cảnh báo và thông điệp dựa trên mực nước dự báo cao nhất.

    Ngưỡng hồ Núi Cốc:
        > 47.40m : NGUY HIỂM  — mở toàn bộ cửa xả, sơ tán dân hạ lưu
        > 46.80m : CẢNH BÁO   — theo dõi liên tục, sẵn sàng vận hành xả
        ≤ 46.80m : BÌNH THƯỜNG — vận hành bình thường

    Args:
        max_water_level: Mực nước dự báo cao nhất trong tất cả chân trời (m)

    Returns:
        alert_level (str):   Mức cảnh báo
        alert_message (str): Thông điệp hành động
    """
    if max_water_level > NGUY_HIEM:
        level = "NGUY HIỂM"
        message = (
            f"⛔ MỰC NƯỚC DỰ BÁO {max_water_level:.2f}m — VƯỢT NGƯỠNG NGUY HIỂM "
            f"({NGUY_HIEM}m). YÊU CẦU: Mở toàn bộ cửa xả lũ, sơ tán dân cư hạ lưu "
            f"ngay lập tức, báo cáo Ban chỉ huy PCTT tỉnh Thái Nguyên."
        )
    elif max_water_level > CANH_BAO:
        level = "CẢNH BÁO"
        message = (
            f"⚠️  MỰC NƯỚC DỰ BÁO {max_water_level:.2f}m — VƯỢT NGƯỠNG CẢNH BÁO "
            f"({CANH_BAO}m). YÊU CẦU: Theo dõi liên tục, sẵn sàng vận hành cửa xả, "
            f"thông báo chính quyền và người dân hạ lưu."
        )
    else:
        level = "BÌNH THƯỜNG"
        message = (
            f"✅ Mực nước dự báo {max_water_level:.2f}m — Trong ngưỡng bình thường "
            f"(≤ {CANH_BAO}m). Tiếp tục theo dõi định kỳ."
        )
    return level, message


# ---------------------------------------------------------------------------
# Khởi tạo FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hệ thống dự báo mực nước hồ Núi Cốc",
    description=(
        "API dự báo mực nước hồ Núi Cốc (Thái Nguyên) sử dụng mô hình "
        "Bi-LSTM với ước lượng bất định Monte Carlo Dropout. "
        "Hỗ trợ dự báo các chân trời 1d, 3d, 7d, 14d, 30d (tần suất Ngày)."
    ),
    version="4.0",
    contact={
        "name": "Đồ án tốt nghiệp",
        "url": "https://github.com/",
    },
    license_info={"name": "MIT"},
)


# ---------------------------------------------------------------------------
# Route trang chủ HTML Dashboard
# ---------------------------------------------------------------------------
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hệ thống dự báo mực nước hồ Núi Cốc</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            body {
                margin: 0;
                font-family: 'Outfit', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #f8fafc;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                overflow-x: hidden;
            }
            .container {
                max-width: 800px;
                padding: 40px;
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                text-align: center;
                margin: 20px;
                animation: fadeIn 0.8s ease-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            h1 {
                font-size: 2.8rem;
                font-weight: 800;
                margin-bottom: 10px;
                background: linear-gradient(to right, #38bdf8, #3b82f6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            p.tagline {
                font-size: 1.1rem;
                color: #94a3b8;
                margin-bottom: 30px;
                line-height: 1.6;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            .card {
                background: rgba(15, 23, 42, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 16px;
                transition: all 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                border-color: rgba(56, 189, 248, 0.4);
                box-shadow: 0 10px 20px rgba(56, 189, 248, 0.1);
            }
            .card-title {
                font-weight: 600;
                font-size: 1rem;
                color: #38bdf8;
                margin-bottom: 8px;
            }
            .card-desc {
                font-size: 0.85rem;
                color: #cbd5e1;
            }
            .btn {
                display: inline-block;
                padding: 14px 32px;
                background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
                color: white;
                text-decoration: none;
                font-weight: 600;
                border-radius: 50px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
            }
            .btn:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6);
                background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            }
            .footer {
                margin-top: 40px;
                font-size: 0.8rem;
                color: #64748b;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Hệ thống dự báo mực nước hồ Núi Cốc</h1>
            <p class="tagline">API phục vụ dự báo mực nước hồ chứa sử dụng mô hình học máy nâng cao <strong>Bi-LSTM</strong> kết hợp ước lượng bất định Monte Carlo Dropout và cân bằng nước.</p>
            
            <div class="grid">
                <div class="card">
                    <div class="card-title">Cửa sổ đầu vào</div>
                    <div class="card-desc">60 ngày liên tục gần nhất của 26 đặc trưng khí tượng thủy văn.</div>
                </div>
                <div class="card">
                    <div class="card-title">Chân trời dự báo</div>
                    <div class="card-desc">Hỗ trợ các mốc thời gian: 1 ngày, 3 ngày, 7 ngày, 14 ngày và 30 ngày.</div>
                </div>
                <div class="card">
                    <div class="card-title">Độ không chắc chắn</div>
                    <div class="card-desc">Khoảng tin cậy 95% mô phỏng qua 50 mẫu Monte Carlo Dropout.</div>
                </div>
            </div>

            <a href="/docs" class="btn">Mở Tài Liệu API (Swagger UI)</a>
            
            <div class="footer">
                Đồ án Tốt nghiệp &copy; 2026 | Khoa Thủy văn - Thủy lợi
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


# ---------------------------------------------------------------------------
# Sự kiện khởi động — tải model và scaler
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """
    Hook khởi động FastAPI: tải toàn bộ model Bi-LSTM, feature scaler và target scalers.
    Kết quả được lưu vào app_state để tái sử dụng trên mọi request.
    """
    logger.info("=== Khởi động hệ thống dự báo mực nước hồ Núi Cốc v4.0 ===")
    models, scaler, target_scalers = load_models_and_scaler()
    app_state["models"] = models
    app_state["scaler"] = scaler
    app_state["target_scalers"] = target_scalers
    logger.info("=== Server sẵn sàng phục vụ ===")


# ---------------------------------------------------------------------------
# Endpoint: POST /predict
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Dự báo mực nước hồ Núi Cốc",
    description=(
        "Nhận ma trận đặc trưng 60×26 (60 ngày gần nhất, 26 đặc trưng ngày), "
        "trả về dự báo mực nước cho các chân trời 1d, 3d, 7d, 14d, 30d "
        "kèm khoảng tin cậy 95% và mức cảnh báo lũ."
    ),
    tags=["Dự báo"],
)
async def predict(request: ForecastRequest) -> ForecastResponse:
    """
    Endpoint dự báo mực nước hồ Núi Cốc.

    Quy trình xử lý:
        1. Kiểm tra shape đầu vào (60, 26)
        2. Chuẩn hoá đặc trưng bằng feature scaler
        3. Với mỗi horizon d có model: gọi Monte Carlo Dropout prediction
        4. Tính mức cảnh báo dựa trên mực nước dự báo cao nhất
        5. Trả về ForecastResponse đầy đủ
    """
    models: dict = app_state["models"]
    scaler = app_state["scaler"]

    # ---- Kiểm tra scaler ----
    if scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Feature scaler chưa được tải. Kiểm tra file models/feature_scaler_daily.pkl.",
        )

    # ---- Kiểm tra có ít nhất 1 model ----
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Chưa có model nào được tải. Kiểm tra thư mục models/.",
        )

    # ---- Validate shape: phải là (60, 26) ----
    features_raw = request.features
    n_rows = len(features_raw)
    if n_rows != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Số bước thời gian không hợp lệ: nhận được {n_rows}, "
                f"yêu cầu {WINDOW_SIZE} (60 ngày)."
            ),
        )
    for i, row in enumerate(features_raw):
        if len(row) != FEATURE_COUNT:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Hàng {i} có {len(row)} đặc trưng, "
                    f"yêu cầu {FEATURE_COUNT} đặc trưng."
                ),
            )

    # ---- Chuyển thành numpy array và chuẩn hoá ----
    # X_raw: shape (60, 26)
    X_raw = np.array(features_raw, dtype=np.float32)

    try:
        # Scaler được fit trên shape (n_samples, 26) — reshape 2D trước khi transform
        X_scaled = scaler.transform(X_raw)  # shape (60, 26)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi chuẩn hoá đặc trưng: {exc}",
        ) from exc

    # Thêm chiều batch: shape (1, 60, 26) để đưa vào model
    X_input = X_scaled[np.newaxis, ...]  # shape (1, 60, 26)

    # ---- Dự báo từng chân trời ----
    forecasts: list[HorizonForecast] = []
    models_used: list[int] = []

    for d in FORECAST_DAYS:
        if d not in models:
            # Không có model cho chân trời này — bỏ qua
            logger.debug("Không có model cho t%dd — bỏ qua.", d)
            continue

        model = models[d]
        target_scaler = app_state["target_scalers"].get(d)
        try:
            mean_wl, ci_lower, ci_upper = predict_with_mc_dropout(
                model=model,
                X_input=X_input,
                target_scaler=target_scaler,
                n_samples=50,
            )
        except Exception as exc:
            logger.error("Lỗi khi dự báo t%dd: %s", d, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi khi dự báo chân trời t{d}d: {exc}",
            ) from exc

        forecasts.append(
            HorizonForecast(
                horizon_d=d,
                water_level_m=round(mean_wl, 4),
                ci95_lower=round(ci_lower, 4),
                ci95_upper=round(ci_upper, 4),
            )
        )
        models_used.append(d)
        logger.info(
            "t%02dd → mực nước = %.3fm [CI95: %.3f – %.3f]",
            d, mean_wl, ci_lower, ci_upper,
        )

    # ---- Tính mức cảnh báo dựa trên mực nước dự báo cao nhất ----
    if forecasts:
        max_predicted_wl = max(f.water_level_m for f in forecasts)
    else:
        # Không có dự báo nào — không thể xác định mức cảnh báo
        raise HTTPException(
            status_code=503,
            detail="Không thể thực hiện dự báo — không có model phù hợp.",
        )

    alert_level, alert_message = determine_alert_level(max_predicted_wl)

    # ---- Thời điểm xử lý request (UTC) ----
    request_time = datetime.now(tz=timezone.utc).isoformat()

    return ForecastResponse(
        request_time=request_time,
        forecasts=forecasts,
        alert_level=alert_level,
        alert_message=alert_message,
        models_used=sorted(models_used),
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái server",
    description="Trả về trạng thái server, các model đã tải và thông tin cấu hình.",
    tags=["Hệ thống"],
)
async def health() -> HealthResponse:
    """
    Health-check endpoint.

    Trả về:
        - status: 'ok' nếu đủ model và scaler, 'degraded' nếu thiếu
        - models_loaded: danh sách chân trời (giờ) đã có model sẵn sàng
        - feature_count: số đặc trưng đầu vào (18)
        - window_size: kích thước cửa sổ thời gian (48)
        - scaler_loaded: True nếu scaler đã tải thành công
    """
    models: dict = app_state["models"]
    scaler = app_state["scaler"]

    models_loaded = sorted(models.keys())
    scaler_loaded = scaler is not None

    # Đánh giá trạng thái tổng thể
    if scaler_loaded and len(models_loaded) == len(FORECAST_DAYS):
        status = "ok"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        feature_count=FEATURE_COUNT,
        window_size=WINDOW_SIZE,
        scaler_loaded=scaler_loaded,
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /features
# ---------------------------------------------------------------------------

@app.get(
    "/features",
    summary="Danh sách đặc trưng đầu vào",
    description=(
        "Trả về danh sách tên và thứ tự các đặc trưng đầu vào theo đúng "
        "thứ tự cột mà model yêu cầu."
    ),
    tags=["Thông tin"],
)
async def get_features() -> dict:
    """
    Trả về danh sách và thứ tự các đặc trưng đầu vào.

    Thứ tự cột trong ma trận features[] của ForecastRequest phải khớp
    chính xác với danh sách này.
    """
    return {
        "feature_count": FEATURE_COUNT,
        "window_size": WINDOW_SIZE,
        "features": [
            {"index": idx, "name": col}
            for idx, col in enumerate(FEATURE_COLS)
        ],
    }


# ---------------------------------------------------------------------------
# Endpoint: GET /thresholds
# ---------------------------------------------------------------------------

@app.get(
    "/thresholds",
    summary="Ngưỡng cảnh báo lũ hồ Núi Cốc",
    description="Trả về các ngưỡng mực nước cảnh báo lũ của hồ Núi Cốc, Thái Nguyên.",
    tags=["Thông tin"],
)
async def get_thresholds() -> dict:
    """
    Trả về thông tin ngưỡng cảnh báo mực nước hồ Núi Cốc.

    Nguồn:
        Quy trình vận hành hồ chứa nước Núi Cốc — Bộ Nông nghiệp và PTNT.
    """
    return {
        "ho_nui_coc": {
            "ten_ho": "Hồ Núi Cốc",
            "tinh": "Thái Nguyên",
            "don_vi": "m (mét so với mực nước biển)",
            "nguong": {
                "muc_nuoc_dang_binh_thuong_MNDBT": {
                    "gia_tri_m": MNDBT,
                    "mo_ta": "Mực nước dâng bình thường — vận hành tích nước thông thường",
                },
                "nguong_canh_bao": {
                    "gia_tri_m": CANH_BAO,
                    "mo_ta": (
                        "Ngưỡng cảnh báo — theo dõi liên tục, "
                        "sẵn sàng vận hành cửa xả lũ"
                    ),
                },
                "nguong_nguy_hiem": {
                    "gia_tri_m": NGUY_HIEM,
                    "mo_ta": (
                        "Ngưỡng nguy hiểm — mở toàn bộ cửa xả, "
                        "sơ tán dân cư vùng hạ lưu, báo cáo khẩn"
                    ),
                },
            },
            "cap_canh_bao": [
                {
                    "cap": "BÌNH THƯỜNG",
                    "dieu_kien": f"Mực nước dự báo ≤ {CANH_BAO}m",
                    "hanh_dong": "Vận hành bình thường, theo dõi định kỳ",
                },
                {
                    "cap": "CẢNH BÁO",
                    "dieu_kien": f"{CANH_BAO}m < Mực nước dự báo ≤ {NGUY_HIEM}m",
                    "hanh_dong": (
                        "Theo dõi liên tục, sẵn sàng vận hành cửa xả, "
                        "thông báo chính quyền và người dân hạ lưu"
                    ),
                },
                {
                    "cap": "NGUY HIỂM",
                    "dieu_kien": f"Mực nước dự báo > {NGUY_HIEM}m",
                    "hanh_dong": (
                        "Mở toàn bộ cửa xả lũ, sơ tán dân cư vùng hạ lưu, "
                        "báo cáo khẩn Ban chỉ huy PCTT tỉnh Thái Nguyên"
                    ),
                },
            ],
        }
    }


# ---------------------------------------------------------------------------
# Điểm vào — chạy trực tiếp bằng Python
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Chạy server với host 0.0.0.0 (lắng nghe mọi interface) trên port 8000
    # Tương đương lệnh: uvicorn 08_api_serve:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

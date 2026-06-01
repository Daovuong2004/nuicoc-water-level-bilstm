"""
================================================================================
08_api_serve.py — FastAPI Inference Server
Hệ thống dự báo mực nước hồ Núi Cốc (Thái Nguyên)
Mô hình: Bi-LSTM + Attention với Monte Carlo Dropout
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
from keras.models import load_model  # type: ignore

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
# 18 đặc trưng theo đúng thứ tự đã dùng khi huấn luyện model
FEATURE_COLS = [
    "rain_1d",          # Lượng mưa ngày (mm)
    "rain_3d",          # Mưa tích lũy 3 ngày (mm)
    "rain_7d",          # Mưa tích lũy 7 ngày (mm)
    "rain_14d",         # Mưa tích lũy 14 ngày (mm)
    "temperature",      # Nhiệt độ trung bình ngày (°C)
    "humidity",         # Độ ẩm trung bình ngày (%)
    "water_level_lag1", # Mực nước trễ 1 ngày (m)
    "water_level_lag3", # Mực nước trễ 3 ngày (m)
    "water_level_lag7", # Mực nước trễ 7 ngày (m)
    "water_level_lag14",# Mực nước trễ 14 ngày (m)
    "water_level_lag30",# Mực nước trễ 30 ngày (m)
    "water_level_roll7",# Trung bình trượt mực nước 7 ngày (m)
    "water_level_roll30",# Trung bình trượt mực nước 30 ngày (m)
    "water_level_std7", # Độ lệch chuẩn mực nước 7 ngày (m)
    "month_sin",        # Mã hoá tuần hoàn tháng (sin)
    "month_cos",        # Mã hoá tuần hoàn tháng (cos)
    "season_wet",       # Mùa mưa (0/1)
    "season_dry",       # Mùa khô (0/1)
    "dH_dt_daily",      # Tốc độ biến đổi mực nước ngày (m/ngày)
    "Q_out_daily",      # Lưu lượng xả ngày ước lượng (m³/s)
    "Q_out_roll7",      # Trung bình trượt lưu lượng xả 7 ngày (m³/s)
]

# Số đặc trưng đầu vào
FEATURE_COUNT = len(FEATURE_COLS)  # 21

# Kích thước cửa sổ thời gian (ngày)
WINDOW_SIZE = 30

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
# Hàm tải model và scaler khi khởi động
# ---------------------------------------------------------------------------
def load_models_and_scaler() -> tuple[dict, object]:
    """
    Tải toàn bộ model Bi-LSTM+Attention và feature scaler từ thư mục models/.

    Quy trình:
        1. Load scaler (MinMaxScaler) từ models/feature_scaler_daily.pkl
        2. Với mỗi chân trời d trong FORECAST_DAYS, load
           models/bilstm_t{d}d.keras (nếu tồn tại)
        3. Model hoặc scaler bị thiếu sẽ bị bỏ qua và ghi log warning

    Returns:
        models_dict (dict): {horizon_d: keras_model}
        scaler: fitted MinMaxScaler hoặc None nếu không tìm thấy file
    """
    models_dict = {}

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

    # ---- Tải từng model theo chân trời dự báo ----
    for d in FORECAST_DAYS:
        model_path = os.path.join(MODEL_DIR, f"bilstm_t{d}d.keras")
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                models_dict[d] = model
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

    logger.info(
        "Khởi động hoàn tất — %d model sẵn sàng: %s",
        len(models_dict),
        sorted(models_dict.keys()),
    )
    return models_dict, scaler


# ---------------------------------------------------------------------------
# Biến toàn cục — được điền khi app khởi động
# ---------------------------------------------------------------------------
# Sử dụng dict để chứa state, tránh vấn đề với global trong FastAPI
app_state: dict = {
    "models": {},    # {horizon_h (int): keras model}
    "scaler": None,  # fitted StandardScaler
}


# ---------------------------------------------------------------------------
# Pydantic models — Request / Response
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    """
    Payload đầu vào cho endpoint POST /predict.

    Attributes:
        features:   Ma trận đặc trưng shape (30, 21) — dữ liệu 30 ngày gần nhất.
                    Thứ tự cột phải khớp với FEATURE_COLS.
        timestamp:  Thời điểm quan sát cuối cùng (ISO 8601), tùy chọn.
                    Ví dụ: "2026-05-20T14:00:00+07:00"
    """
    features: list[list[float]] = Field(
        ...,
        description=(
            "Ma trận đặc trưng shape (30, 21). "
            "Hàng = bước thời gian (ngày), Cột = đặc trưng theo thứ tự FEATURE_COLS."
        ),
        example=[[0.0] * 21] * 30,
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
# Hàm dự báo với Monte Carlo Dropout
# ---------------------------------------------------------------------------

def predict_with_mc_dropout(
    model,
    X_input: np.ndarray,
    n_samples: int = 50,
) -> tuple[float, float, float]:
    """
    Dự báo mực nước với ước lượng bất định bằng Monte Carlo Dropout.

    Nguyên lý Monte Carlo Dropout:
        Trong quá trình inference, ta giữ các lớp Dropout ở chế độ TRAINING
        (training=True). Điều này có nghĩa mỗi lần forward pass sẽ cho ra
        kết quả khác nhau do các neuron bị dropout ngẫu nhiên. Chạy N lần
        và lấy thống kê (mean, std) để xấp xỉ phân phối posterior của dự báo,
        từ đó thu được khoảng tin cậy mà không cần ensemble nhiều model riêng biệt.

    Args:
        model:      Keras model đã được load (Bi-LSTM + Attention + Dropout)
        X_input:    Mảng numpy shape (1, WINDOW_SIZE, FEATURE_COUNT)
        n_samples:  Số lần lấy mẫu Monte Carlo (mặc định 50)

    Returns:
        mean (float):       Mực nước dự báo trung bình (m)
        ci95_lower (float): Cận dưới khoảng tin cậy 95% (m)
        ci95_upper (float): Cận trên khoảng tin cậy 95% (m)
    """
    # Thu thập n_samples dự báo — mỗi lần chạy với Dropout ngẫu nhiên
    predictions = []
    for _ in range(n_samples):
        # training=True kích hoạt Dropout ngay cả trong inference
        pred = model(X_input, training=True)
        # pred có shape (1, 1) hoặc (1,) — lấy giá trị vô hướng
        predictions.append(float(np.squeeze(pred.numpy())))

    predictions = np.array(predictions)  # shape (n_samples,)

    # Tính thống kê
    mean_pred = float(np.mean(predictions))
    std_pred = float(np.std(predictions))

    # Khoảng tin cậy 95%: mean ± 1.96 * std (xấp xỉ phân phối chuẩn)
    ci95_lower = mean_pred - 1.96 * std_pred
    ci95_upper = mean_pred + 1.96 * std_pred

    return mean_pred, ci95_lower, ci95_upper


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
        "Bi-LSTM + Attention với ước lượng bất định Monte Carlo Dropout. "
        "Hỗ trợ dự báo các chân trời 1d, 3d, 7d, 14d, 30d (tần suất Ngày)."
    ),
    version="3.0",
    contact={
        "name": "Đồ án tốt nghiệp",
        "url": "https://github.com/",
    },
    license_info={"name": "MIT"},
)


# ---------------------------------------------------------------------------
# Sự kiện khởi động — tải model và scaler
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """
    Hook khởi động FastAPI: tải toàn bộ model Bi-LSTM và feature scaler.
    Kết quả được lưu vào app_state để tái sử dụng trên mọi request.
    """
    logger.info("=== Khởi động hệ thống dự báo mực nước hồ Núi Cốc v2.0 ===")
    models, scaler = load_models_and_scaler()
    app_state["models"] = models
    app_state["scaler"] = scaler
    logger.info("=== Server sẵn sàng phục vụ ===")


# ---------------------------------------------------------------------------
# Endpoint: POST /predict
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Dự báo mực nước hồ Núi Cốc",
    description=(
        "Nhận ma trận đặc trưng 30×21 (30 ngày gần nhất, 21 đặc trưng ngày), "
        "trả về dự báo mực nước cho các chân trời 1d, 3d, 7d, 14d, 30d "
        "kèm khoảng tin cậy 95% và mức cảnh báo lũ."
    ),
    tags=["Dự báo"],
)
async def predict(request: ForecastRequest) -> ForecastResponse:
    """
    Endpoint dự báo mực nước hồ Núi Cốc.

    Quy trình xử lý:
        1. Kiểm tra shape đầu vào (30, 21)
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

    # ---- Validate shape: phải là (30, 21) ----
    features_raw = request.features
    n_rows = len(features_raw)
    if n_rows != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Số bước thời gian không hợp lệ: nhận được {n_rows}, "
                f"yêu cầu {WINDOW_SIZE} (30 ngày)."
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
    # X_raw: shape (30, 21)
    X_raw = np.array(features_raw, dtype=np.float32)

    try:
        # Scaler được fit trên shape (n_samples, 21) — reshape 2D trước khi transform
        X_scaled = scaler.transform(X_raw)  # shape (30, 21)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi chuẩn hoá đặc trưng: {exc}",
        ) from exc

    # Thêm chiều batch: shape (1, 30, 21) để đưa vào model
    X_input = X_scaled[np.newaxis, ...]  # shape (1, 30, 21)

    # ---- Dự báo từng chân trời ----
    forecasts: list[HorizonForecast] = []
    models_used: list[int] = []

    for d in FORECAST_DAYS:
        if d not in models:
            # Không có model cho chân trời này — bỏ qua
            logger.debug("Không có model cho t%dd — bỏ qua.", d)
            continue

        model = models[d]
        try:
            mean_wl, ci_lower, ci_upper = predict_with_mc_dropout(
                model=model,
                X_input=X_input,
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

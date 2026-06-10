"""
================================================================================
08_api_serve.py — FastAPI Inference Server
Hệ thống dự báo mực nước hồ Núi Cốc (Thái Nguyên)
Mô hình: Bi-LSTM với Monte Carlo Dropout
================================================================================

Mô tả:
    Server cung cấp API REST để dự báo mực nước hồ Núi Cốc theo nhiều chân trời
    thời gian (1d, 3d, 7d, 14d, 30d) dựa trên dữ liệu 21 ngày gần nhất.
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
from keras.layers import Input, LSTM, Dense, Dropout
from keras.regularizers import l2

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURE_COLS,
    FEATURE_COUNT,
    WINDOW_SIZE,
    FORECAST_DAYS,
    L2_REG,
    DROPOUT_RATE,
    LSTM_UNITS,
    USE_BIDIRECTIONAL,
    RECURRENT_DROPOUT,
    MC_SAMPLES,
    PREDICT_DELTA_H,
    ENSEMBLE_PERSISTENCE_WEIGHT,
    TRAIN_CONFIG_PATH,
    MNDBT,
    CANH_BAO,
    NGUY_HIEM,
    MODEL_DIR,
    SCALER_PATH,
)

# ---------------------------------------------------------------------------
# Cấu hình logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api_serve")

# (FEATURE_COLS, WINDOW_SIZE, ... imported from config.py v5)


# ---------------------------------------------------------------------------
# Kiến trúc mô hình Bi-LSTM để rebuild cho MC Dropout
# ---------------------------------------------------------------------------
def build_bilstm(input_shape: tuple, mc_dropout: bool = False) -> Model:
    """
    Kiến trúc Bi-LSTM v5.1 — đồng bộ với 06_bilstm_model.py.

    Bidirectional(LSTM(...), merge_mode='concat') → output dim = 2 * LSTM_UNITS[0]
    Khi mc_dropout=True: Dropout layer bật training=True để dùng cho MC inference.
    """
    from keras.layers import Bidirectional  # đảm bảo import
    inputs = Input(shape=input_shape, name="input_sequence")
    if USE_BIDIRECTIONAL:
        x = Bidirectional(
            LSTM(
                LSTM_UNITS[0],
                return_sequences=False,
                recurrent_dropout=RECURRENT_DROPOUT,
                name="lstm_fwd_bwd",
            ),
            merge_mode="concat",
            name="bilstm_1",
        )(inputs)
    else:
        x = LSTM(
            LSTM_UNITS[0],
            return_sequences=False,
            recurrent_dropout=RECURRENT_DROPOUT,
            name="lstm_1",
        )(inputs)
    if mc_dropout:
        x = Dropout(DROPOUT_RATE, name="dropout_1")(x, training=True)
    else:
        x = Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(L2_REG), name="dense_1")(x)
    outputs = Dense(1, activation="linear", name="output")(x)
    model_name = "BiLSTM_v51" if USE_BIDIRECTIONAL else "LSTM_v51"
    return Model(inputs=inputs, outputs=outputs, name=model_name)


# ---------------------------------------------------------------------------
# Hàm tải model và scaler khi khởi động
# ---------------------------------------------------------------------------
def load_models_and_scaler() -> tuple[dict, object, dict]:
    """
    Tải toàn bộ model Bi-LSTM, feature scaler và các target scalers từ thư mục models/.

    Quy trình:
        1. Load StandardScaler (feature) từ models/feature_scaler_daily.pkl
        2. Với mỗi chân trời d trong FORECAST_DAYS [1,3,7,14,30], load
           models/bilstm_t{d}d.keras (nếu tồn tại)
        3. Load target StandardScaler tương ứng từ models/target_scaler_t{d}d.pkl
        4. Model hoặc scaler bị thiếu sẽ bị bỏ qua và ghi log warning

    Returns:
        models_dict (dict): {horizon_d: keras_model}
        scaler: fitted StandardScaler hoặc None nếu không tìm thấy file
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
    Payload đầu vào cho endpoint POST /predict (v5).

    features:      (WINDOW_SIZE, n_features) đã chuẩn hóa — thứ tự FEATURE_COLS.
    base_level_m:  Mực nước H(t) tại ngày cuối (m, chưa scale) — ghép ΔH → H(t+d).
    """
    features: list[list[float]] = Field(
        ...,
        description=f"Ma trận ({WINDOW_SIZE}, {FEATURE_COUNT}) — đã StandardScaler.",
        json_schema_extra={"example": [[0.0] * FEATURE_COUNT] * WINDOW_SIZE},
    )
    base_level_m: float = Field(
        ...,
        description="Mực nước thực tại thời điểm quan sát cuối H(t) (m)",
        json_schema_extra={"example": 42.5},
    )
    timestamp: str | None = Field(
        default=None,
        description="Thời điểm quan sát cuối cùng (ISO 8601), tùy chọn.",
        json_schema_extra={"example": "2026-05-20T14:00:00+07:00"},
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
    base_level_m: float,
    target_scaler: object = None,
    n_samples: int = MC_SAMPLES,
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

    if target_scaler is not None:
        delta_mean = float(np.squeeze(target_scaler.inverse_transform([[pred_scaled]])))
        mc_delta = target_scaler.inverse_transform(
            mc_preds_scaled.reshape(-1, 1)
        ).flatten()
    else:
        delta_mean = pred_scaled
        mc_delta = mc_preds_scaled

    if PREDICT_DELTA_H:
        mean_wl = base_level_m + delta_mean
        mc_levels = base_level_m + mc_delta
    else:
        mean_wl = delta_mean
        mc_levels = mc_delta

    w = ENSEMBLE_PERSISTENCE_WEIGHT
    mean_wl = w * base_level_m + (1.0 - w) * mean_wl
    mc_levels = w * base_level_m + (1.0 - w) * mc_levels

    std_pred = float(np.std(mc_levels))
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
# Lifespan: khởi động và dọn dẹp model (FastAPI v0.110+ hiện đại)
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance):  # noqa: ARG001
    """
    Context manager lifespan — thay thế @app.on_event('startup') đã deprecated.
    Khởi động: tải toàn bộ model Bi-LSTM, feature scaler và target scalers.
    """
    logger.info("=== Khởi động hệ thống dự báo mực nước hồ Núi Cốc v5.1 ===")
    models, scaler, target_scalers = load_models_and_scaler()
    app_state["models"] = models
    app_state["scaler"] = scaler
    app_state["target_scalers"] = target_scalers
    logger.info("=== Server sẵn sàng phục vụ ===")
    yield   # ← server đang chạy
    logger.info("=== Server đã dừng ===")


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
    version="5.1",
    contact={
        "name": "Đồ án tốt nghiệp",
        "url": "https://github.com/",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,   # ← kết nối lifespan hiện đại
)


# ---------------------------------------------------------------------------
# Endpoint mới: GET /forecast?date=YYYY-MM-DD  — tự đọc dataset, không cần nhập liệu
# ---------------------------------------------------------------------------
from fastapi.responses import HTMLResponse
from typing import Optional

@app.get(
    "/forecast",
    summary="Dự báo tự động từ dataset (không cần nhập liệu)",
    description=(
        "Server tự đọc 21 ngày dữ liệu từ dataset_full.csv kết thúc tại ngày chỉ định, "
        "rồi trả về dự báo mực nước cho 5 chân trời. "
        "Tham số date là tùy chọn, mặc định lấy ngày mới nhất trong dataset."
    ),
    tags=["Dự báo"],
)
async def forecast_from_date(date: Optional[str] = None) -> dict:
    """
    Endpoint đơn giản: chỉ cần truyền ngày (hoặc không truyền gì),
    server tự lo phần còn lại.

    Parameters
    ----------
    date : str, optional
        Ngày kết thúc cửa sổ 21 ngày, định dạng YYYY-MM-DD.
        Nếu bỏ trống → lấy ngày mới nhất trong dataset.

    Returns
    -------
    dict với các trường:
        issue_date   : ngày phát hành dự báo
        base_level_m : mực nước H(t) tại ngày cuối window
        forecasts    : danh sách dự báo 5 chân trời
        alert_level  : mức cảnh báo
        alert_message: thông điệp cảnh báo
    """
    import os
    # Tìm file dataset
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "final", "dataset_full.csv",
    )
    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=503,
            detail="Không tìm thấy data/final/dataset_full.csv. Hãy chạy 05_integrate.py trước.",
        )

    import pandas as pd
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)

    # Xác định ngày kết thúc window
    if date:
        try:
            end_date = pd.Timestamp(date)
        except Exception:
            raise HTTPException(status_code=422, detail=f"Ngày không hợp lệ: {date}. Định dạng: YYYY-MM-DD")
        avail = df.index[df.index <= end_date]
        if len(avail) == 0:
            raise HTTPException(status_code=422, detail=f"Không có dữ liệu trước ngày {date}.")
        end_date = avail[-1]
    else:
        end_date = df.index[-1]

    end_pos = df.index.get_loc(end_date)
    if end_pos < WINDOW_SIZE - 1:
        raise HTTPException(
            status_code=422,
            detail=f"Không đủ {WINDOW_SIZE} ngày trước {end_date.date()}.",
        )

    # Kiểm tra features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=503, detail=f"Dataset thiếu cột: {missing}")

    window        = df.iloc[end_pos - WINDOW_SIZE + 1 : end_pos + 1]
    X_raw         = window[FEATURE_COLS].values.astype("float32")
    base_level_m  = float(df.loc[end_date, "water_level_m"])
    issue_date    = str(end_date.date())

    # Kiểm tra scaler và models
    scaler  = app_state["scaler"]
    models  = app_state["models"]
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler chưa được tải.")
    if not models:
        raise HTTPException(status_code=503, detail="Chưa có model nào được tải.")

    X_scaled = scaler.transform(X_raw)
    X_input  = X_scaled[np.newaxis, ...]          # (1, 21, 16)

    forecasts   = []
    models_used = []
    for d in FORECAST_DAYS:
        if d not in models:
            continue
        target_scaler = app_state["target_scalers"].get(d)
        try:
            mean_wl, ci_lo, ci_hi = predict_with_mc_dropout(
                model=models[d],
                X_input=X_input,
                base_level_m=base_level_m,
                target_scaler=target_scaler,
            )
        except Exception as exc:
            logger.error("Lỗi forecast t%dd: %s", d, exc)
            continue
        forecasts.append({
            "horizon_d":     d,
            "water_level_m": round(mean_wl, 4),
            "ci95_lower":    round(ci_lo,   4),
            "ci95_upper":    round(ci_hi,   4),
            "delta_m":       round(mean_wl - base_level_m, 4),
        })
        models_used.append(d)

    if not forecasts:
        raise HTTPException(status_code=503, detail="Không thể thực hiện dự báo.")

    max_wl = max(f["water_level_m"] for f in forecasts)
    alert_level, alert_message = determine_alert_level(max_wl)

    return {
        "issue_date":    issue_date,
        "base_level_m":  round(base_level_m, 4),
        "forecasts":     forecasts,
        "models_used":   sorted(models_used),
        "alert_level":   alert_level,
        "alert_message": alert_message,
    }

# ---------------------------------------------------------------------------
# Endpoint: POST /forecast-realtime — Dự báo ngày THỰC TẾ (tương lai)
# ---------------------------------------------------------------------------
import math as _math

class RealtimeRequest(BaseModel):
    """Không cần nhập gì — server tự đọc dữ liệu mới nhất từ dataset."""
    water_level_m: float | None = Field(
        default=None,
        description="Mực nước hồ (m). Bỏ trống = tự lấy từ dòng cuối dataset.",
        json_schema_extra={"example": 42.5},
    )
    rain_today_mm: float | None = Field(
        default=None,
        description="Lượng mưa (mm). Bỏ trống = tự lấy từ dòng cuối dataset.",
        json_schema_extra={"example": 5.0},
    )
    date: str | None = Field(
        default=None,
        description="Ngày dự báo (YYYY-MM-DD). Bỏ trống = ngày cuối cùng của dataset.",
        json_schema_extra={"example": "2025-12-28"},
    )


@app.post(
    "/forecast-realtime",
    summary="⚡ Dự báo TƯƠNG LAI — bấm 1 nút, không cần nhập liệu",
    description=(
        "Không cần nhập gì! Server tự đọc mực nước + lượng mưa từ dữ liệu mới nhất, "
        "xây dựng cửa sổ 21 ngày, và trả về dự báo t+1d, t+3d, t+7d, t+14d, t+30d "
        "kèm ngày dự kiến cụ thể và khoảng tin cậy 95%."
    ),
    tags=["Dự báo"],
)
async def forecast_realtime(req: RealtimeRequest) -> dict:
    # --- Tải dataset lịch sử (cần trước để auto-fill) ---
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "final", "dataset_full.csv",
    )
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=503, detail="Không tìm thấy dataset_full.csv.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # --- Xác định ngày (mặc định = ngày cuối dataset) ---
    if req.date:
        try:
            today = pd.Timestamp(req.date)
        except Exception:
            raise HTTPException(status_code=422, detail=f"Ngày không hợp lệ: {req.date}")
    else:
        today = pd.Timestamp.now().normalize()

    # Tìm hàng dữ liệu khả dụng gần nhất trước hoặc bằng ngày today
    avail_rows = df.loc[:today]
    if len(avail_rows) == 0:
        raise HTTPException(status_code=422, detail=f"Không tìm thấy dữ liệu lịch sử trước ngày {today.date()}.")
    active_row = avail_rows.iloc[-1]

    # --- Auto-fill từ dataset nếu không nhập ---
    active_wl = active_row["water_level_m"]
    if pd.isna(active_wl):
        active_wl = 39.50
    active_rain = active_row.get("rain_1d", 0.0)
    if pd.isna(active_rain):
        active_rain = 0.0

    H_today    = req.water_level_m if req.water_level_m is not None else float(active_wl)
    rain_today = max(0.0, req.rain_today_mm) if req.rain_today_mm is not None else float(active_rain)
    month      = today.month

    hist = avail_rows.tail(60).copy()

    # --- Climatology theo tháng ---
    clim_temp = float(df[df.index.month == month]["temperature"].mean()) if "temperature" in df.columns else 25.0
    clim_humi = float(df[df.index.month == month]["humidity"].mean()) if "humidity" in df.columns else 80.0
    if np.isnan(clim_temp):
        clim_temp = 25.0
    if np.isnan(clim_humi):
        clim_humi = 80.0

    # --- Water level lag features ---
    def wl_lag(days: int) -> float:
        target = today - pd.Timedelta(days=days)
        avail = hist.index[hist.index <= target]
        return float(hist.loc[avail[-1], "water_level_m"]) if len(avail) > 0 else H_today

    lag7 = wl_lag(7)
    lag14 = wl_lag(14)
    lag30 = wl_lag(30)

    # --- Rolling water level ---
    recent7_vals = hist["water_level_m"].values[-6:].tolist() + [H_today]
    roll7 = float(np.mean(recent7_vals))
    std7 = float(np.std(recent7_vals))

    # --- Rainfall tích lũy ---
    def rain_sum(days: int) -> float:
        start = today - pd.Timedelta(days=days - 1)
        window_rain = hist[hist.index >= start]["rain_1d"].sum() if "rain_1d" in hist.columns else 0.0
        return float(window_rain) + rain_today

    # --- Month encoding ---
    month_sin = _math.sin(2 * _math.pi * month / 12)
    month_cos = _math.cos(2 * _math.pi * month / 12)
    season_wet = 1.0 if month in [5, 6, 7, 8, 9, 10] else 0.0

    # --- Hàng đặc trưng hôm nay ---
    today_row = {
        "rain_1d": rain_today, "rain_3d": rain_sum(3),
        "rain_7d": rain_sum(7), "rain_14d": rain_sum(14),
        "rain_30d": rain_sum(30), "temperature": clim_temp,
        "humidity": clim_humi, "water_level_lag7": lag7,
        "water_level_lag14": lag14, "water_level_lag30": lag30,
        "water_level_roll7": roll7, "water_level_std7": std7,
        "month_sin": month_sin, "month_cos": month_cos,
        "season_wet": season_wet, "season_dry": 1.0 - season_wet,
        "water_level_m": H_today,
    }

    # --- Ghép cửa sổ 21 ngày: 20 ngày lịch sử + hôm nay ---
    hist20 = hist.tail(WINDOW_SIZE - 1)
    today_df = pd.DataFrame([today_row], index=[today])
    window_df = pd.concat([hist20, today_df]).tail(WINDOW_SIZE)

    missing = [c for c in FEATURE_COLS if c not in window_df.columns]
    if missing:
        raise HTTPException(status_code=503, detail=f"Thiếu features: {missing}")

    X_raw = window_df[FEATURE_COLS].values.astype("float32")
    scaler = app_state["scaler"]
    models = app_state["models"]
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler chưa được tải.")
    if not models:
        raise HTTPException(status_code=503, detail="Chưa có model nào được tải.")

    X_scaled = scaler.transform(X_raw)
    X_input = X_scaled[np.newaxis, ...]

    forecasts = []
    models_used = []
    for d in FORECAST_DAYS:
        if d not in models:
            continue
        target_scaler = app_state["target_scalers"].get(d)
        try:
            mean_wl, ci_lo, ci_hi = predict_with_mc_dropout(
                model=models[d], X_input=X_input,
                base_level_m=H_today, target_scaler=target_scaler,
            )
        except Exception as exc:
            logger.error("Lỗi realtime forecast t%dd: %s", d, exc)
            continue
        forecasts.append({
            "horizon_d": d,
            "forecast_date": str((today + pd.Timedelta(days=d)).date()),
            "water_level_m": round(mean_wl, 4),
            "ci95_lower": round(ci_lo, 4),
            "ci95_upper": round(ci_hi, 4),
            "delta_m": round(mean_wl - H_today, 4),
        })
        models_used.append(d)

    if not forecasts:
        raise HTTPException(status_code=503, detail="Không thể thực hiện dự báo.")

    max_wl = max(f["water_level_m"] for f in forecasts)
    alert_level, alert_message = determine_alert_level(max_wl)

    logger.info(
        "Realtime forecast: %s | H(t)=%.2fm | mưa=%.1fmm | %d chân trời",
        today.date(), H_today, rain_today, len(forecasts),
    )
    return {
        "issue_date": str(today.date()),
        "base_level_m": round(H_today, 4),
        "rain_today_mm": round(rain_today, 2),
        "forecasts": forecasts,
        "models_used": sorted(models_used),
        "alert_level": alert_level,
        "alert_message": alert_message,
        "note": (
            f"Dự báo từ H(t)={H_today:.2f}m ngày {today.date()}. "
            f"Nhiệt độ/độ ẩm ước tính theo khí hậu tháng {month}."
        ),
    }


# ---------------------------------------------------------------------------
# Route trang chủ HTML Dashboard — tương tác đầy đủ
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    html_content = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dự báo mực nước hồ Núi Cốc — Bi-LSTM</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Outfit',sans-serif;background:#0b1120;color:#e2e8f0;min-height:100vh;padding:24px 16px 48px}
    .header{text-align:center;padding:32px 0 20px}
    .header h1{font-size:clamp(1.5rem,4vw,2.5rem);font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2}
    .header p{margin-top:6px;color:#64748b;font-size:.9rem}
    .tabs{display:flex;justify-content:center;gap:4px;margin-bottom:24px}
    .tab-btn{padding:10px 28px;border:1px solid rgba(255,255,255,.1);border-radius:12px;background:rgba(30,41,59,.6);color:#94a3b8;font-size:.95rem;font-weight:600;font-family:inherit;cursor:pointer;transition:all .2s}
    .tab-btn.active{background:linear-gradient(135deg,#0ea5e9,#6366f1);color:#fff;border-color:transparent;box-shadow:0 4px 18px rgba(14,165,233,.35)}
    .tab-btn:hover:not(.active){border-color:rgba(56,189,248,.4);color:#e2e8f0}
    .tab-content{display:none}.tab-content.active{display:block}
    .panel{max-width:740px;margin:0 auto 24px;background:rgba(30,41,59,.75);border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:24px 28px;backdrop-filter:blur(14px);box-shadow:0 12px 40px rgba(0,0,0,.4)}
    .panel-title{font-size:.95rem;font-weight:600;color:#94a3b8;margin-bottom:14px;text-transform:uppercase;letter-spacing:.04em}
    .form-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
    .form-group{display:flex;flex-direction:column;gap:4px}
    .form-group label{font-size:.8rem;color:#64748b}
    .form-group input{padding:10px 14px;background:rgba(15,23,42,.8);border:1px solid rgba(255,255,255,.12);border-radius:10px;color:#f1f5f9;font-size:1rem;font-family:inherit;outline:none;transition:border-color .2s}
    .form-group input:focus{border-color:#38bdf8}
    .form-group .unit{font-size:.75rem;color:#475569;margin-top:2px}
    .control-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    input[type=date]{flex:1;min-width:160px;padding:10px 14px;background:rgba(15,23,42,.8);border:1px solid rgba(255,255,255,.12);border-radius:10px;color:#f1f5f9;font-size:1rem;font-family:inherit;outline:none;transition:border-color .2s}
    input[type=date]:focus{border-color:#38bdf8}
    .btn-predict{padding:10px 28px;background:linear-gradient(135deg,#0ea5e9,#6366f1);border:none;border-radius:10px;color:#fff;font-size:.95rem;font-weight:700;font-family:inherit;cursor:pointer;transition:all .25s;box-shadow:0 4px 18px rgba(14,165,233,.35);white-space:nowrap}
    .btn-predict:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(14,165,233,.5)}
    .btn-predict:disabled{opacity:.5;cursor:not-allowed;transform:none}
    .spinner{display:none;width:20px;height:20px;border:3px solid rgba(255,255,255,.2);border-top-color:#38bdf8;border-radius:50%;animation:spin .7s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .alert-banner{max-width:740px;margin:0 auto 16px;padding:14px 20px;border-radius:12px;font-weight:600;font-size:.95rem;display:none;animation:fadeSlide .4s ease-out}
    @keyframes fadeSlide{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:none}}
    .alert-ok{background:rgba(16,185,129,.15);border:1px solid #10b981;color:#6ee7b7}
    .alert-warn{background:rgba(245,158,11,.15);border:1px solid #f59e0b;color:#fcd34d}
    .alert-danger{background:rgba(239,68,68,.15);border:1px solid #ef4444;color:#fca5a5}
    .results{max-width:740px;margin:0 auto;display:none}
    .kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:12px;margin-bottom:18px}
    .kpi{background:rgba(30,41,59,.7);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:16px 12px;text-align:center;transition:all .25s}
    .kpi:hover{border-color:rgba(56,189,248,.35);transform:translateY(-3px)}
    .kpi-label{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
    .kpi-date{font-size:.72rem;color:#818cf8;margin-bottom:4px;font-weight:600}
    .kpi-value{font-size:1.5rem;font-weight:800;color:#f1f5f9}
    .kpi-unit{font-size:.7rem;color:#94a3b8;margin-top:1px}
    .kpi-delta{font-size:.78rem;margin-top:3px}
    .up{color:#f87171} .down{color:#34d399}
    .chart-card{background:rgba(30,41,59,.7);border:1px solid rgba(255,255,255,.07);border-radius:18px;padding:20px;margin-bottom:18px}
    .chart-card h3{font-size:.95rem;color:#94a3b8;margin-bottom:14px}
    .info-bar{display:flex;gap:20px;flex-wrap:wrap;background:rgba(15,23,42,.6);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:12px 16px;font-size:.82rem;color:#64748b;margin-bottom:16px}
    .info-bar span b{color:#e2e8f0}
    footer{text-align:center;margin-top:30px;color:#334155;font-size:.76rem}
    footer a{color:#38bdf8;text-decoration:none}
    .error-msg{color:#f87171;text-align:center;padding:10px;font-size:.88rem;display:none}
    .note-bar{max-width:740px;margin:0 auto 14px;padding:10px 16px;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25);border-radius:10px;font-size:.82rem;color:#a5b4fc;display:none}
  </style>
</head>
<body>
<div class="header">
  <h1>🌊 Dự báo mực nước hồ Núi Cốc</h1>
  <p>Bi-LSTM · Monte Carlo Dropout · 5 chân trời dự báo</p>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('realtime')">⚡ Dự báo tương lai</button>
  <button class="tab-btn" onclick="switchTab('history')">📂 Tra cứu lịch sử</button>
</div>
<!-- TAB 1: DỰ BÁO TƯƠNG LAI -->
<div class="tab-content active" id="tab-realtime">
  <div class="panel">
    <div class="panel-title">Dự báo mực nước tương lai</div>
    <p style="color:#94a3b8;font-size:.88rem;margin-bottom:16px;line-height:1.5">
      Hệ thống tự động đọc <b style="color:#38bdf8">mực nước</b> và <b style="color:#38bdf8">lượng mưa</b>
      từ dữ liệu quan trắc mới nhất trong dataset.<br>
      Chỉ cần bấm nút bên dưới để nhận kết quả dự báo cho <b style="color:#818cf8">5 chân trời</b> tương lai.
    </p>
    <div class="control-row">
      <button class="btn-predict" id="btnRealtime" onclick="runRealtime()" style="font-size:1.05rem;padding:14px 36px">
        ⚡ Dự báo tương lai
      </button>
      <div class="spinner" id="spinRealtime"></div>
    </div>

    <!-- Cấu hình giả lập kịch bản (Tùy chọn) -->
    <div style="margin-top:20px;border-top:1px solid rgba(255,255,255,.06);padding-top:16px">
      <div onclick="toggleConfig()" style="cursor:pointer;display:flex;align-items:center;justify-content:space-between;font-size:.85rem;color:#64748b;user-select:none;font-weight:600">
        <span>⚙️ Thiết lập kịch bản giả lập (Tùy chọn nâng cao)</span>
        <span id="configToggleIcon" style="transition:transform .2s">▼</span>
      </div>
      <div id="configFields" style="display:none;margin-top:16px">
        <div class="form-grid">
          <div class="form-group">
            <label for="wlInput">Mực nước hiện tại H(t) (m):</label>
            <input type="number" id="wlInput" step="0.01" placeholder="Mặc định: 39.50" min="30" max="50">
            <span class="unit">Mặc định lấy từ dữ liệu thực tế mới nhất</span>
          </div>
          <div class="form-group">
            <label for="rainInput">Lượng mưa ngày hôm nay (mm):</label>
            <input type="number" id="rainInput" step="0.1" placeholder="Mặc định: 0.04" min="0" max="500">
            <span class="unit">Lượng mưa trong ngày dự báo</span>
          </div>
        </div>
        <div class="form-grid" style="grid-template-columns:1fr">
          <div class="form-group">
            <label for="customDateInput">Ngày dự báo:</label>
            <input type="date" id="customDateInput" min="2017-01-22">
            <span class="unit">Ngày bắt đầu dự báo (Mặc định: Ngày hôm nay)</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<!-- TAB 2: TRA CỨU LỊCH SỬ -->
<div class="tab-content" id="tab-history">
  <div class="panel">
    <div class="panel-title">Chọn ngày tra cứu từ dữ liệu lịch sử (2017–2025)</div>
    <div class="control-row">
      <label for="dateInput">📅 Ngày:</label>
      <input type="date" id="dateInput" value="2025-12-28" min="2017-01-22" max="2025-12-28">
      <button class="btn-predict" id="btnHistory" onclick="runHistory()">🔍 Tra cứu</button>
      <div class="spinner" id="spinHistory"></div>
    </div>
  </div>
</div>
<div class="note-bar" id="noteBar"></div>
<div class="alert-banner" id="alertBanner"></div>
<div class="error-msg" id="errorMsg"></div>
<div class="results" id="results">
  <div class="info-bar" id="infoBar"></div>
  <div class="kpi-row" id="kpiRow"></div>
  <div class="chart-card">
    <h3>📈 Biểu đồ dự báo mực nước + Khoảng tin cậy 95%</h3>
    <canvas id="forecastChart" height="85"></canvas>
  </div>
</div>
<footer>
  <a href="/docs">📖 Tài liệu API</a> &nbsp;·&nbsp;
  <a href="/health">/health</a> &nbsp;·&nbsp;
  <a href="/features">/features</a> &nbsp;·&nbsp;
  Đồ án Tốt nghiệp © 2026
</footer>
<script>
let chartInstance = null;
document.addEventListener("DOMContentLoaded", () => {
  const today = new Date();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, '0');
  const dd = String(today.getDate()).padStart(2, '0');
  document.getElementById('customDateInput').value = `${yyyy}-${mm}-${dd}`;
});
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach((b,i) =>
    b.classList.toggle('active', (tab==='realtime'&&i===0)||(tab==='history'&&i===1)));
  document.getElementById('tab-realtime').classList.toggle('active', tab==='realtime');
  document.getElementById('tab-history').classList.toggle('active', tab==='history');
  ['results','alertBanner','noteBar','errorMsg'].forEach(id =>
    document.getElementById(id).style.display='none');
}
function toggleConfig() {
  const fields = document.getElementById('configFields');
  const icon = document.getElementById('configToggleIcon');
  if(fields.style.display === 'none') {
    fields.style.display = 'block';
    icon.textContent = '▲';
  } else {
    fields.style.display = 'none';
    icon.textContent = '▼';
  }
}
async function runRealtime() {
  const btn=document.getElementById('btnRealtime');
  const spin=document.getElementById('spinRealtime');
  btn.disabled=true;spin.style.display='block';hideAll();
  
  const bodyData = {};
  const wlVal = document.getElementById('wlInput').value;
  const rainVal = document.getElementById('rainInput').value;
  const dateVal = document.getElementById('customDateInput').value;
  
  if (wlVal !== "") bodyData.water_level_m = parseFloat(wlVal);
  if (rainVal !== "") bodyData.rain_today_mm = parseFloat(rainVal);
  if (dateVal !== "") bodyData.date = dateVal;

  try{
    const resp=await fetch('/forecast-realtime',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(bodyData)
    });
    if(!resp.ok){const e=await resp.json();throw new Error(e.detail||`HTTP ${resp.status}`);}
    renderResults(await resp.json(),true);
  }catch(e){showError(e.message);}
  finally{btn.disabled=false;spin.style.display='none';}
}
async function runHistory() {
  const date=document.getElementById('dateInput').value;
  const btn=document.getElementById('btnHistory');
  const spin=document.getElementById('spinHistory');
  btn.disabled=true;spin.style.display='block';hideAll();
  try{
    const resp=await fetch(date?`/forecast?date=${date}`:'/forecast');
    if(!resp.ok){const e=await resp.json();throw new Error(e.detail||`HTTP ${resp.status}`);}
    renderResults(await resp.json(),false);
  }catch(e){showError(e.message);}
  finally{btn.disabled=false;spin.style.display='none';}
}
function hideAll(){['results','alertBanner','noteBar','errorMsg'].forEach(id=>document.getElementById(id).style.display='none');}
function showError(msg){const el=document.getElementById('errorMsg');el.textContent='⚠️ '+msg;el.style.display='block';}
function renderResults(data,isRealtime){
  const noteEl=document.getElementById('noteBar');
  if(isRealtime&&data.note){noteEl.textContent='💡 '+data.note;noteEl.style.display='block';}else{noteEl.style.display='none';}
  const alertEl=document.getElementById('alertBanner');
  const cls={'BÌNH THƯỜNG':'alert-ok','CẢNH BÁO':'alert-warn','NGUY HIỂM':'alert-danger'};
  alertEl.className='alert-banner '+(cls[data.alert_level]||'alert-ok');
  alertEl.innerHTML=data.alert_message;alertEl.style.display='block';
  const mode=isRealtime?'⚡ Dự báo tương lai':'📂 Tra cứu lịch sử';
  let ih=`<span>📅 Ngày: <b>${data.issue_date}</b></span><span>💧 H(t): <b>${data.base_level_m.toFixed(2)} m</b></span><span>🤖 Models: <b>t+${data.models_used.join('d, t+')}d</b></span><span>📌 ${mode}</span>`;
  if(isRealtime&&data.rain_today_mm!==undefined)ih+=`<span>🌧️ Mưa: <b>${data.rain_today_mm}mm</b></span>`;
  document.getElementById('infoBar').innerHTML=ih;
  const kpiRow=document.getElementById('kpiRow');kpiRow.innerHTML='';
  const base=data.base_level_m;
  data.forecasts.forEach(f=>{
    const d=f.horizon_d,wl=f.water_level_m.toFixed(3);
    const delta=f.delta_m!==undefined?f.delta_m:f.water_level_m-base;
    const sign=delta>=0?'▲':'▼',dcls=delta>=0?'up':'down';
    const dateStr=f.forecast_date?`<div class="kpi-date">${f.forecast_date}</div>`:'';
    kpiRow.innerHTML+=`<div class="kpi"><div class="kpi-label">t+${d} ngày</div>${dateStr}<div class="kpi-value">${wl}</div><div class="kpi-unit">mét</div><div class="kpi-delta ${dcls}">${sign}${Math.abs(delta).toFixed(3)} m</div></div>`;
  });
  const labels=data.forecasts.map(f=>f.forecast_date||`t+${f.horizon_d}d`);
  const vals=data.forecasts.map(f=>f.water_level_m);
  const lowers=data.forecasts.map(f=>f.ci95_lower);
  const uppers=data.forecasts.map(f=>f.ci95_upper);
  const ciW=uppers.map((u,i)=>u-lowers[i]);
  if(chartInstance)chartInstance.destroy();
  const ctx=document.getElementById('forecastChart').getContext('2d');
  chartInstance=new Chart(ctx,{
    data:{labels,datasets:[
      {type:'bar',label:'CI95 (độ rộng)',data:ciW,backgroundColor:'rgba(99,102,241,.18)',borderColor:'rgba(99,102,241,.4)',borderWidth:1,yAxisID:'y2',order:2},
      {type:'line',label:'Mực nước dự báo (m)',data:vals,borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,.12)',borderWidth:3,pointRadius:7,pointBackgroundColor:'#38bdf8',fill:false,tension:.35,yAxisID:'y',order:1,z:10},
      {type:'line',label:'CI95 dưới',data:lowers,borderColor:'rgba(56,189,248,.3)',borderWidth:1.5,borderDash:[4,4],pointRadius:4,fill:false,yAxisID:'y',order:3},
      {type:'line',label:'CI95 trên',data:uppers,borderColor:'rgba(56,189,248,.3)',borderWidth:1.5,borderDash:[4,4],pointRadius:4,fill:false,yAxisID:'y',order:4},
      {type:'line',label:`H(t)=${base.toFixed(2)}m`,data:Array(labels.length).fill(base),borderColor:'rgba(251,191,36,.6)',borderWidth:1.5,borderDash:[6,3],pointRadius:0,fill:false,yAxisID:'y',order:5},
    ]},
    options:{responsive:true,interaction:{mode:'index',intersect:false},
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:11,family:'Outfit'}}},tooltip:{callbacks:{label:c=>{const v=c.parsed.y;return v==null?'':`${c.dataset.label}: ${v.toFixed(4)}`}}}},
      scales:{x:{ticks:{color:'#94a3b8'},grid:{color:'rgba(255,255,255,.05)'}},y:{ticks:{color:'#94a3b8',callback:v=>v.toFixed(2)+'m'},grid:{color:'rgba(255,255,255,.05)'},title:{display:true,text:'Mực nước (m)',color:'#64748b'}},y2:{position:'right',ticks:{color:'#6366f1',callback:v=>v.toFixed(2)+'m'},grid:{drawOnChartArea:false},title:{display:true,text:'Độ rộng CI95',color:'#6366f1'}}}}
  });
  document.getElementById('results').style.display='block';
}
</script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)



# ---------------------------------------------------------------------------
# Endpoint: POST /predict
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Dự báo mực nước hồ Núi Cốc",
    description=(
        f"Nhận ma trận đặc trưng {WINDOW_SIZE}×{FEATURE_COUNT} ({WINDOW_SIZE} ngày gần nhất, {FEATURE_COUNT} đặc trưng ngày), "
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

    base_level_m = request.base_level_m

    # ---- Validate shape ----
    features_raw = request.features
    n_rows = len(features_raw)
    if n_rows != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Số bước thời gian không hợp lệ: nhận được {n_rows}, "
                f"yêu cầu {WINDOW_SIZE} ngày."
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
                base_level_m=base_level_m,
                target_scaler=target_scaler,
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
        - models_loaded: danh sách chân trời (ngày) đã có model sẵn sàng
        - feature_count: số đặc trưng đầu vào (16)
        - window_size: kích thước cửa sổ thời gian (21 ngày)
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

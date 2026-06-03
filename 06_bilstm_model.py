"""
Bước 6: Huấn luyện và đánh giá mô hình Bi-LSTM (v5.1 — chống overfitting)
==========================================================================
  - Bi-LSTM 2 chiều, 64 units/chiều (→ output 128), window 21 ngày
  - Dự báo ΔH = H(t+d) - H(t), ghép lại H(t) khi inference
  - 16 features (không có water_level_m — tránh học vẹt)
  - sample_weight: quan trắc thật=1.0, nội suy/synthetic=0.25
  - Ensemble với persistence (cấu hình ENSEMBLE_PERSISTENCE_WEIGHT)
Split: Train(2019-2022) | Test(2023, EarlyStopping) | Val(2024+, báo cáo)
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

def evaluate_threshold_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """Tính F1-Score cho bài toán phân loại nhị phân: vượt ngưỡng vận hành hay không?"""
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    # Bắt buộc tính recall & precision cho sự kiện vượt ngưỡng (Positive class)
    return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from config import (
    WINDOW_SIZE,
    FORECAST_DAYS,
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    MIN_DELTA_ES,
    LSTM_UNITS,
    USE_BIDIRECTIONAL,
    RECURRENT_DROPOUT,
    DROPOUT_RATE,
    L2_REG,
    LEARNING_RATE,
    MC_SAMPLES,
    FEATURE_COLS,
    TARGET_COL,
    BASE_LEVEL_COL,
    PREDICT_DELTA_H,
    ENSEMBLE_PERSISTENCE_WEIGHT,
    ES_VAL_FRACTION,
    SAMPLE_WEIGHT_FLOOD,
    FLOOD_DELTA_THRESHOLD_M,
    APPLY_LAG_D_ALIGNMENT,
    TRAIN_CONFIG_PATH,
    MNDBT,
    target_delta_col,
    target_abs_col,
)

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD DỮ LIỆU
# ============================================================
def load_dataset_csv(path: str) -> pd.DataFrame:
    """
    Đọc file CSV được lưu bởi df.to_csv() với DatetimeIndex.
    Cột đầu tiên là index timestamp.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy: '{path}'\n"
            "Hãy chạy '05_integrate.py' trước."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index của '{path}' không phải DatetimeIndex!")
    return df


def validate_and_select_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple:
    """
    Kiểm tra bộ dữ liệu có đủ features (v5) không.
    """
    missing = [c for c in FEATURE_COLS if c not in df_train.columns]
    if missing:
        raise KeyError(
            f"Thiếu {len(missing)} cột đặc trưng trong dataset: {missing}\n"
            "Ảy chạy lại '05_integrate.py' để tạo đúng bộ dữ liệu."
        )
    logger.info("✓ Đủ %d đặc trưng ngày (v5).", len(FEATURE_COLS))
    return df_train, df_val, df_test, FEATURE_COLS


# ============================================================
# TẠO SEQUENCES
# ============================================================
def create_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    window_size: int,
) -> tuple:
    """
    Tạo (X, y, timestamps, base_levels, weights) cho LSTM.
    Cửa sổ features kết thúc tại ngày i (gồm lag đến ngày i).
    """
    features = df[feature_cols].values
    targets  = df[target_col].values
    bases    = df[BASE_LEVEL_COL].values
    weights  = (
        df["sample_weight"].values
        if "sample_weight" in df.columns
        else np.ones(len(df))
    )
    idx = df.index
    X, y, timestamps, base_levels, sw = [], [], [], [], []

    for i in range(window_size - 1, len(df)):
        X.append(features[i - window_size + 1 : i + 1])
        y.append(targets[i])
        timestamps.append(idx[i])
        base_levels.append(bases[i])
        sw.append(weights[i])

    return (
        np.array(X),
        np.array(y),
        pd.DatetimeIndex(timestamps),
        np.array(base_levels),
        np.array(sw),
    )


# ============================================================
# KIẾN TRÚC MÔ HÌNH LSTM (v5)
# ============================================================
def build_bilstm(input_shape: tuple, lstm_units: list,
                 dropout_rate: float) -> Model:
    """
    Kiến trúc Bi-LSTM (v5.1).

    Nếu USE_BIDIRECTIONAL = True (mặc định):
      - Bi-LSTM: xử lý sequence cả hai chiều (forward + backward)
      - Mỗi chiều: lstm_units[0] units → output ghép = 2 * lstm_units[0]
      - Vận dụng: học các quy luật thủy văn có tính chu kỳ tốt hơn LSTM đơn hướng

    Nếu USE_BIDIRECTIONAL = False (fallback):
      - LSTM đơn hướng thông thường.

    Đầu vào: (batch, WINDOW_SIZE, n_features)
    Đầu ra: dự báo ΔH (scalar)
    """
    inputs = Input(shape=input_shape, name="input_sequence")
    if USE_BIDIRECTIONAL:
        x = Bidirectional(
            LSTM(
                lstm_units[0],
                return_sequences=False,
                recurrent_dropout=RECURRENT_DROPOUT,
                name="lstm_fwd_bwd",
            ),
            merge_mode="concat",    # output dim = 2 * lstm_units[0]
            name="bilstm_1",
        )(inputs)
    else:
        x = LSTM(
            lstm_units[0],
            return_sequences=False,
            recurrent_dropout=RECURRENT_DROPOUT,
            name="lstm_1",
        )(inputs)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    # Dense layer: input dim tự động phù hợp với cả Bi-LSTM (128) và LSTM (64)
    x = Dense(32, activation="relu", kernel_regularizer=l2(L2_REG), name="dense_1")(x)
    outputs = Dense(1, activation="linear", name="output")(x)
    model_name = "BiLSTM_v51" if USE_BIDIRECTIONAL else "LSTM_v51"
    return Model(inputs=inputs, outputs=outputs, name=model_name)


def delta_to_level(
    delta_m: np.ndarray,
    base_m: np.ndarray,
    target_scaler,
) -> np.ndarray:
    """Chuyển ΔH (scaled) → mực nước tuyệt đối (m)."""
    if target_scaler is not None:
        delta_m = target_scaler.inverse_transform(
            delta_m.reshape(-1, 1)
        ).flatten()
    return base_m + delta_m


def blend_with_persistence(
    pred_level: np.ndarray,
    base_m: np.ndarray,
    weight_persist: float = ENSEMBLE_PERSISTENCE_WEIGHT,
) -> np.ndarray:
    """Hợp nhất: w * H(t) + (1-w) * H_model."""
    w = float(weight_persist)
    return w * base_m + (1.0 - w) * pred_level


def apply_lag_d_alignment(
    df: pd.DataFrame,
    horizon_d: int,
    cols: list,
) -> pd.DataFrame:
    """
    Hiệu chỉnh lệch d ngày (không đổi mô hình).

    Khi mô hình gần persistence: pred(valid=T) ≈ H(T-d) thay vì H(T).
    Dịch chuỗi dự báo NGƯỢC d ngày trên trục valid_time:
        pred_aligned(T) = pred_raw(T + d)

    Công thức pandas: shift(-d) trên index valid_time.
    """
    out = df.sort_values("valid_time").copy()
    vt = pd.to_datetime(out["valid_time"])
    for col in cols:
        if col not in out.columns:
            continue
        s = pd.Series(out[col].values, index=vt)
        out[f"{col}_aligned"] = s.shift(-horizon_d).reindex(vt).values
    return out


# ============================================================
# MONTE CARLO DROPOUT — ƯỚC LƯỢNG ĐỘ KHÔNG CHẮC CHẮN
# ============================================================
def predict_with_mc_dropout(
    model: Model,
    X_input: np.ndarray,
    n_samples: int = MC_SAMPLES,
) -> tuple:
    """
    Monte Carlo Dropout — ước lượng khoảng tin cậy 95% cho dự báo.

    Ý tưởng cốt lõi:
      Trong training, Dropout ngẫu nhiên tắt một số neuron mỗi forward pass.
      Thông thường, inference tắt Dropout (training=False).
      MC Dropout BẬT Dropout cả trong inference (training=True),
      chạy N lần → N dự báo khác nhau → phân phối xấp xỉ posterior.

    Parameters
    ----------
    model : keras.Model
        Mô hình Bi-LSTM đã huấn luyện (có Dropout layers).
    X_input : np.ndarray
        Shape (n_samples_data, window_size, n_features).
    n_samples : int
        Số lần chạy MC (càng nhiều càng chính xác, mặc định 50).

    Returns
    -------
    tuple
        (mean_pred, ci95_lower, ci95_upper) — mỗi array shape (n_data,)
    """
    # Chạy n_samples lần với Dropout bật (training=True)
    mc_preds = np.stack([
        model(X_input, training=True).numpy().flatten()
        for _ in range(n_samples)
    ], axis=0)  # shape: (n_samples, n_data)

    mean_pred  = mc_preds.mean(axis=0)
    std_pred   = mc_preds.std(axis=0)

    # Khoảng tin cậy 95% theo phân phối chuẩn
    ci95_lower = mean_pred - 1.96 * std_pred
    ci95_upper = mean_pred + 1.96 * std_pred

    return mean_pred, ci95_lower, ci95_upper


# ============================================================
# CHỈ SỐ ĐÁNH GIÁ THỦY VĂN
# ============================================================
def nash_sutcliffe_efficiency(y_true: np.ndarray,
                               y_pred: np.ndarray) -> float:
    """
    Hệ số hiệu quả Nash-Sutcliffe (NSE) — chỉ số chuẩn trong thủy văn.

    NSE = 1 - Σ(obs - sim)² / Σ(obs - mean(obs))²

    NSE = 1.0: Mô hình hoàn hảo
    NSE = 0.0: Mô hình ngang bằng dùng giá trị trung bình
    NSE < 0.0: Mô hình kém hơn dùng giá trị trung bình

    Thang đánh giá: NSE ≥ 0.75 → Tốt | 0.60–0.75 → Khá | < 0.60 → Yếu
    """
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - num / den) if den != 0 else np.nan


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     label: str = "") -> dict:
    """
    Tính đầy đủ các chỉ số đánh giá — đáp ứng yêu cầu hội đồng và chuẩn WMO:
      - RMSE  : Sại số căn trung bình bình phương
      - MAE   : Sại số tuyệt đối trung bình
      - R²    : Hệ số xác định (Coefficient of Determination)
      - NSE   : Hệ số Nash-Sutcliffe (chuẩn vàng thủy văn học)
      - PBIAS : Sai lệch phần trăm — Percent Bias (chuẩn WMO)
    """
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    nse   = nash_sutcliffe_efficiency(y_true, y_pred)

    # R² (Coefficient of Determination) — được đề cập trong đề tài
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan

    # PBIAS (Percent Bias) — chuẩn WMO, dương = dự báo thiên cao, âm = thiên thấp
    pbias = float(100.0 * np.sum(y_pred - y_true) / np.sum(y_true)) if np.sum(y_true) != 0 else np.nan

    logger.info(
        "  [%s] RMSE=%.4fm | MAE=%.4fm | R²=%.4f | NSE=%.4f | PBIAS=%.2f%%",
        label, rmse, mae, r2, nse, pbias,
    )
    return {"rmse": rmse, "mae": mae, "r2": r2, "nse": nse, "pbias": pbias}


# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================
def compute_shap_importance(model: Model, X_train: np.ndarray,
                             X_test: np.ndarray, feature_cols: list,
                             horizon_d: int) -> None:
    """
    Tính và visualize SHAP values để giải thích mô hình (XAI).

    SHAP (SHapley Additive exPlanations) định lượng đóng góp của
    từng feature vào mỗi dự báo — trả lời câu hỏi:
    "Tại sao mô hình dự báo mực nước cao/thấp tại thời điểm này?"

    Đây là yêu cầu quan trọng trong nghiên cứu thủy văn:
    mô hình "hộp đen" cần được giải thích để các nhà quản lý
    hồ chứa tin tưởng và sử dụng kết quả dự báo.

    Parameters
    ----------
    model : keras.Model
        Mô hình đã huấn luyện.
    X_train, X_test : np.ndarray
        Dữ liệu train (làm background) và test (giải thích).
    feature_cols : list of str
        Tên các features.
    horizon_d : int
        Khoảng dự báo tính bằng ngày (để đặt tên file lưu).
    """
    try:
        import shap

        logger.info("  [SHAP] Tính feature importance cho t+%dd...", horizon_d)

        # Dùng 100 mẫu train làm background (tránh OOM)
        background = X_train[:100]
        test_sample = X_test[:50]

        # GradientExplainer phù hợp với model TensorFlow/Keras
        explainer   = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_sample)

        # shap_values shape: (n_test, window, n_features)
        # Lấy mean absolute SHAP theo chiều time và sample
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 4:
            shap_arr = shap_arr[0]   # regression output index

        mean_shap = np.abs(shap_arr).mean(axis=(0, 1))   # (n_features,)

        # Sắp xếp và vẽ biểu đồ bar
        sorted_idx  = np.argsort(mean_shap)[::-1]
        sorted_vals = mean_shap[sorted_idx]
        sorted_feat = [feature_cols[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = ["crimson" if "Q_out" in f or "dH" in f else "steelblue"
                   for f in sorted_feat]
        ax.barh(range(len(sorted_feat)), sorted_vals[::-1],
                color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(len(sorted_feat)))
        ax.set_yticklabels(sorted_feat[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(
            f"SHAP Feature Importance — Bi-LSTM t+{horizon_d}d\n"
            "(Màu đỏ: features Q_out mới)",
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        save_path = f"results/shap_importance_t{horizon_d}d.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  [SHAP] Đã lưu: %s", save_path)

        # Lưu SHAP values số để dùng trong báo cáo
        shap_df = pd.DataFrame({
            "feature":       sorted_feat,
            "mean_abs_shap": sorted_vals,
        })
        shap_df.to_csv(f"results/shap_values_t{horizon_d}d.csv", index=False)

    except ImportError:
        logger.warning(
            "[SHAP] Thư viện 'shap' chưa được cài đặt."
        )
    except Exception as exc:
        logger.warning("[SHAP] Không thể tính SHAP: %s", exc)


# ============================================================
# HUẤN LUYỆN & ĐÁNH GIÁ
# ============================================================
def train_and_evaluate(
    horizon_d: int,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    Huấn luyện Bi-LSTM và đánh giá đầy đủ cho một khoảng dự báo (ngày).

    Quy trình:
      1. Tạo sequences từ dữ liệu đã chuẩn hóa
      2. Build model Bi-LSTM
      3. Train với EarlyStopping + ReduceLROnPlateau
      4. Đánh giá trên val và test (lũ Yagi 2024)
      5. Dự báo với MC Dropout → khoảng tin cậy 95%
      6. Tính SHAP feature importance
      7. Lưu biểu đồ, kết quả

    Returns
    -------
    tuple
        (metrics_val, metrics_test, trained_model)
    """
    target_col = (
        target_delta_col(horizon_d) if PREDICT_DELTA_H else target_abs_col(horizon_d)
    )
    abs_col = target_abs_col(horizon_d)
    model_path = f"models/bilstm_t{horizon_d}d.keras"

    logger.info("=" * 58)
    logger.info(
        "  HUẤN LUYỆN: t+%dd | %d features | LSTM v5 | target=%s",
        horizon_d, len(feature_cols), target_col,
    )
    logger.info("=" * 58)

    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for col in (target_col, abs_col, BASE_LEVEL_COL):
            if col not in split.columns:
                raise KeyError(
                    f"Thiếu cột '{col}' trong tập '{name}'.\nChạy lại 05_integrate.py."
                )

    X_train, y_train, ts_train, base_train, sw_train = create_sequences(
        df_train, feature_cols, target_col, WINDOW_SIZE
    )
    # Tăng trọng số sự kiện biến động mạnh (|ΔH| lớn)
    y_delta_raw = df_train.loc[ts_train, target_col].values
    sw_train = sw_train * np.where(
        np.abs(y_delta_raw) >= FLOOD_DELTA_THRESHOLD_M,
        SAMPLE_WEIGHT_FLOOD,
        1.0,
    )
    X_test, y_test, ts_test, base_test, sw_test = create_sequences(
        df_test, feature_cols, target_col, WINDOW_SIZE
    )
    X_val, y_val, ts_val, base_val, sw_val = create_sequences(
        df_val, feature_cols, target_col, WINDOW_SIZE
    )
    y_val_abs = df_val.loc[ts_val, abs_col].values
    y_test_abs = df_test.loc[ts_test, abs_col].values

    logger.info("  Train: %s | Test (EarlyStopping): %s | Val (Bao cao): %s",
                X_train.shape, X_test.shape, X_val.shape)

    import joblib
    from sklearn.preprocessing import StandardScaler

    # Chuẩn hóa biến mục tiêu (Target Scaling) bằng StandardScaler thay vì MinMaxScaler.
    # Lý do: MinMaxScaler ép mục tiêu về [0, 1]. Khi gặp bão Yagi (47.6m) cao hơn max của tập Train (46.8m),
    # target sẽ vượt ra ngoài khoảng [0, 1] (Extrapolation). Mạng Neural rất khó dự báo vượt ngưỡng này.
    # StandardScaler (z-score) không có biên cứng, giúp mô hình "thoáng" hơn trong việc ngoại suy đỉnh lũ.
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled  = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # EarlyStopping: 85% đầu train = fit, 15% cuối = val nội bộ (cùng pipeline/scaler)
    n_es = max(int(len(X_train) * ES_VAL_FRACTION), 1)
    n_es = min(n_es, len(X_train) - 1)
    X_fit, y_fit_s, sw_fit = X_train[:-n_es], y_train_scaled[:-n_es], sw_train[:-n_es]
    X_es, y_es_s = X_train[-n_es:], y_train_scaled[-n_es:]
    logger.info(
        "  Fit: %s | ES-val (cuối train): %s | Test 2023 (báo cáo): %s",
        X_fit.shape, X_es.shape, X_test.shape,
    )

    # Lưu target scaler phục vụ API
    scaler_path = f"models/target_scaler_t{horizon_d}d.pkl"
    joblib.dump(target_scaler, scaler_path)
    logger.info("  [Scaler] Đã lưu target scaler: %s", scaler_path)

    model = build_bilstm(
        input_shape=(WINDOW_SIZE, len(feature_cols)),
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.Huber(delta=1.0),  # Huber: ít nhạy với outlier/đỉnh lũ
        metrics=["mae"],
    )

    if horizon_d == FORECAST_DAYS[0]:
        model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=PATIENCE, min_delta=MIN_DELTA_ES,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(model_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
        CSVLogger(f"results/training_log_t{horizon_d}d.csv"),
    ]

    # Huấn luyện trên target đã chuẩn hóa
    history = model.fit(
        X_fit, y_fit_s,
        validation_data=(X_es, y_es_s),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=sw_fit,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred_test_det_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_val_det_scaled  = model.predict(X_val, verbose=0).flatten()

    if PREDICT_DELTA_H:
        y_pred_test_det = delta_to_level(y_pred_test_det_scaled, base_test, target_scaler)
        y_pred_val_det  = delta_to_level(y_pred_val_det_scaled, base_val, target_scaler)
    else:
        y_pred_test_det = target_scaler.inverse_transform(
            y_pred_test_det_scaled.reshape(-1, 1)
        ).flatten()
        y_pred_val_det = target_scaler.inverse_transform(
            y_pred_val_det_scaled.reshape(-1, 1)
        ).flatten()

    y_pred_test_det = blend_with_persistence(y_pred_test_det, base_test)
    y_pred_val_det  = blend_with_persistence(y_pred_val_det, base_val)

    # ── Dự báo khoảng tin cậy với MC Dropout (Tập Kiểm định / Val set) ────────────────────
    logger.info("  [MC Dropout] Chay %d mau de uoc luong khoang tin cay...", MC_SAMPLES)
    # Lấy các mẫu dự báo dạng scaled
    mc_preds_scaled = np.stack([
        model(X_val, training=True).numpy().flatten()
        for _ in range(MC_SAMPLES)
    ], axis=0)  # shape: (MC_SAMPLES, n_data)

    # Nghịch đảo chuẩn hóa từng mẫu về mét
    mc_levels = np.stack([
        delta_to_level(mc_preds_scaled[i], base_val, target_scaler)
        if PREDICT_DELTA_H
        else target_scaler.inverse_transform(mc_preds_scaled[i].reshape(-1, 1)).flatten()
        for i in range(MC_SAMPLES)
    ], axis=0)
    mc_levels = np.stack([
        blend_with_persistence(mc_levels[i], base_val) for i in range(MC_SAMPLES)
    ], axis=0)
    std_pred_meters = mc_levels.std(axis=0)

    # Điểm dự báo chính (Point Forecast) là dự báo tất định (deterministic) để tối ưu NSE
    y_pred_val_mean = y_pred_val_det

    # Khoảng tin cậy 95% đối xứng quanh điểm dự báo tất định
    y_pred_val_lo = y_pred_val_mean - 1.96 * std_pred_meters
    y_pred_val_hi = y_pred_val_mean + 1.96 * std_pred_meters
    
    ci_width_mean = np.mean(y_pred_val_hi - y_pred_val_lo)
    logger.info("  [MC Dropout] Mean CI width: %.2f m", ci_width_mean) # FIXED: Thêm log để theo dõi độ rộng trung bình của khoảng tin cậy.

    metrics_test = evaluate_metrics(
        y_test_abs, y_pred_test_det,
        label=f"Test (EarlyStopping) t+{horizon_d}d",
    )
    metrics_val = evaluate_metrics(
        y_val_abs, y_pred_val_mean,
        label=f"Val  (Kiem dinh doc lap) t+{horizon_d}d",
    )

    # Naive đúng cho horizon d: H(t+d) ≈ H(t) khi phát hành t
    y_persist_val = base_val
    metrics_persist = evaluate_metrics(
        y_val_abs, y_persist_val,
        label=f"Val  Naive H(t)→H(t+{horizon_d}d) t+{horizon_d}d",
    )
    metrics_val["nse_persistence"] = metrics_persist["nse"]
    metrics_val["rmse_persistence"] = metrics_persist["rmse"]
    logger.info(
        "  [So sanh] NSE model=%.4f vs persistence=%.4f",
        metrics_val["nse"], metrics_persist["nse"],
    )

    XA_LU = 46.50
    f1_mndbt = evaluate_threshold_f1(y_val_abs, y_pred_val_mean, MNDBT)
    f1_xalu  = evaluate_threshold_f1(y_val_abs, y_pred_val_mean, XA_LU)
    
    logger.info(f"  [Metrics Thực Chiến] F1-Score tại MNDBT ({MNDBT}m): {f1_mndbt:.4f}")
    logger.info(f"  [Metrics Thực Chiến] F1-Score tại Ngưỡng Xả lũ ({XA_LU}m): {f1_xalu:.4f}")
    # Đẩy vào object metrics để visualize nếu cần
    metrics_val["f1_mndbt"] = f1_mndbt
    metrics_val["f1_xalu"] = f1_xalu
    # ============================================================

    df_result = pd.DataFrame({
        "issue_time":   ts_val,
        "valid_time":   ts_val + pd.to_timedelta(horizon_d, unit="D"),
        "actual":       y_val_abs,
        "predicted":    y_pred_val_mean,
        "persistence":  y_persist_val,
        "ci95_lower":   y_pred_val_lo,
        "ci95_upper":   y_pred_val_hi,
        "error":        y_pred_val_mean - y_val_abs,
    })
    if APPLY_LAG_D_ALIGNMENT:
        df_result = apply_lag_d_alignment(
            df_result,
            horizon_d,
            cols=["predicted", "ci95_lower", "ci95_upper"],
        )
        df_result["error_aligned"] = (
            df_result["predicted_aligned"] - df_result["actual"]
        )
        mask = df_result["predicted_aligned"].notna()
        if mask.sum() > 10:
            m_al = evaluate_metrics(
                df_result.loc[mask, "actual"].values,
                df_result.loc[mask, "predicted_aligned"].values,
                label=f"Val  Sau căn lệch d (t+{horizon_d}d)",
            )
            metrics_val["nse_aligned"] = m_al["nse"]
            metrics_val["rmse_aligned"] = m_al["rmse"]
    df_result.to_csv(f"results/predictions_t{horizon_d}d.csv", index=False)

    plot_results(df_result, history, horizon_d, metrics_val, len(feature_cols))

    # SHAP (cho khoảng dự báo đầu tiên và cuối để tiết kiệm thời gian)
    if horizon_d in [1, 30]:
        compute_shap_importance(model, X_train, X_test, feature_cols, horizon_d)

    if horizon_d == FORECAST_DAYS[0]:
        train_cfg = {
            "version": "5.0",
            "predict_delta_h": PREDICT_DELTA_H,
            "window_size": WINDOW_SIZE,
            "feature_cols": FEATURE_COLS,
            "ensemble_persistence_weight": ENSEMBLE_PERSISTENCE_WEIGHT,
            "lstm_units": LSTM_UNITS,
            "dropout_rate": DROPOUT_RATE,
        }
        with open(TRAIN_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(train_cfg, f, ensure_ascii=False, indent=2)

    return metrics_val, metrics_test, model


# ============================================================
# BIỂU ĐỒ KẾT QUẢ (có khoảng tin cậy MC Dropout)
# ============================================================
def plot_results(df_result: pd.DataFrame, history,
                 horizon_d: int, metrics: dict,
                 n_features: int) -> None:
    """
    Vẽ 2 biểu đồ:
      1. Đường cong loss (train vs val)
      2. Dự báo vs Thực tế theo valid_time (= issue_time + horizon)
         — trục X là ngày mực nước được dự báo, tránh lệch 1 ngày trên đồ thị
    """
    df_plot_src = df_result.copy()
    if "valid_time" not in df_plot_src.columns:
        issue_col = (
            "issue_time" if "issue_time" in df_plot_src.columns else "timestamp"
        )
        df_plot_src["valid_time"] = (
            pd.to_datetime(df_plot_src[issue_col])
            + pd.to_timedelta(horizon_d, unit="D")
        )
    time_col = "valid_time"
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle(
        f"LSTM v5 | t+{horizon_d}d | {n_features} features\n"
        f"RMSE={metrics['rmse']:.4f}m | MAE={metrics['mae']:.4f}m | NSE={metrics['nse']:.4f}",
        fontsize=12, fontweight="bold",
    )

    # Biểu đồ 1: Loss curve
    ax1 = axes[0]
    ax1.plot(history.history["loss"],     label="Train Loss", color="steelblue")
    ax1.plot(history.history["val_loss"], label="Val Loss",   color="orange")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Huber Loss (ΔH scaled)")
    ax1.set_title(
        "Đường cong hội tộ — Train (85% đầu) vs Val nội bộ (15% cuối train)\n"
        "(Không dùng 2023 — tránh khe loss train/val giả)"
    )
    ax1.legend(); ax1.grid(alpha=0.3)

    # Biểu đồ 2: trục X = valid_time (ngày mực nước t+d), không phải ngày phát hành
    ax2 = axes[1]
    yagi_mask = (
        (df_plot_src[time_col] >= "2024-09-07")
        & (df_plot_src[time_col] <= "2024-09-15")
    )
    df_plot = df_plot_src[yagi_mask] if yagi_mask.sum() >= 2 else df_plot_src.tail(100)
    t_axis = df_plot[time_col]

    ax2.plot(t_axis, df_plot["actual"],
             label="Thực tế H(valid)", color="royalblue", linewidth=2)

    use_aligned = (
        APPLY_LAG_D_ALIGNMENT
        and "predicted_aligned" in df_plot.columns
        and df_plot["predicted_aligned"].notna().any()
    )
    if use_aligned:
        ax2.plot(
            t_axis, df_plot["predicted_aligned"],
            label=f"Dự báo t+{horizon_d}d (đã căn lệch d ngày)",
            color="tomato", linewidth=2.0,
        )
        ax2.plot(
            t_axis, df_plot["predicted"],
            label="Trước căn (raw)",
            color="darkorange", linewidth=1.0, linestyle="--", alpha=0.55,
        )
        lo_col, hi_col = "ci95_lower_aligned", "ci95_upper_aligned"
        if lo_col in df_plot.columns and df_plot[lo_col].notna().any():
            ax2.fill_between(
                t_axis, df_plot[lo_col], df_plot[hi_col],
                alpha=0.20, color="tomato", label="CI95 (đã căn)",
            )
    else:
        ax2.plot(
            t_axis, df_plot["predicted"],
            label=f"Dự báo t+{horizon_d}d",
            color="tomato", linewidth=1.5, linestyle="--",
        )
        ax2.fill_between(
            t_axis, df_plot["ci95_lower"], df_plot["ci95_upper"],
            alpha=0.20, color="tomato", label="Khoảng tin cậy 95%",
        )

    if "persistence" in df_plot.columns:
        ax2.plot(
            t_axis, df_plot["persistence"],
            label="Naive H(ngày phát hành)",
            color="gray", linewidth=1.0, linestyle=":",
        )

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1 if len(df_plot) < 20 else 7))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.set_xlabel("Ngày mực nước được dự báo (valid time)")
    ax2.set_ylabel("Mực nước (m)")
    align_note = (
        " — hiển thị đã căn lệch d (post-process)"
        if use_aligned else ""
    )
    nse_al = metrics.get("nse_aligned")
    sub = (
        f"NSE (căn lệch)={nse_al:.4f} | " if nse_al is not None else ""
    )
    ax2.set_title(
        f"Dự báo vs Thực tế — Lũ Yagi 9/2024 (t+{horizon_d}d){align_note}\n"
        f"({sub}đường đỏ = pred_aligned, gạch cam = raw)"
    )
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"results/plot_t{horizon_d}d.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  [Biểu đồ] Đã lưu: %s", save_path)


# ============================================================
# BẢNG TỔNG HỢP
# ============================================================
def print_summary_table(all_metrics: dict) -> None:
    """In bang tong hop va luu JSON ket qua.

    Chi bao cao chi so tren tap TEST (kiem dinh doc lap 2024-2025).
    Chi so Val chi hien thi de tham khao kiem tra overfitting.
    """
    print("\n" + "=" * 90)
    print("  BANG TONG HOP — Bi-LSTM v5.1 (Ho Nui Coc)")
    print("  [BAO CAO CHINH: Val 2024-2025 — bao gom lu Yagi 9/2024]")
    print("=" * 90)
    print(
        f"{'Khoang':>10} | {'RMSE':>8} | {'MAE':>8} | {'R2':>8} "
        f"| {'NSE':>8} | {'NSE_pers':>8} | {'PBIAS%':>8} | {'Danh gia':>8}"
    )
    print("-" * 90)

    for d, (val_m, test_m) in all_metrics.items():
        nse   = val_m["nse"]
        nse_p = val_m.get("nse_persistence", float("nan"))
        r2    = val_m.get("r2", float("nan"))
        pbias = val_m.get("pbias", float("nan"))
        tag   = "Tot" if nse >= 0.75 else ("Kha" if nse >= 0.60 else "Yeu")
        print(
            f"  t+{d:>2}d [Val ] | {val_m['rmse']:>8.4f} | {val_m['mae']:>8.4f} | {r2:>8.4f} "
            f"| {nse:>8.4f} | {nse_p:>8.4f} | {pbias:>7.2f}% | {tag:>8}"
        )
    print("=" * 90)
    print("  Test (2023, EarlyStopping) metrics [chi tham khao overfitting]:")
    for d, (val_m, test_m) in all_metrics.items():
        r2_t = test_m.get('r2', float('nan'))
        print(f"    t+{d:>2}d [Test]: RMSE={test_m['rmse']:.4f}m  NSE={test_m['nse']:.4f}  R2={r2_t:.4f}")

    results_json = {f"t+{d}d": {"val": v, "test": t}
                    for d, (v, t) in all_metrics.items()}
    with open("results/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu: results/metrics_summary.json")


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("BƯỚC 6: HUẤN LUYỆN Bi-LSTM (v5.1 — ΔH + anti-overfit + Bidirectional)")
    logger.info("Thời gian: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("Kiến trúc: %s | %d units/chiều | PATIENCE=%d",
                "Bi-LSTM" if USE_BIDIRECTIONAL else "LSTM", LSTM_UNITS[0], PATIENCE)
    logger.info("=" * 60)

    tf.random.set_seed(42)
    np.random.seed(42)

    logger.info("\n[Load] Doc du lieu tu data/final/ ...")
    df_train = load_dataset_csv("data/final/dataset_train.csv")
    df_test  = load_dataset_csv("data/final/dataset_test.csv")
    df_val   = load_dataset_csv("data/final/dataset_val.csv")

    logger.info("  Train (Huan luyen)        : %d ban ghi (%s -> %s)",
                len(df_train), df_train.index.min().date(), df_train.index.max().date())
    logger.info("  Test  (EarlyStopping)     : %d ban ghi (%s -> %s)  <- dung som",
                len(df_test), df_test.index.min().date(), df_test.index.max().date())
    logger.info("  Val   (Kiem dinh doc lap) : %d ban ghi (%s -> %s)  <- BAO CAO LUAN VAN",
                len(df_val), df_val.index.min().date(), df_val.index.max().date())

    # Dam bao khong co data leakage: tap Test phai ket thuc truoc tap Val
    assert df_test.index.max() < df_val.index.min(), (
        f"DATA LEAKAGE: Test ket thuc {df_test.index.max().date()} "
        f">= Val bat dau {df_val.index.min().date()}!"
    )

    # Chọn feature set
    df_train, df_test, df_val, feature_cols = validate_and_select_features(
        df_train, df_test, df_val
    )
    logger.info("  Số features: %d", len(feature_cols))

    # Huấn luyện từng khoảng dự báo
    all_metrics = {}
    for d in FORECAST_DAYS:
        # Chú ý: hàm train_and_evaluate định nghĩa arg thứ 3 là df_val, arg thứ 4 là df_test.
        # Ở đây ta truyền df_val (2024-2025) vào arg df_val, df_test (2023) vào arg df_test.
        val_m, test_m, _ = train_and_evaluate(
            d, df_train, df_val, df_test, feature_cols
        )
        all_metrics[d] = (val_m, test_m)

    print_summary_table(all_metrics)
    logger.info(
        "\n✓ Hoàn thành! Kết quả tại 'results/'.\n"
        "  → Tiếp theo: python 06b_baseline_comparison.py (ablation study)\n"
        "  → Tiếp theo: python 08_api_serve.py (khởi động API)\n"
        "  → Hoặc: pip install shap && python 06_bilstm_model.py (SHAP)"
    )


if __name__ == "__main__":
    main()
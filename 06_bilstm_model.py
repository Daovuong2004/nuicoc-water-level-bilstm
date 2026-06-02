"""
Bước 6: Huấn luyện và đánh giá mô hình Bi-LSTM
==================================================
Kiến trúc mô hình Bi-LSTM (v4.0):
  Input(60, 26)
  → BiLSTM(256, return_seq=True) → Dropout(0.25) → BatchNorm
  → BiLSTM(128, return_seq=False) → Dropout(0.25) → BatchNorm
  → Dense(64, relu, L2) → Dense(32, relu) → Dense(1, linear)

Cải tiến so với v3.0:
  1. Bi-LSTM thuần túy — không dùng Self-Attention (đúng tên đề tài)
  2. Window mở rộng 30 → 60 ngày — bắt được chu kỳ mùa mưa/khô
  3. LSTM units tăng [128,64] → [256,128] — tăng capacity mô hình
  4. Huber Loss thay MSE — ít nhạy với đỉnh lũ cực trị
  5. L2 regularization + Dropout 0.25 — chống overfitting tốt hơn
  6. Bộ features mới (26 đặc trưng): thêm rain_30d, lag60, roll60, delta_h_7d/30d

Bộ features (26 đặc trưng):
  - Khí tượng    : rain_1d/3d/7d/14d/30d, temperature, humidity
  - Lag mực nước : water_level_lag1/3/7/14/30/60
  - Rolling stats : roll7/30/60, std7
  - Temporal      : month_sin/cos, season_wet/dry
  - Xu hướng     : delta_h_7d, delta_h_30d
  - Q_out         : dH_dt_daily, Q_out_daily, Q_out_roll7

Cửa sổ đầu vào : 60 ngày
Đầu ra         : dự báo t+1, t+3, t+7, t+14, t+30 (m)
Split          : Train(2017-2022) | Val(2023, EarlyStopping) | Test(2024-2025, báo cáo)
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
from keras.models import Model
from keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization,
)
from keras.regularizers import l2
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# CẤU HÌNH SIÊU THAM SỐ (v4.0 — Bi-LSTM thuần túy)
# ============================================================
WINDOW_SIZE    = 60            # Cửa sổ nhìn lại (ngày) — 30 → 60 (bắt chu kỳ mùa)
FORECAST_DAYS  = [1, 3, 7, 14, 30]
BATCH_SIZE     = 32
MAX_EPOCHS     = 300          # 200 → 300
PATIENCE       = 20           # 15 → 20 (EarlyStopping kiên nhẫn hơn)
LSTM_UNITS     = [256, 128]   # [128,64] → [256,128] (tăng capacity)
DROPOUT_RATE   = 0.25         # 0.2 → 0.25
L2_REG         = 1e-4         # L2 regularization cho Dense layers [MỚI]
LEARNING_RATE  = 0.001
MC_SAMPLES     = 50           # Số mẫu Monte Carlo Dropout

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# FEATURE COLUMNS — 26 đặc trưng (v4.0, đồng bộ với 05_integrate.py)
# ============================================================
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

TARGET_COL = "water_level_m"


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
    Kiểm tra bộ dữ liệu có đủ 26 features (v4.0) không.
    """
    missing = [c for c in FEATURE_COLS if c not in df_train.columns]
    if missing:
        raise KeyError(
            f"Thiếu {len(missing)} cột đặc trưng trong dataset: {missing}\n"
            "Ảy chạy lại '05_integrate.py' để tạo đúng bộ dữ liệu."
        )
    logger.info("✓ Đủ 26 đặc trưng ngày (v4.0).")
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
    Tạo cặp (X, y) dạng cửa sổ trượt cho LSTM.

    X[i] = features tại giờ [i-window_size .. i-1]
    y[i] = target tại giờ i

    Returns: X (n, window, features), y (n,), timestamps
    """
    features   = df[feature_cols].values
    targets    = df[target_col].values
    idx        = df.index
    X, y, timestamps = [], [], []

    for i in range(window_size, len(df)):
        X.append(features[i - window_size:i])
        y.append(targets[i])
        timestamps.append(idx[i])

    return np.array(X), np.array(y), pd.DatetimeIndex(timestamps)


# ============================================================
# KIẾN TRÚC MÔ HÌNH Bi-LSTM
# ============================================================
def build_bilstm(input_shape: tuple, lstm_units: list,
                  dropout_rate: float) -> Model:
    """
    Bi-LSTM 2 lớp thuần túy — kiến trúc chính của đề tài.

    Lý do dùng Bi-LSTM cho bài toán thủy văn:
      Bi-LSTM xử lý sequence theo cả 2 chiều (xuôi và ngược).
      Chiều xuôi: học xu hướng tăng mực nước theo mưa tích lũy.
      Chiều ngược: học bối cảnh thủy văn tương lai (như xu hướng giảm).
      Kết hợp cả 2 chiều giúp mô hình nắm bắt tốt hơn tính tuần hoàn
      mùa mưa/khô của hồ chứa.

    Kiến trúc:
      BiLSTM(256) → Dropout(0.25)
      → BiLSTM(128) → Dropout(0.25)
      → Dense(64, relu, L2) → Dense(32, relu) → Dense(1, linear)

    Parameters
    ----------
    input_shape : tuple
        (window_size, n_features) ví dụ (60, 26)
    lstm_units : list of int
        Số unit LSTM lớp 1 và 2.
    dropout_rate : float
        Tỉ lệ Dropout.

    Returns
    -------
    keras.Model
    """
    inputs = Input(shape=input_shape, name="input_sequence")

    # ── Lớp BiLSTM 1: trích xuất đặc trưng chuỗi ────────────────────────
    x = Bidirectional(
        LSTM(lstm_units[0], return_sequences=True, name="bilstm_1"),
        name="bidirectional_1",
    )(inputs)
    x = Dropout(dropout_rate, name="dropout_1")(x)

    # ── Lớp BiLSTM 2: tổng hợp thành vector cố định ─────────────────
    x = Bidirectional(
        LSTM(lstm_units[1], return_sequences=False, name="bilstm_2"),
        name="bidirectional_2",
    )(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)

    # ── Lớp Dense: ánh xạ sang không gian dự báo ──────────────────
    x       = Dense(64, activation="relu", kernel_regularizer=l2(L2_REG),
                    name="dense_1")(x)
    x       = Dense(32, activation="relu", name="dense_2")(x)
    outputs = Dense(1,  activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="BiLSTM_v4")


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
    """Tính RMSE, MAE, NSE cho một tập dự báo."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    nse  = nash_sutcliffe_efficiency(y_true, y_pred)
    logger.info("  [%s] RMSE=%.4fm | MAE=%.4fm | NSE=%.4f", label, rmse, mae, nse)
    return {"rmse": rmse, "mae": mae, "nse": nse}


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
    target_col = f"target_t{horizon_d}d"
    model_path = f"models/bilstm_t{horizon_d}d.keras"

    logger.info("=" * 58)
    logger.info("  HUẤN LUYỆN: t+%dd | %d features | Bi-LSTM", horizon_d, len(feature_cols))
    logger.info("=" * 58)

    # Kiểm tra cột target
    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if target_col not in split.columns:
            raise KeyError(
                f"Thiếu cột '{target_col}' trong tập '{name}'.\n"
                f"Chạy lại 05_integrate.py."
            )

    # Tao sequences
    # X_val  -> dung cho EarlyStopping + ReduceLR (khong dung de bao cao)
    # X_test -> dung de tinh RMSE/MAE/NSE bao cao trong luan van
    X_train, y_train, _        = create_sequences(df_train, feature_cols, target_col, WINDOW_SIZE)
    X_val,   y_val,   _        = create_sequences(df_val,   feature_cols, target_col, WINDOW_SIZE)
    X_test,  y_test,  ts_test  = create_sequences(df_test,  feature_cols, target_col, WINDOW_SIZE)
    logger.info("  Train: %s | Val (EarlyStopping): %s | Test (Bao cao): %s",
                X_train.shape, X_val.shape, X_test.shape)

    import joblib
    from sklearn.preprocessing import MinMaxScaler

    # Chuẩn hóa biến mục tiêu (Target Scaling) để tăng hiệu quả hội tụ
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    # Lưu target scaler phục vụ API
    scaler_path = f"models/target_scaler_t{horizon_d}d.pkl"
    joblib.dump(target_scaler, scaler_path)
    logger.info("  [Scaler] Đã lưu target scaler: %s", scaler_path)

    # Build model Bi-LSTM thuần túy
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
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
        CSVLogger(f"results/training_log_t{horizon_d}d.csv"),
    ]

    # Huấn luyện trên target đã chuẩn hóa
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Dự báo tất định (deterministic) ────────────────────
    y_pred_val_scaled = model.predict(X_val, verbose=0).flatten()
    y_pred_val = target_scaler.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()

    y_pred_test_det_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_test_det = target_scaler.inverse_transform(y_pred_test_det_scaled.reshape(-1, 1)).flatten()

    # ── Dự báo khoảng tin cậy với MC Dropout (Test set) ────────────────────
    logger.info("  [MC Dropout] Chay %d mau de uoc luong khoang tin cay...", MC_SAMPLES)
    # Lấy các mẫu dự báo dạng scaled
    mc_preds_scaled = np.stack([
        model(X_test, training=True).numpy().flatten()
        for _ in range(MC_SAMPLES)
    ], axis=0)  # shape: (MC_SAMPLES, n_data)

    # Nghịch đảo chuẩn hóa từng mẫu về mét
    mc_preds_meters = np.stack([
        target_scaler.inverse_transform(mc_preds_scaled[i].reshape(-1, 1)).flatten()
        for i in range(MC_SAMPLES)
    ], axis=0)  # shape: (MC_SAMPLES, n_data)

    # Tính độ lệch chuẩn trên đơn vị mét
    std_pred_meters = mc_preds_meters.std(axis=0)

    # Điểm dự báo chính (Point Forecast) là dự báo tất định (deterministic) để tối ưu NSE
    y_pred_test_mean = y_pred_test_det

    # Khoảng tin cậy 95% đối xứng quanh điểm dự báo tất định
    y_pred_test_lo = y_pred_test_mean - 1.96 * std_pred_meters
    y_pred_test_hi = y_pred_test_mean + 1.96 * std_pred_meters

    # Danh gia trên đơn vị mét:
    metrics_val  = evaluate_metrics(y_val, y_pred_val,
                                    label=f"Val  (EarlyStopping) t+{horizon_d}d")
    metrics_test = evaluate_metrics(y_test, y_pred_test_mean,
                                    label=f"Test (Kiem dinh doc lap) t+{horizon_d}d")

    # Lưu kết quả
    df_result = pd.DataFrame({
        "timestamp":    ts_test,
        "actual":       y_test,
        "predicted":    y_pred_test_mean,
        "ci95_lower":   y_pred_test_lo,
        "ci95_upper":   y_pred_test_hi,
        "error":        y_pred_test_mean - y_test,
    })
    df_result.to_csv(f"results/predictions_t{horizon_d}d.csv", index=False)

    # Vẽ biểu đồ
    plot_results(df_result, history, horizon_d, metrics_test, len(feature_cols))

    # SHAP (cho khoảng dự báo đầu tiên và cuối để tiết kiệm thời gian)
    if horizon_d in [1, 30]:
        compute_shap_importance(model, X_train, X_test, feature_cols, horizon_d)

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
      2. Dự báo vs Thực tế + Khoảng tin cậy 95% (MC Dropout)
         tại sự kiện lũ Yagi tháng 9/2024
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle(
        f"Bi-LSTM | t+{horizon_d}d | {n_features} features\n"
        f"RMSE={metrics['rmse']:.4f}m | MAE={metrics['mae']:.4f}m | NSE={metrics['nse']:.4f}",
        fontsize=12, fontweight="bold",
    )

    # Biểu đồ 1: Loss curve
    ax1 = axes[0]
    ax1.plot(history.history["loss"],     label="Train Loss", color="steelblue")
    ax1.plot(history.history["val_loss"], label="Val Loss",   color="orange")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss")
    ax1.set_title("Đường cong hội tụ (Training curve)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Biểu đồ 2: Dự báo vs Thực tế với khoảng tin cậy
    ax2 = axes[1]
    yagi_mask = (
        (df_result["timestamp"] >= "2024-09-07")
        & (df_result["timestamp"] <= "2024-09-15")
    )
    df_plot = df_result[yagi_mask] if yagi_mask.sum() >= 2 else df_result.tail(100)

    ax2.plot(df_plot["timestamp"], df_plot["actual"],
             label="Thực tế", color="royalblue", linewidth=2)
    ax2.plot(df_plot["timestamp"], df_plot["predicted"],
             label=f"Dự báo t+{horizon_d}d", color="tomato",
             linewidth=1.5, linestyle="--")

    # Khoảng tin cậy 95% từ MC Dropout
    ax2.fill_between(
        df_plot["timestamp"],
        df_plot["ci95_lower"], df_plot["ci95_upper"],
        alpha=0.20, color="tomato", label="Khoảng tin cậy 95% (MC Dropout)",
    )

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1 if len(df_plot) < 20 else 7))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.set_xlabel("Thời gian"); ax2.set_ylabel("Mực nước (m)")
    ax2.set_title("Dự báo vs Thực tế — Lũ Yagi tháng 9/2024\n(Vùng bóng: Khoảng tin cậy 95%)")
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
    print("\n" + "=" * 74)
    print("  BANG TONG HOP — Bi-LSTM (Ho Nui Coc)")
    print("  [BAO CAO CHINH: Test 2024-2025 — bao gom lu Yagi 9/2024]")
    print("=" * 74)
    print(f"{'Khoang':>10} | {'RMSE (m)':>10} | {'MAE (m)':>10} | {'NSE':>8} | {'Danh gia':>10}")
    print("-" * 74)

    for d, (val_m, test_m) in all_metrics.items():
        nse = test_m["nse"]
        tag = "Tot" if nse >= 0.75 else ("Kha" if nse >= 0.60 else "Yeu")
        print(
            f"  t+{d:>2}d [Test] | "
            f"{test_m['rmse']:>10.4f} | "
            f"{test_m['mae']:>10.4f} | "
            f"{nse:>8.4f} | {tag:>10}"
        )
    print("=" * 74)
    print("  Val (2023, EarlyStopping) metrics [chi tham khao overfitting]:")
    for d, (val_m, test_m) in all_metrics.items():
        print(f"    t+{d:>2}d [Val ]: RMSE={val_m['rmse']:.4f}m  NSE={val_m['nse']:.4f}")

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
    logger.info("BƯỚC 6: HUẤN LUYỆN Bi-LSTM (v4.0)")
    logger.info("Thời gian: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    tf.random.set_seed(42)
    np.random.seed(42)

    logger.info("\n[Load] Doc du lieu tu data/final/ ...")
    df_train = load_dataset_csv("data/final/dataset_train.csv")
    df_val   = load_dataset_csv("data/final/dataset_val.csv")
    df_test  = load_dataset_csv("data/final/dataset_test.csv")

    logger.info("  Train (Calibration)       : %d ban ghi (%s -> %s)",
                len(df_train), df_train.index.min().date(), df_train.index.max().date())
    logger.info("  Val   (EarlyStopping)     : %d ban ghi (%s -> %s)  <- khong bao cao",
                len(df_val), df_val.index.min().date(), df_val.index.max().date())
    logger.info("  Test  (Kiem dinh doc lap) : %d ban ghi (%s -> %s)  <- BAO CAO LUAN VAN",
                len(df_test), df_test.index.min().date(), df_test.index.max().date())

    # Dam bao khong co data leakage: tap Val phai ket thuc truoc tap Test
    assert df_val.index.max() < df_test.index.min(), (
        f"DATA LEAKAGE: Val ket thuc {df_val.index.max().date()} "
        f">= Test bat dau {df_test.index.min().date()}!"
    )

    # Chọn feature set
    df_train, df_val, df_test, feature_cols = validate_and_select_features(
        df_train, df_val, df_test
    )
    logger.info("  Số features: %d", len(feature_cols))

    # Huấn luyện từng khoảng dự báo
    all_metrics = {}
    for d in FORECAST_DAYS:
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
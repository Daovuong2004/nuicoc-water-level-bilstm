"""
Bước 6: Huấn luyện và đánh giá mô hình Bi-LSTM + Self-Attention
================================================================
Kiến trúc mô hình (v3.0):
  Input(48, 18)
  → BiLSTM(128, return_seq=True) → Dropout(0.2) → BatchNorm
  → Multi-Head Self-Attention(heads=4, key_dim=32) → Residual + LayerNorm
  → BiLSTM(64, return_seq=False)  → Dropout(0.2) → BatchNorm
  → Dense(32, relu) → Dense(1, linear)

Cải tiến so với v2.0:
  1. Multi-Head Self-Attention — học "timestep nào quan trọng nhất"
     trong cửa sổ 48h (giải quyết long-range dependency)
  2. Monte Carlo Dropout — ước lượng khoảng tin cậy 95% cho dự báo
  3. SHAP Feature Importance — giải thích mô hình (XAI)
  4. Lưu attention weights để visualize

Bộ features (18 đặc trưng):
  - Khí tượng    : rain_1h/6h/24h, temperature, humidity
  - Lag mực nước : water_level_lag1/2/3/6/12
  - Cửa xả       : so_cua_xa, dang_xa_cua
  - Q_out        : Q_out_smooth, Q_out_lag1, Q_out_lag6,
                   Q_out_roll24, dQout_dt, xa_dot_ngot

Cửa sổ đầu vào : 48 giờ
Đầu ra         : dự báo t+1, t+3, t+6, t+12, t+24 (m)
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
    BatchNormalization, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Add,
)
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger,
)
from keras.saving import register_keras_serializable
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# CẤU HÌNH SIÊU THAM SỐ
# ============================================================
WINDOW_SIZE    = 48            # Cửa sổ nhìn lại (giờ)
FORECAST_HOURS = [1, 3, 6, 12, 24]
BATCH_SIZE     = 32
MAX_EPOCHS     = 200
PATIENCE       = 15
LSTM_UNITS     = [128, 64]
DROPOUT_RATE   = 0.2
LEARNING_RATE  = 0.001
ATTN_HEADS     = 4             # Số đầu Attention
ATTN_KEY_DIM   = 32            # Chiều key/query của Attention
MC_SAMPLES     = 50            # Số mẫu Monte Carlo Dropout

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# FEATURE COLUMNS — 18 đặc trưng (đồng bộ với 05_integrate.py)
# ============================================================
FEATURE_COLS = [
    "rain_1h", "rain_6h", "rain_24h",
    "temperature", "humidity",
    "water_level_lag1", "water_level_lag2", "water_level_lag3",
    "water_level_lag6", "water_level_lag12",
    "so_cua_xa", "dang_xa_cua",
    "Q_out_smooth", "Q_out_lag1", "Q_out_lag6",
    "Q_out_roll24", "dQout_dt", "xa_dot_ngot",
]

# Dự phòng nếu chưa chạy bước 7 (thiếu cột Q_out)
FEATURE_COLS_FALLBACK = [
    "rain_1h", "rain_6h", "rain_24h",
    "temperature", "humidity",
    "water_level_lag1", "water_level_lag2", "water_level_lag3",
    "water_level_lag6", "water_level_lag12",
    "so_cua_xa", "dang_xa_cua",
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
    Kiểm tra bộ dữ liệu có đủ 18 features (v2.0+) không.
    Tự động fallback về 12 features nếu thiếu cột Q_out.
    """
    qout_cols = [c for c in FEATURE_COLS if c not in FEATURE_COLS_FALLBACK]
    missing   = [c for c in qout_cols if c not in df_train.columns]

    if missing:
        logger.warning(
            "Thiếu %d cột Q_out: %s → dùng 12 features (fallback).", len(missing), missing
        )
        feature_cols = FEATURE_COLS_FALLBACK
    else:
        logger.info("✓ Đủ 18 features (v2.0 với Q_out).")
        feature_cols = FEATURE_COLS

    return df_train, df_val, df_test, feature_cols


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
# KIẾN TRÚC MÔ HÌNH Bi-LSTM + MULTI-HEAD SELF-ATTENTION
# ============================================================
def build_bilstm_attention(input_shape: tuple, lstm_units: list,
                            dropout_rate: float) -> Model:
    """
    Bi-LSTM hai lớp với Multi-Head Self-Attention ở giữa.

    Lý do dùng Attention trong bài toán thủy văn:
      Mực nước lúc 8h sáng bị ảnh hưởng bởi cơn mưa lúc 20h hôm qua,
      không phải chỉ 1h trước. LSTM xử lý kém khi "bước thời gian quan
      trọng" nằm xa trong sequence. Self-Attention giải quyết điều này:
      mỗi timestep "nhìn" tất cả các timestep khác và học trọng số
      quan trọng — cho phép mô hình tập trung vào đỉnh mưa quan trọng
      dù nó nằm ở đầu cửa sổ 48h.

    Kiến trúc:
      BiLSTM(128) → Dropout → BatchNorm
      → [Multi-Head Attention (4 heads, key_dim=32) + Residual + LayerNorm]
      → BiLSTM(64) → Dropout → BatchNorm
      → Dense(32) → Dense(1)

    Parameters
    ----------
    input_shape : tuple
        (window_size, n_features)
    lstm_units : list of int
        Số unit LSTM lớp 1 và 2.
    dropout_rate : float
        Tỉ lệ Dropout (dùng cả trong MC Dropout inference).

    Returns
    -------
    keras.Model
    """
    inputs = Input(shape=input_shape, name="input_sequence")

    # ── Lớp BiLSTM 1: trích xuất đặc trưng chuỗi ──────────────
    x = Bidirectional(
        LSTM(lstm_units[0], return_sequences=True, name="bilstm_1"),
        name="bidirectional_1",
    )(inputs)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = BatchNormalization(name="bn_1")(x)

    # ── Multi-Head Self-Attention ───────────────────────────────
    # Q = K = V = x (self-attention)
    # Mỗi timestep "nhìn" toàn bộ sequence, học trọng số tầm quan trọng
    attn_out = MultiHeadAttention(
        num_heads=ATTN_HEADS,
        key_dim=ATTN_KEY_DIM,
        dropout=dropout_rate,
        name="multi_head_attention",
    )(x, x)  # query=x, value=x (self-attention)

    # Residual connection + Layer Normalization (theo kiến trúc Transformer)
    x = Add(name="residual_add")([x, attn_out])
    x = LayerNormalization(epsilon=1e-6, name="layer_norm")(x)

    # ── Lớp BiLSTM 2: tổng hợp thành vector cố định ───────────
    x = Bidirectional(
        LSTM(lstm_units[1], return_sequences=False, name="bilstm_2"),
        name="bidirectional_2",
    )(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)
    x = BatchNormalization(name="bn_2")(x)

    # ── Lớp Dense: ánh xạ sang không gian dự báo ──────────────
    x       = Dense(32, activation="relu", name="dense_1")(x)
    outputs = Dense(1,  activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs)


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

    Ý nghĩa thực tiễn trong thủy văn:
      Thay vì nói "mực nước t+6h là 46.5m", mô hình nói:
      "46.5m ± 0.3m (95% CI)" — thông tin quan trọng hơn nhiều
      cho đơn vị vận hành hồ chứa trong quyết định xả cửa.

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
                             horizon_h: int) -> None:
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
    horizon_h : int
        Khoảng dự báo (để đặt tên file lưu).
    """
    try:
        import shap

        logger.info("  [SHAP] Tính feature importance cho t+%dh...", horizon_h)

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
        colors  = ["crimson" if "Q_out" in f or "xa" in f else "steelblue"
                   for f in sorted_feat]
        ax.barh(range(len(sorted_feat)), sorted_vals[::-1],
                color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(len(sorted_feat)))
        ax.set_yticklabels(sorted_feat[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(
            f"SHAP Feature Importance — Bi-LSTM+Attention t+{horizon_h}h\n"
            "(Màu đỏ: features Q_out mới thêm trong v2.0)",
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        save_path = f"results/shap_importance_t{horizon_h}h.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  [SHAP] Đã lưu: %s", save_path)

        # Lưu SHAP values số để dùng trong báo cáo
        shap_df = pd.DataFrame({
            "feature":       sorted_feat,
            "mean_abs_shap": sorted_vals,
        })
        shap_df.to_csv(f"results/shap_values_t{horizon_h}h.csv", index=False)

    except ImportError:
        logger.warning(
            "[SHAP] Thư viện 'shap' chưa được cài đặt. "
            "Chạy: pip install shap"
        )
    except Exception as exc:
        logger.warning("[SHAP] Không thể tính SHAP: %s", exc)


# ============================================================
# HUẤN LUYỆN & ĐÁNH GIÁ
# ============================================================
def train_and_evaluate(
    horizon_h: int,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    Huấn luyện Bi-LSTM+Attention và đánh giá đầy đủ cho một khoảng dự báo.

    Quy trình:
      1. Tạo sequences từ dữ liệu đã chuẩn hóa
      2. Build model Bi-LSTM+Attention
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
    target_col = f"target_t{horizon_h}h"
    model_path = f"models/bilstm_t{horizon_h}h.keras"

    logger.info("=" * 58)
    logger.info("  HUẤN LUYỆN: t+%dh | %d features | Attention", horizon_h, len(feature_cols))
    logger.info("=" * 58)

    # Kiểm tra cột target
    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if target_col not in split.columns:
            raise KeyError(
                f"Thiếu cột '{target_col}' trong tập '{name}'.\n"
                f"Chạy lại 05_integrate.py."
            )

    # Tạo sequences
    X_train, y_train, _        = create_sequences(df_train, feature_cols, target_col, WINDOW_SIZE)
    X_val,   y_val,   _        = create_sequences(df_val,   feature_cols, target_col, WINDOW_SIZE)
    X_test,  y_test,  ts_test  = create_sequences(df_test,  feature_cols, target_col, WINDOW_SIZE)
    logger.info("  Train: %s | Val: %s | Test: %s", X_train.shape, X_val.shape, X_test.shape)

    # Build model
    model = build_bilstm_attention(
        input_shape=(WINDOW_SIZE, len(feature_cols)),
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    if horizon_h == FORECAST_HOURS[0]:
        model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
        CSVLogger(f"results/training_log_t{horizon_h}h.csv"),
    ]

    # Huấn luyện
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Dự báo trung bình (deterministic) ────────────────────
    y_pred_val  = model.predict(X_val,  verbose=0).flatten()

    # ── Dự báo với MC Dropout (probabilistic) ────────────────
    logger.info("  [MC Dropout] Chạy %d mẫu để ước lượng khoảng tin cậy...", MC_SAMPLES)
    y_pred_test_mean, y_pred_test_lo, y_pred_test_hi = predict_with_mc_dropout(
        model, X_test, n_samples=MC_SAMPLES
    )

    # Đánh giá
    metrics_val  = evaluate_metrics(y_val, y_pred_val,
                                    label=f"Val  t+{horizon_h}h")
    metrics_test = evaluate_metrics(y_test, y_pred_test_mean,
                                    label=f"Test t+{horizon_h}h (Yagi)")

    # Lưu kết quả
    df_result = pd.DataFrame({
        "timestamp":    ts_test,
        "actual":       y_test,
        "predicted":    y_pred_test_mean,
        "ci95_lower":   y_pred_test_lo,
        "ci95_upper":   y_pred_test_hi,
        "error":        y_pred_test_mean - y_test,
    })
    df_result.to_csv(f"results/predictions_t{horizon_h}h.csv", index=False)

    # Vẽ biểu đồ
    plot_results(df_result, history, horizon_h, metrics_test, len(feature_cols))

    # SHAP (cho khoảng dự báo đầu tiên và cuối để tiết kiệm thời gian)
    if horizon_h in [1, 24]:
        compute_shap_importance(model, X_train, X_test, feature_cols, horizon_h)

    return metrics_val, metrics_test, model


# ============================================================
# BIỂU ĐỒ KẾT QUẢ (có khoảng tin cậy MC Dropout)
# ============================================================
def plot_results(df_result: pd.DataFrame, history,
                 horizon_h: int, metrics: dict,
                 n_features: int) -> None:
    """
    Vẽ 2 biểu đồ:
      1. Đường cong loss (train vs val)
      2. Dự báo vs Thực tế + Khoảng tin cậy 95% (MC Dropout)
         tại sự kiện lũ Yagi tháng 9/2024
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle(
        f"Bi-LSTM + Self-Attention | t+{horizon_h}h | {n_features} features\n"
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
    df_plot = df_result[yagi_mask] if yagi_mask.sum() > 10 else df_result.tail(500)

    ax2.plot(df_plot["timestamp"], df_plot["actual"],
             label="Thực tế", color="royalblue", linewidth=2)
    ax2.plot(df_plot["timestamp"], df_plot["predicted"],
             label=f"Dự báo t+{horizon_h}h", color="tomato",
             linewidth=1.5, linestyle="--")

    # Khoảng tin cậy 95% từ MC Dropout
    ax2.fill_between(
        df_plot["timestamp"],
        df_plot["ci95_lower"], df_plot["ci95_upper"],
        alpha=0.20, color="tomato", label="Khoảng tin cậy 95% (MC Dropout)",
    )

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.set_xlabel("Thời gian"); ax2.set_ylabel("Mực nước (m)")
    ax2.set_title("Dự báo vs Thực tế — Lũ Yagi tháng 9/2024\n(Vùng bóng: Khoảng tin cậy 95%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"results/plot_t{horizon_h}h.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  [Biểu đồ] Đã lưu: %s", save_path)


# ============================================================
# BẢNG TỔNG HỢP
# ============================================================
def print_summary_table(all_metrics: dict) -> None:
    """In bảng tổng hợp và lưu JSON kết quả."""
    print("\n" + "=" * 70)
    print("  BẢNG TỔNG HỢP — Bi-LSTM + Self-Attention (Hồ Núi Cốc)")
    print("=" * 70)
    print(f"{'Khoảng':>10} | {'RMSE (m)':>10} | {'MAE (m)':>10} | {'NSE':>8} | {'Đánh giá':>10}")
    print("-" * 70)

    for h, (val_m, test_m) in all_metrics.items():
        nse = test_m["nse"]
        tag = "Tốt ✓" if nse >= 0.75 else ("Khá" if nse >= 0.60 else "Yếu ✗")
        print(
            f"  t+{h:>2}h (Test) | "
            f"{test_m['rmse']:>10.4f} | "
            f"{test_m['mae']:>10.4f} | "
            f"{nse:>8.4f} | {tag:>10}"
        )
    print("=" * 70)

    results_json = {f"t+{h}h": {"val": v, "test": t}
                    for h, (v, t) in all_metrics.items()}
    with open("results/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu: results/metrics_summary.json")


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("BƯỚC 6: HUẤN LUYỆN Bi-LSTM + SELF-ATTENTION (v3.0)")
    logger.info("Thời gian: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    tf.random.set_seed(42)
    np.random.seed(42)

    # Load dữ liệu
    logger.info("\n[Load] Đọc dữ liệu từ data/final/ ...")
    df_train = load_dataset_csv("data/final/dataset_train.csv")
    df_val   = load_dataset_csv("data/final/dataset_val.csv")
    df_test  = load_dataset_csv("data/final/dataset_test.csv")

    logger.info("  Train : %d bản ghi (%s → %s)",
                len(df_train), df_train.index.min().date(), df_train.index.max().date())
    logger.info("  Val   : %d bản ghi (%s → %s)",
                len(df_val), df_val.index.min().date(), df_val.index.max().date())
    logger.info("  Test  : %d bản ghi (%s → %s)",
                len(df_test), df_test.index.min().date(), df_test.index.max().date())

    # Chọn feature set
    df_train, df_val, df_test, feature_cols = validate_and_select_features(
        df_train, df_val, df_test
    )
    logger.info("  Số features: %d", len(feature_cols))

    # Huấn luyện từng khoảng dự báo
    all_metrics = {}
    for h in FORECAST_HOURS:
        val_m, test_m, _ = train_and_evaluate(
            h, df_train, df_val, df_test, feature_cols
        )
        all_metrics[h] = (val_m, test_m)

    print_summary_table(all_metrics)
    logger.info(
        "\n✓ Hoàn thành! Kết quả tại 'results/'.\n"
        "  → Tiếp theo: python 06b_baseline_comparison.py (ablation study)\n"
        "  → Tiếp theo: python 08_api_serve.py (khởi động API)\n"
        "  → Hoặc: pip install shap && python 06_bilstm_model.py (SHAP)"
    )


if __name__ == "__main__":
    main()
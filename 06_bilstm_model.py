"""
Bước 6: Huấn luyện và đánh giá mô hình Bi-LSTM (v7 — Phase-Aware Loss + TLCC Rain Lags)
==========================================================================
Kiến trúc thực tế (khớp với code bên dưới):
  - Bidirectional(LSTM(64, recurrent_dropout=0.2)) → output 128 chiều
  - Dropout(0.5)
  - Dense(32, relu) + L2(1e-3)
  - Dense(1, linear) → Dự báo ΔH (scaled)
  Cửa sổ: 21 ngày | 20 features (+4 lag mưa TLCC, không có water_level_m)
  Riêng t+7d: window=45, LSTM(96), Dropout(0.3), Dense(64), 25 features

[v7] Thay đổi chống trễ pha:
  Loss = Huber(delta=1.0)
       + λ_phase * mean(|Δy_true − Δy_pred|)   ← phạt sai lệch độ dốc (độ pha)
       − λ_nse   * NSE(y_true, y_pred)           ← thưởng hiệu quả thủy văn
  Feature: thêm rain_1d_lag1/2/3/5 từ phân tích TLCC

Quy ước tập dữ liệu (rất quan trọng, tránh nhầm lẫn):
  df_train = dataset_train.csv  (2019-04 → 2022-12) : Huấn luyện tham số
  df_test  = dataset_test.csv   (2023-01 → 2023-12) : Theo dõi EarlyStopping
  df_val   = dataset_val.csv    (2024-01 → nay)     : Kết quả chính thức luận văn

Chi tiết EarlyStopping:
  ES thực chất dùng 15% cuối của X_train (ES_VAL_FRACTION=0.15) làm validation.
  Target scaler = StandardScaler() (không phải MinMaxScaler) — hỗ trợ ngoại suy đỉnh lũ.
  Ensemble persistence weight = 0.0 (tắt) — tránh đỉnh bị lệch d ngày.
"""

import os
import json
import logging
import traceback
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

# ============================================================
# TÁI LẬP KẺT QUẢ (Reproducibility) — ĐẶT Sử TOÀN CỤC
# ============================================================
# Phải đặt trước khi import Keras/TF để đảm bảo mọi operation đều deterministic.
# Lưu ý: recurrent_dropout > 0 vẫn có thể không hoàn toàn deterministic trên GPU.
tf.random.set_seed(42)
np.random.seed(42)

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
    FEATURE_COLS, FORECAST_DAYS, TARGET_COL, BASE_LEVEL_COL,
    PREDICT_DELTA_H, TARGET_DELTA_PREFIX,
    WINDOW_SIZE, LSTM_UNITS, USE_BIDIRECTIONAL,
    RECURRENT_DROPOUT, DROPOUT_RATE, LEARNING_RATE,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, MIN_DELTA_ES, L2_REG, MC_SAMPLES,
    ENSEMBLE_PERSISTENCE_WEIGHT, ES_VAL_FRACTION,
    SAMPLE_WEIGHT_FLOOD, FLOOD_DELTA_THRESHOLD_M,
    APPLY_LAG_D_ALIGNMENT, MNDBT, CANH_BAO, NGUY_HIEM,
    target_delta_col, target_abs_col,
    # [v6] Config riêng cho t+7d
    FEATURE_COLS_T7D, WINDOW_SIZE_T7D, LSTM_UNITS_T7D,
    DROPOUT_RATE_T7D, DENSE_UNITS_T7D, L2_REG_T7D, PATIENCE_T7D,
    # [v7] Phase-Aware Loss + TLCC
    LAMBDA_PHASE, LAMBDA_NSE, RAIN_LAG_EXTRA,
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
# [v7] PHASE-AWARE LOSS — Kết hợp Huber + Phạt Độ Dốc + Thưởng NSE
# ============================================================
def build_phase_aware_loss(
    lambda_phase: float = LAMBDA_PHASE,
    lambda_nse:   float = LAMBDA_NSE,
):
    """
    Loss tùy chỉnh chống trễ pha cho dự báo thủy văn.

    Công thức:
        L = Huber(δ=1.0)
          + λ_phase * mean(|Δy_true − Δy_pred|)
          − λ_nse   * NSE(y_true, y_pred)

    Giải thích từng thành phần:
      - Huber(δ=1.0)         : ít nhạy với outlier đỉnh lũ (giữ nguyên ưu điểm v5)
      - λ_phase * grad_err   : phạt sai lệch độ dốc (cần sàt cùng xu hướng thực
                               tế), giảm hiệu ứng đồ thị dịch phải d ngày
      - λ_nse   * NSE        : thưởng mô hình có NSE cao — tối đa hoá tiêu
                               chuẩn thủy văn Nash-Sutcliffe Efficiency

    Parameters
    ----------
    lambda_phase : float
        Trọng số penalty gradient (độ dốc). Mặc định 0.1.
        Giảm xuống 0.05 nếu loss dao động mạnh trong quá trình train.
    lambda_nse : float
        Trọng số thưởng NSE. Mặc định 0.05.
        Tăng lên 0.1 nếu muốn ưu tiên hiệu quả thủy văn.

    Returns
    -------
    Callable
        Hàm loss nhận (y_true, y_pred) trả về scalar tensor.
    """
    huber = tf.keras.losses.Huber(delta=1.0)

    def _loss(y_true, y_pred):
        # --- Thành phần 1: Huber Loss cơ bản ---
        loss_val = huber(y_true, y_pred)

        # --- Thành phần 2: Phạt sai lệch độ dốc (gradient phase penalty) ---
        # Tính đạo hàm 1 bước thông qua central difference
        # dy[i] = y[i+1] − y[i] — đưa về cùng khích thước bằng cách pad 0 ở cuối
        if lambda_phase > 0.0:
            dy_true = tf.concat(
                [y_true[1:] - y_true[:-1], tf.zeros_like(y_true[:1])], axis=0
            )
            dy_pred = tf.concat(
                [y_pred[1:] - y_pred[:-1], tf.zeros_like(y_pred[:1])], axis=0
            )
            phase_penalty = tf.reduce_mean(tf.abs(dy_true - dy_pred))
            loss_val = loss_val + lambda_phase * phase_penalty

        # --- Thành phần 3: Thưởng NSE (Nash-Sutcliffe Efficiency) ---
        # NSE càng gần 1.0 → loss càng nhỏ → mô hình được thưởng
        # tránh chia 0 khi y_true constant (ví dụ mùa khô mực nước ổn định)
        if lambda_nse > 0.0:
            y_mean   = tf.reduce_mean(y_true)
            ss_res   = tf.reduce_sum(tf.square(y_true - y_pred))
            ss_tot   = tf.reduce_sum(tf.square(y_true - y_mean))
            nse_val  = 1.0 - ss_res / (ss_tot + 1e-8)
            loss_val = loss_val - lambda_nse * nse_val

        return loss_val

    return _loss


# ============================================================
# KIẾN TRÚC MÔ HÌNH LSTM (v5)
# ============================================================
def build_bilstm(input_shape: tuple, lstm_units: list,
                 dropout_rate: float, dense_units: int = 32,
                 l2_reg: float = None) -> Model:
    """
    Kiến trúc Bi-LSTM (v5.1 + v6 per-horizon config).

    Nếu USE_BIDIRECTIONAL = True (mặc định):
      - Bi-LSTM: xử lý sequence cả hai chiều (forward + backward)
      - Mỗi chiều: lstm_units[0] units → output ghép = 2 * lstm_units[0]
      - Vận dụng: học các quy luật thủy văn có tính chu kỳ tốt hơn LSTM đơn hướng

    Nếu USE_BIDIRECTIONAL = False (fallback):
      - LSTM đơn hướng thông thường.

    Đầu vào: (batch, window_size, n_features)
    Đầu ra: dự báo ΔH (scalar)
    """
    reg = l2(l2_reg if l2_reg is not None else L2_REG)
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
    # Dense layer: size có thể điều chỉnh theo horizon
    x = Dense(dense_units, activation="relu", kernel_regularizer=reg, name="dense_1")(x)
    outputs = Dense(1, activation="linear", name="output")(x)
    model_name = "BiLSTM_v6" if USE_BIDIRECTIONAL else "LSTM_v6"
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

    Parameters
    ----------
    model : keras.Model
    X_train : np.ndarray  (background reference)
    X_test  : np.ndarray  (data to explain)
    feature_cols : list of str
    horizon_d : int
    """
    try:
        import shap

        logger.info("  [SHAP] Tính feature importance cho t+%dd...", horizon_d)

        # Dùng tối đa 100 mẫu train làm background (tránh OOM)
        n_bg = min(100, len(X_train))
        n_te = min(50,  len(X_test))
        background  = X_train[:n_bg]
        test_sample = X_test[:n_te]

        # GradientExplainer phù hợp với model TensorFlow/Keras
        explainer   = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_sample)

        # shap_values có thể là list (multi-output) hoặc ndarray (single output)
        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values[0])  # lấy output đầu tiên
        else:
            shap_arr = np.array(shap_values)

        # Đảm bảo shape đúng: (n_samples, window, n_features)
        if shap_arr.ndim == 4:
            shap_arr = shap_arr[0]   # regression output index
        if shap_arr.ndim != 3:
            raise ValueError(
                f"SHAP array shape không hợp lệ: {shap_arr.shape}, "
                f"cần (n_samples, window, n_features)"
            )

        mean_shap = np.abs(shap_arr).mean(axis=(0, 1))   # (n_features,)

        # Sắp xếp và vẽ biểu đồ bar
        sorted_idx  = np.argsort(mean_shap)[::-1]
        sorted_vals = mean_shap[sorted_idx]
        sorted_feat = [feature_cols[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = ["crimson" if "dH" in f or "delta" in f else "steelblue"
                   for f in sorted_feat]
        ax.barh(range(len(sorted_feat)), sorted_vals[::-1],
                color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(len(sorted_feat)))
        ax.set_yticklabels(sorted_feat[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(
            f"SHAP Feature Importance — Bi-LSTM t+{horizon_d}d",
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
        logger.info("  [SHAP] Top 5 features: %s",
                    dict(zip(sorted_feat[:5], sorted_vals[:5].round(6))))

    except ImportError:
        logger.warning(
            "[SHAP] Thư viện 'shap' chưa được cài đặt. "
            "Cài bằng: pip install shap"
        )
    except Exception as exc:
        logger.warning("[SHAP] Không thể tính SHAP: %s", exc)
        logger.debug("[SHAP] Traceback:\n%s", traceback.format_exc())


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

    # ── [v6] Per-horizon config: t+7d dùng config mở rộng riêng ───────────────────────────
    if horizon_d == 7:
        _window      = WINDOW_SIZE_T7D     # 45 (vs 21)
        _feat_cols   = FEATURE_COLS_T7D    # 21 features (vs 16)
        _lstm_units  = LSTM_UNITS_T7D      # [96] (vs [64])
        _dropout     = DROPOUT_RATE_T7D    # 0.3 (vs 0.5)
        _dense_units = DENSE_UNITS_T7D     # 64 (vs 32)
        _l2_reg      = L2_REG_T7D          # 5e-4 (vs 1e-3)
        _patience    = PATIENCE_T7D        # 25 (vs 20)
        logger.info(
            "  [v6] t+7d EXTENDED CONFIG: window=%d, features=%d, "
            "lstm=%s, dropout=%.1f, dense=%d",
            _window, len(_feat_cols), _lstm_units, _dropout, _dense_units,
        )
    else:
        _window      = WINDOW_SIZE
        _feat_cols   = feature_cols
        _lstm_units  = LSTM_UNITS
        _dropout     = DROPOUT_RATE
        _dense_units = 32
        _l2_reg      = L2_REG
        _patience    = PATIENCE

    logger.info("=" * 58)
    logger.info(
        "  HUẤN LUYỆN: t+%dd | %d features | window=%d | LSTM%s | target=%s",
        horizon_d, len(_feat_cols), _window, _lstm_units, target_col,
    )
    logger.info("=" * 58)

    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for col in (target_col, abs_col, BASE_LEVEL_COL):
            if col not in split.columns:
                raise KeyError(
                    f"Thiếu cột '{col}' trong tập '{name}'.\nChạy lại 05_integrate.py."
                )
        # Kiểm tra features mới có trong dataset không
        missing = [f for f in _feat_cols if f not in split.columns]
        if missing:
            raise KeyError(
                f"Tập '{name}' thiếu các features: {missing}\n"
                "Chạy lại 05_integrate.py để tạo lại dataset với features mới."
            )

    X_train, y_train, ts_train, base_train, sw_train = create_sequences(
        df_train, _feat_cols, target_col, _window
    )
    # Tăng trọng số sự kiện biến động mạnh (|ΔH| lớn)
    y_delta_raw = df_train.loc[ts_train, target_col].values
    sw_train = sw_train * np.where(
        np.abs(y_delta_raw) >= FLOOD_DELTA_THRESHOLD_M,
        SAMPLE_WEIGHT_FLOOD,
        1.0,
    )
    X_test, y_test, ts_test, base_test, sw_test = create_sequences(
        df_test, _feat_cols, target_col, _window
    )
    X_val, y_val, ts_val, base_val, sw_val = create_sequences(
        df_val, _feat_cols, target_col, _window
    )
    y_val_abs  = df_val.loc[ts_val,  abs_col].values
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
        input_shape=(_window, len(_feat_cols)),
        lstm_units=_lstm_units,
        dropout_rate=_dropout,
        dense_units=_dense_units,
        l2_reg=_l2_reg,
    )
    # [v7] Phase-Aware Loss: Huber + gradient penalty + NSE reward
    # Thay thế Huber đơn thuần — giảm trễ pha, tăng NSE thủy văn
    _loss_fn = build_phase_aware_loss(
        lambda_phase=LAMBDA_PHASE,
        lambda_nse=LAMBDA_NSE,
    )
    logger.info(
        "  [v7] Phase-Aware Loss: Huber + λ_phase=%.2f * grad_err − λ_nse=%.2f * NSE",
        LAMBDA_PHASE, LAMBDA_NSE,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=_loss_fn,
        metrics=["mae"],
    )

    if horizon_d == FORECAST_DAYS[0]:
        model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=_patience, min_delta=MIN_DELTA_ES,
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
            "version": "7.0",
            "predict_delta_h": PREDICT_DELTA_H,
            "window_size": WINDOW_SIZE,
            "window_size_t7d": WINDOW_SIZE_T7D,
            "feature_cols": FEATURE_COLS,
            "feature_cols_t7d": FEATURE_COLS_T7D,
            "feature_count": len(FEATURE_COLS),
            "feature_count_t7d": len(FEATURE_COLS_T7D),
            "ensemble_persistence_weight": ENSEMBLE_PERSISTENCE_WEIGHT,
            "lstm_units": LSTM_UNITS,
            "lstm_units_t7d": LSTM_UNITS_T7D,
            "dropout_rate": DROPOUT_RATE,
            "dropout_rate_t7d": DROPOUT_RATE_T7D,
            # [v7] Phase-Aware Loss
            "lambda_phase": LAMBDA_PHASE,
            "lambda_nse": LAMBDA_NSE,
            "rain_lag_extra": RAIN_LAG_EXTRA,
        }
        _config_path = os.path.join("models", "train_config.json")
        with open(_config_path, "w", encoding="utf-8") as f:
            json.dump(train_cfg, f, ensure_ascii=False, indent=2)

    return metrics_val, metrics_test, model


# ============================================================
# BIỂU ĐỒ KẾT QUẢ (có khoảng tin cậy MC Dropout)
# ============================================================
def plot_results(df_result: pd.DataFrame, history,
                 horizon_d: int, metrics: dict,
                 n_features: int) -> None:
    """
    Vẽ biểu đồ so sánh Dự báo vs Thực tế theo trục valid_time (thời điểm mực nước thực tế)
    trên toàn bộ tập kiểm tra độc lập 2024-2025. Cấu trúc và định dạng giống hệt ảnh mẫu.
    Đồng thời vẽ biểu đồ Loss Curve lưu riêng ra file results/loss_t{horizon_d}d.png.
    """
    df_plot = df_result.copy()

    # ── Đảm bảo cột issue_time và valid_time tồn tại ─────────────────────────
    if "issue_time" not in df_plot.columns:
        df_plot["issue_time"] = df_plot.get(
            "timestamp", df_plot.index
        )
    df_plot["issue_time"] = pd.to_datetime(df_plot["issue_time"])
    if "valid_time" not in df_plot.columns:
        df_plot["valid_time"] = (
            df_plot["issue_time"] + pd.to_timedelta(horizon_d, unit="D")
        )
    df_plot["valid_time"] = pd.to_datetime(df_plot["valid_time"])
    df_plot = df_plot.sort_values("valid_time").reset_index(drop=True)

    # ── 1. VẼ BIỂU ĐỒ DỰ BÁO VẬN HÀNH (TRỤC VALID_TIME) ───────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    t = df_plot["valid_time"]

    # Vùng khoảng tin cậy 95% (MC Dropout)
    if "ci95_lower" in df_plot.columns and "ci95_upper" in df_plot.columns:
        ax.fill_between(t, df_plot["ci95_lower"], df_plot["ci95_upper"],
                        alpha=0.15, color="tomato", label="Khoang tin cay 95% (MC Dropout)",
                        zorder=1)

    # Đường thực tế H(t)
    ax.plot(t, df_plot["actual"],
            label="Thuc te H(t)",
            color="royalblue", linewidth=1.8, zorder=3)

    # Đường dự báo t+d
    ax.plot(t, df_plot["predicted"],
            label=f"Du bao Bi-LSTM t+{horizon_d}d",
            color="tomato", linewidth=1.5, linestyle="--", zorder=4)

    # Đường Naive Persistence để so sánh
    if "persistence" in df_plot.columns:
        ax.plot(t, df_plot["persistence"],
                color="gray", linewidth=0.8, linestyle=":",
                label="Naive H(t) (persistence)", zorder=2, alpha=0.7)

    # Vẽ các ngưỡng vận hành (MNDBT và Ngưỡng xả lũ)
    from config import MNDBT
    ax.axhline(MNDBT, color="darkorange", linestyle="-.", linewidth=1.0, alpha=0.9,
               label=f"MNDBT = {MNDBT} m")
    ax.axhline(46.50, color="crimson", linestyle="-.", linewidth=1.0, alpha=0.9,
               label="Nguong xa lu = 46.5 m")

    # Đánh dấu vùng Bão Yagi (màu vàng nhạt)
    yagi_s = pd.Timestamp("2024-09-07")
    yagi_e = pd.Timestamp("2024-09-15")
    ax.axvspan(yagi_s, yagi_e, alpha=0.12, color="gold", zorder=0,
               label="Bao Yagi 09/2024")

    # Annotate đỉnh lũ Yagi
    yagi_data = df_plot[(df_plot["valid_time"] >= yagi_s) & (df_plot["valid_time"] <= yagi_e)]
    if len(yagi_data) > 0:
        max_yagi_act = yagi_data["actual"].max()
        ax.annotate(
            f"Bao Yagi\n(actual = {max_yagi_act:.2f}m)",
            xy=(yagi_s + pd.Timedelta(days=1.5), max_yagi_act),
            xytext=(yagi_s - pd.Timedelta(days=22), df_plot["actual"].max() - 0.2),
            fontsize=8, color="darkred", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
        )

    # Đánh dấu vùng Bão Matmo 10/2025 (nếu có dữ liệu)
    matmo_s = pd.Timestamp("2025-10-01")
    matmo_e = pd.Timestamp("2025-10-10")
    if (df_plot["valid_time"] >= matmo_s).any():
        ax.axvspan(matmo_s, matmo_e, alpha=0.10, color="lightblue", zorder=0,
                   label="Bao Matmo 10/2025")

    # Định dạng trục X (Thời gian hiển thị tháng/năm)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)

    ax.set_xlabel("Ngay (valid_time)", fontsize=11)
    ax.set_ylabel("Muc nuoc (m)", fontsize=11)

    nse_val = metrics.get("nse")
    rmse_val = metrics.get("rmse")
    mae_val = metrics.get("mae")

    ax.set_title(
        f"Bi-LSTM Du bao muc nuoc ho Nui Coc  |  Chan troi t+{horizon_d}d  "
        f"|  Toan bo tap kiem tra doc lap 2024-2025 ({len(df_plot)} ngay)\n"
        f"RMSE = {rmse_val:.4f} m    MAE = {mae_val:.4f} m    NSE = {nse_val:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    # Thiết lập khoảng giới hạn trục Y
    y_min = df_plot[["actual", "predicted"]].min().min() - 0.3
    y_max = df_plot[["actual", "predicted"]].max().max() + 0.5
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    save_path = f"results/plot_t{horizon_d}d.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  [Biểu đồ] Đã lưu: %s", save_path)

    # ── 2. VẼ BIỂU ĐỒ LOSS CURVE RIÊNG BIỆT ──────────────────────────────────
    if history is not None and hasattr(history, "history") and "loss" in history.history:
        fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
        ax_loss.plot(history.history["loss"], label="Train Loss", color="steelblue", linewidth=1.8)
        if "val_loss" in history.history:
            ax_loss.plot(history.history["val_loss"], label="Val Loss", color="darkorange", linewidth=1.8)
        ax_loss.set_xlabel("Epoch", fontsize=11)
        ax_loss.set_ylabel("Phase-Aware Loss (ΔH scaled)", fontsize=11)
        ax_loss.set_title(f"Quá trình hội tụ hàm tổn thất (Loss Curve) — Bi-LSTM t+{horizon_d}d", fontsize=12, fontweight="bold")
        ax_loss.legend(fontsize=10)
        ax_loss.grid(alpha=0.25)

        plt.tight_layout()
        loss_save_path = f"results/loss_t{horizon_d}d.png"
        plt.savefig(loss_save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  [Biểu đồ Loss] Đã lưu: %s", loss_save_path)


# ============================================================
# BẢNG TỔNG HỢP
# ============================================================
def print_summary_table(all_metrics: dict) -> None:
    """In bang tong hop va luu JSON ket qua.

    Chi bao cao chi so tren tap TEST (kiem dinh doc lap 2024-2025).
    Chi so Val chi hien thi de tham khao kiem tra overfitting.
    """
    print("\n" + "=" * 90)
    print("  BANG TONG HOP — Bi-LSTM (Ho Nui Coc)")
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
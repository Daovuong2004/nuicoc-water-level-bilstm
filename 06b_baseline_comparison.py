"""
Bước 6b: Ablation Study / Baseline Comparison
==============================================
Mục đích:
    So sánh 4 mô hình để chứng minh rằng Bi-LSTM + Attention là kiến trúc
    tốt nhất cho bài toán dự báo mực nước hồ Núi Cốc, Thái Nguyên.

Bốn mô hình được so sánh:
    1. SARIMA        — mô hình thống kê truyền thống (baseline thống kê)
    2. LSTM          — LSTM đơn chiều (baseline học sâu, không BiDir)
    3. Bi-LSTM       — Bi-LSTM không có Attention (baseline loại bỏ Attn)
    4. Bi-LSTM+Attn  — Mô hình đề xuất (đã huấn luyện tại bước 06)

Các chỉ số đánh giá:
    - RMSE  (Root Mean Squared Error)  : đơn vị mét
    - MAE   (Mean Absolute Error)      : đơn vị mét
    - NSE   (Nash-Sutcliffe Efficiency): không thứ nguyên, càng gần 1 càng tốt

Kết quả được lưu tại:
    results/baseline_comparison_nse.png
    results/baseline_comparison_rmse.png
    results/ablation_study.json

Khoảng dự báo: t+1h, t+3h, t+6h, t+12h, t+24h
Cửa sổ đầu vào: 48 giờ
Tập kiểm tra: Lũ Yagi tháng 9/2024
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime

# Ensure standard output and error output use UTF-8 to prevent UnicodeEncodeError on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # Backend không cần GUI — chạy được trên server/headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import (
    Input, Bidirectional, LSTM, Dense,
    Dropout, BatchNormalization, Multiply,
    Permute, Flatten, RepeatVector, Lambda,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import mean_squared_error, mean_absolute_error

# statsmodels cho SARIMA — import có thể lỗi nếu chưa cài
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False
    warnings.warn(
        "Chưa cài statsmodels → Bỏ qua SARIMA. "
        "Cài bằng: pip install statsmodels"
    )

# ============================================================
# CẤU HÌNH LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# CẤU HÌNH SIÊU THAM SỐ
# ============================================================
WINDOW_SIZE    = 30           # Số ngày nhìn lại (input sequence length)
FORECAST_DAYS  = [1, 3, 7, 14, 30]   # Các khoảng dự báo (ngày)
BATCH_SIZE     = 32
MAX_EPOCHS     = 200
PATIENCE_ES    = 15           # EarlyStopping patience
PATIENCE_LR    = 7            # ReduceLROnPlateau patience

# Đặt seed toàn cục để tái lập kết quả
tf.random.set_seed(42)
np.random.seed(42)

# Tạo thư mục đầu ra
os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# FEATURE COLUMNS
# ============================================================
FEATURE_COLS = [
    "rain_1d", "rain_3d", "rain_7d", "rain_14d",
    "temperature", "humidity",
    "water_level_lag1", "water_level_lag3", "water_level_lag7", "water_level_lag14", "water_level_lag30",
    "water_level_roll7", "water_level_roll30", "water_level_std7",
    "month_sin", "month_cos", "season_wet", "season_dry",
    "dH_dt_daily", "Q_out_daily", "Q_out_roll7",
]

TARGET_COL = "water_level_m"   # Cột mực nước thực tế (m)

# Tên hiển thị và màu sắc cho từng mô hình trong biểu đồ
MODEL_NAMES  = ["SARIMA", "LSTM", "Bi-LSTM", "Bi-LSTM+Attn"]
MODEL_COLORS = ["#808080", "steelblue", "orange", "crimson"]


# ============================================================
# I. HÀM TẢI DỮ LIỆU
# ============================================================
def load_dataset_csv(path: str) -> pd.DataFrame:
    """
    Đọc file CSV được lưu bởi df.to_csv() với DatetimeIndex.

    Cột đầu tiên là index timestamp (không có tên cột vì đã là index).
    Hàm tự động parse datetime và đặt tên index là 'timestamp'.

    Parameters
    ----------
    path : str
        Đường dẫn đến file CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame với DatetimeIndex tên 'timestamp'.

    Raises
    ------
    FileNotFoundError
        Nếu file không tồn tại.
    ValueError
        Nếu cột đầu tiên không thể parse thành DatetimeIndex.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file: '{path}'\n"
            "Hãy chạy '05_integrate.py' trước để tạo dữ liệu."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index của '{path}' không phải DatetimeIndex!")
    return df


# ============================================================
# II. HÀM TẠO SEQUENCES CHO LSTM
# ============================================================
def create_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    window_size: int,
) -> tuple:
    """
    Tạo cặp (X, y) dạng chuỗi thời gian từ DataFrame.

    Mỗi mẫu X[i] là cửa sổ window_size giờ gần nhất,
    y[i] là giá trị target tại giờ i.

    Parameters
    ----------
    df : pd.DataFrame
        Dữ liệu đầu vào (đã chuẩn hóa).
    feature_cols : list of str
        Danh sách tên cột features.
    target_col : str
        Tên cột giá trị mục tiêu (ví dụ: 'target_t1h').
    window_size : int
        Kích thước cửa sổ trượt (số bước thời gian nhìn lại).

    Returns
    -------
    tuple
        X  : ndarray shape (n_samples, window_size, n_features)
        y  : ndarray shape (n_samples,)
        ts : DatetimeIndex chứa timestamp tương ứng
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
# III. CHỈ SỐ ĐÁNH GIÁ THỦY VĂN
# ============================================================
def nash_sutcliffe_efficiency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Hệ số hiệu quả Nash-Sutcliffe (NSE).

    Công thức:
        NSE = 1 - Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²

    Thang đánh giá:
        NSE ≥ 0.75  → Tốt       (phù hợp cho báo cáo đồ án)
        NSE ≥ 0.60  → Khá
        NSE < 0.60  → Yếu (cần cải thiện)
        NSE = 1.0   → Hoàn hảo
        NSE = 0.0   → Chỉ tốt bằng dùng giá trị trung bình
        NSE < 0.0   → Tệ hơn giá trị trung bình

    Parameters
    ----------
    y_true : ndarray
        Giá trị thực tế.
    y_pred : ndarray
        Giá trị dự báo.

    Returns
    -------
    float
        Giá trị NSE trong khoảng (-∞, 1].
    """
    numerator   = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan
    return float(1.0 - numerator / denominator)


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """
    Tính RMSE, MAE và NSE cho một tập dự báo.

    Parameters
    ----------
    y_true : ndarray
        Giá trị thực tế.
    y_pred : ndarray
        Giá trị dự báo.
    label : str
        Nhãn hiển thị trong log (ví dụ: 'LSTM t+1h').

    Returns
    -------
    dict
        {'rmse': float, 'mae': float, 'nse': float}
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    nse  = nash_sutcliffe_efficiency(y_true, y_pred)
    logger.info(
        "  [%-20s] RMSE: %.4fm | MAE: %.4fm | NSE: %.4f",
        label, rmse, mae, nse,
    )
    return {"rmse": rmse, "mae": mae, "nse": nse}


# ============================================================
# IV. XÂY DỰNG KIẾN TRÚC MÔ HÌNH
# ============================================================

def build_lstm_unidirectional(input_shape: tuple) -> Model:
    """
    LSTM đơn chiều — Baseline học sâu để so sánh với Bi-LSTM.

    Kiến trúc:
        Input(window_size, n_features)
        → LSTM(128, return_sequences=True)
        → Dropout(0.2) → BatchNormalization
        → LSTM(64, return_sequences=False)
        → Dropout(0.2) → BatchNormalization
        → Dense(32, relu)
        → Dense(1, linear)   ← Dự báo mực nước (m)

    Lý do chọn LSTM đơn chiều làm baseline:
        Đây là kiến trúc phổ biến nhất trong dự báo thủy văn.
        So sánh với Bi-LSTM để đo đạc lợi ích của xử lý 2 chiều.

    Parameters
    ----------
    input_shape : tuple
        (window_size, n_features)

    Returns
    -------
    keras.Model
        Mô hình đã được compile với Adam optimizer, loss=MSE.
    """
    inputs = Input(shape=input_shape, name="input_sequence")

    # Lớp LSTM thứ nhất — trích xuất đặc trưng theo chiều xuôi
    x = LSTM(128, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(0.2, name="dropout_1")(x)
    x = BatchNormalization(name="bn_1")(x)

    # Lớp LSTM thứ hai — tổng hợp thành vector cố định
    x = LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)
    x = BatchNormalization(name="bn_2")(x)

    # Lớp Dense — ánh xạ sang không gian dự báo
    x       = Dense(32, activation="relu", name="dense_1")(x)
    outputs = Dense(1,  activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Unidirectional")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_bilstm_simple(input_shape: tuple) -> Model:
    """
    Bi-LSTM đơn giản — không có Attention (baseline loại bỏ Attention).

    Kiến trúc (giống kiến trúc gốc trong 06_bilstm_model.py):
        Input(window_size, n_features)
        → Bidirectional(LSTM(128, return_sequences=True))
        → Dropout(0.2) → BatchNormalization
        → Bidirectional(LSTM(64, return_sequences=False))
        → Dropout(0.2) → BatchNormalization
        → Dense(32, relu)
        → Dense(1, linear)   ← Dự báo mực nước (m)

    Lý do so sánh:
        Để tách biệt đóng góp của cơ chế Attention so với phần Bi-LSTM cơ bản.
        Nếu Bi-LSTM+Attn >> Bi-LSTM đơn giản → Attention có giá trị.

    Parameters
    ----------
    input_shape : tuple
        (window_size, n_features) — có thể là 12 hoặc 18 features.

    Returns
    -------
    keras.Model
        Mô hình đã được compile với Adam optimizer, loss=MSE.
    """
    inputs = Input(shape=input_shape, name="input_sequence")

    # Lớp BiLSTM thứ nhất — học đặc trưng từ cả 2 chiều thời gian
    x = Bidirectional(
        LSTM(128, return_sequences=True, name="bilstm_1"),
        name="bidirectional_1",
    )(inputs)
    x = Dropout(0.2, name="dropout_1")(x)
    x = BatchNormalization(name="bn_1")(x)

    # Lớp BiLSTM thứ hai — tổng hợp biểu diễn chuỗi
    x = Bidirectional(
        LSTM(64, return_sequences=False, name="bilstm_2"),
        name="bidirectional_2",
    )(x)
    x = Dropout(0.2, name="dropout_2")(x)
    x = BatchNormalization(name="bn_2")(x)

    # Lớp Dense — ánh xạ sang không gian dự báo
    x       = Dense(32, activation="relu", name="dense_1")(x)
    outputs = Dense(1,  activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_Simple")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ============================================================
# V. SARIMA BASELINE
# ============================================================
def run_sarima_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    horizon_d: int,
) -> tuple | None:
    """
    Chạy SARIMA làm baseline thống kê.

    Cấu hình SARIMAX:
        order          = (2, 1, 2)    — AR(2), sai phân bậc 1, MA(2)
        seasonal_order = (1, 1, 1, 7) — chu kỳ tuần 7 ngày

    Chiến lược rolling forecast:
        - Mỗi 7 ngày fit lại mô hình 1 lần để cân bằng giữa độ chính xác
          và tốc độ tính toán (tránh fit lại từng bước).
        - Dự báo horizon_d bước tới từ thời điểm hiện tại.
        - Lấy giá trị tại bước horizon_d làm dự báo.

    Parameters
    ----------
    df_train : pd.DataFrame
        Dữ liệu huấn luyện — SARIMA chỉ dùng cột water_level_m.
    df_test : pd.DataFrame
        Dữ liệu kiểm tra để đánh giá.
    horizon_d : int
        Khoảng dự báo tính bằng ngày (1, 3, 7, 14, 30).

    Returns
    -------
    tuple (y_true_array, y_pred_array) hoặc None nếu lỗi.
    """
    if not SARIMA_AVAILABLE:
        logger.warning("statsmodels chưa cài — bỏ qua SARIMA.")
        return None

    logger.info("  [SARIMA] Đang fit trên tập train (%d mẫu)...", len(df_train))

    try:
        # Lấy chuỗi mực nước từ tập train làm lịch sử ban đầu
        train_series = df_train[TARGET_COL].values.tolist()
        test_series  = df_test[TARGET_COL].values

        n_test   = len(test_series)
        y_true   = []   # Giá trị thực tế tại mỗi bước dự báo
        y_pred   = []   # Giá trị dự báo của SARIMA

        # Refitting mỗi 7 ngày để tránh drift quá lớn và chạy nhanh hơn
        refit_interval = 7
        history        = train_series.copy()

        for i in range(n_test - horizon_d):
            # Fit lại mô hình SARIMA mỗi refit_interval bước
            if i % refit_interval == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sarima_model = SARIMAX(
                        history,
                        order=(2, 1, 2),
                        seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    sarima_fit = sarima_model.fit(disp=False, maxiter=100)

                if i % 60 == 0:
                    logger.info(
                        "  [SARIMA] Bước %d/%d — đã fit lại mô hình.",
                        i, n_test - horizon_d,
                    )

            # Dự báo horizon_d bước tới
            forecast = sarima_fit.forecast(steps=horizon_d)
            pred_val = float(forecast.iloc[-1]) if hasattr(forecast, "iloc") \
                else float(forecast[-1])

            # Ghi nhận dự báo và thực tế tại bước horizon_d
            y_pred.append(pred_val)
            y_true.append(test_series[i + horizon_d])

            # Cập nhật lịch sử với giá trị thực tế (rolling)
            history.append(test_series[i])

        logger.info("  [SARIMA] Hoàn thành — %d cặp (y_true, y_pred).", len(y_true))
        return np.array(y_true), np.array(y_pred)

    except Exception as exc:
        logger.error("  [SARIMA] Lỗi khi chạy: %s — Bỏ qua.", exc)
        return None


# ============================================================
# VI. HUẤN LUYỆN MÔ HÌNH KERAS
# ============================================================
def train_keras_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    horizon_d: int,
) -> object:
    """
    Huấn luyện mô hình Keras với EarlyStopping và ReduceLROnPlateau.

    Callbacks:
        EarlyStopping     : patience=15, khôi phục trọng số tốt nhất
        ReduceLROnPlateau : factor=0.5, patience=7, min_lr=1e-6
        ModelCheckpoint   : lưu model tốt nhất vào models/{model_name}_t{d}d.keras

    Parameters
    ----------
    model : keras.Model
        Mô hình đã được build và compile.
    X_train, y_train : ndarray
        Dữ liệu huấn luyện (sequences đã tạo).
    X_val, y_val : ndarray
        Dữ liệu validation.
    model_name : str
        Tên định danh mô hình (dùng để đặt tên file lưu).
    horizon_d : int
        Khoảng dự báo tính bằng ngày.

    Returns
    -------
    keras.callbacks.History
        Lịch sử huấn luyện (loss, val_loss theo epoch).
    """
    model_path = f"models/{model_name}_t{horizon_d}d.keras"

    callbacks = [
        # Dừng sớm nếu val_loss không cải thiện sau PATIENCE_ES epoch
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_ES,
            restore_best_weights=True,
            verbose=0,
        ),
        # Giảm learning rate khi val_loss plateau
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE_LR,
            min_lr=1e-6,
            verbose=0,
        ),
        # Lưu model checkpoint tốt nhất
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    logger.info(
        "  [%s] Bắt đầu huấn luyện t+%dd — train: %s, val: %s",
        model_name, horizon_d, X_train.shape, X_val.shape,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,   # Tắt verbose để log không quá dài
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val   = float(min(history.history["val_loss"]))
    logger.info(
        "  [%s] Hoàn thành — best epoch: %d | val_loss: %.6f | Đã lưu: %s",
        model_name, best_epoch, best_val, model_path,
    )
    return history


# ============================================================
# VII. VẼ BIỂU ĐỒ SO SÁNH
# ============================================================
def plot_comparison_bar(
    all_results: dict,
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    """
    Vẽ grouped bar chart so sánh 4 mô hình trên từng khoảng dự báo.

    Trục X : Khoảng dự báo (t+1h, t+3h, ..., t+24h)
    Trục Y : Giá trị chỉ số đánh giá (NSE hoặc RMSE)
    4 nhóm : SARIMA (gray), LSTM (steelblue), Bi-LSTM (orange), Bi-LSTM+Attn (crimson)

    Parameters
    ----------
    all_results : dict
        all_results[horizon][model_name] = {'rmse':…, 'mae':…, 'nse':…}
        Nếu mô hình không chạy được, giá trị là None.
    metric_key : str
        Tên chỉ số cần vẽ: 'nse' hoặc 'rmse'.
    ylabel : str
        Nhãn trục Y.
    title : str
        Tiêu đề biểu đồ.
    save_path : str
        Đường dẫn lưu file PNG.
    """
    horizons    = sorted(all_results.keys())
    n_horizons  = len(horizons)
    n_models    = len(MODEL_NAMES)
    bar_width   = 0.18
    x_positions = np.arange(n_horizons)

    fig, ax = plt.subplots(figsize=(12, 6))

    for m_idx, (model_name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        # Thu thập giá trị metric của model này trên tất cả horizons
        values = []
        for h in horizons:
            metrics = all_results[h].get(model_name)
            if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                values.append(metrics[metric_key])
            else:
                values.append(np.nan)   # Mô hình không chạy được

        # Vị trí cụm bar cho model này
        offsets = (m_idx - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offsets,
            values,
            width=bar_width,
            label=model_name,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.88,
        )

        # Ghi giá trị lên đầu mỗi thanh bar
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, color="black",
                )

    # Định dạng trục và tiêu đề
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"t+{h}d" for h in horizons], fontsize=11)
    ax.set_xlabel("Khoảng dự báo", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Thêm đường tham chiếu NSE=0.75 (ngưỡng "Tốt")
    if metric_key == "nse":
        ax.axhline(y=0.75, color="green", linestyle=":", linewidth=1.2,
                   label="NSE=0.75 (Tốt)")
        ax.axhline(y=0.60, color="goldenrod", linestyle=":", linewidth=1.0,
                   label="NSE=0.60 (Khá)")
        ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  [Biểu đồ] Đã lưu: %s", save_path)


def plot_comparison_table(all_results: dict) -> None:
    """
    Vẽ 2 biểu đồ grouped bar chart:
        1. So sánh NSE  → results/baseline_comparison_nse.png
        2. So sánh RMSE → results/baseline_comparison_rmse.png

    Parameters
    ----------
    all_results : dict
        all_results[horizon][model_name] = metrics_dict hoặc None.
    """
    logger.info("\n[Vẽ biểu đồ] Tạo biểu đồ so sánh baseline...")

    # Biểu đồ 1: So sánh NSE (Nash-Sutcliffe Efficiency)
    plot_comparison_bar(
        all_results=all_results,
        metric_key="nse",
        ylabel="NSE (Nash-Sutcliffe Efficiency)",
        title=(
            "Ablation Study: So sánh NSE giữa 4 mô hình\n"
            "Hồ Núi Cốc — Tập kiểm tra Lũ Yagi 9/2024"
        ),
        save_path="results/baseline_comparison_nse.png",
    )

    # Biểu đồ 2: So sánh RMSE (Root Mean Squared Error)
    plot_comparison_bar(
        all_results=all_results,
        metric_key="rmse",
        ylabel="RMSE (m)",
        title=(
            "Ablation Study: So sánh RMSE giữa 4 mô hình\n"
            "Hồ Núi Cốc — Tập kiểm tra Lũ Yagi 9/2024"
        ),
        save_path="results/baseline_comparison_rmse.png",
    )


# ============================================================
# VIII. LƯU VÀ IN KẾT QUẢ
# ============================================================
def save_results_json(all_results: dict) -> None:
    """
    Lưu kết quả ablation study vào JSON và in bảng ASCII.

    File JSON: results/ablation_study.json
    Format:
        {
          "t+1h": {
            "SARIMA":       {"rmse": …, "mae": …, "nse": …},
            "LSTM":         {"rmse": …, "mae": …, "nse": …},
            "Bi-LSTM":      {"rmse": …, "mae": …, "nse": …},
            "Bi-LSTM+Attn": {"rmse": …, "mae": …, "nse": …}
          },
          …
        }

    Parameters
    ----------
    all_results : dict
        all_results[horizon][model_name] = metrics_dict hoặc None.
    """
    # Chuyển đổi None thành dict rỗng và làm sạch NaN để serialize được JSON
    json_ready = {}
    for h, model_dict in all_results.items():
        key = f"t+{h}d"
        json_ready[key] = {}
        for model_name, metrics in model_dict.items():
            if metrics is None:
                json_ready[key][model_name] = None
            else:
                # Thay NaN bằng None (JSON-serializable)
                json_ready[key][model_name] = {
                    k: (None if (v is not None and np.isnan(v)) else v)
                    for k, v in metrics.items()
                }

    # Lưu JSON
    json_path = "results/ablation_study.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu: %s", json_path)

    # In bảng ASCII tổng hợp
    _print_ascii_table(all_results)


def _print_ascii_table(all_results: dict) -> None:
    """
    In bảng ASCII đẹp tổng hợp kết quả tất cả mô hình × khoảng dự báo.

    Ví dụ đầu ra:
    ┌──────────┬─────────────────────────────────────────────────────────────────┐
    │  Khoảng  │  SARIMA        │  LSTM          │  Bi-LSTM       │ Bi-LSTM+Attn│
    │          │ RMSE  MAE  NSE │ RMSE  MAE  NSE │ RMSE  MAE  NSE │RMSE MAE NSE│
    ├──────────┼────────────────┼────────────────┼────────────────┼────────────┤
    │  t+ 1d   │ 0.xxx 0.xx x.xx│ ...
    └──────────┴─────────────────────────────────────────────────────────────────┘
    """
    horizons   = sorted(all_results.keys())
    col_w      = 16   # Độ rộng cột mỗi mô hình
    label_w    = 8    # Độ rộng cột khoảng dự báo

    sep = "=" * (label_w + 2 + len(MODEL_NAMES) * (col_w + 3))
    print("\n" + sep)
    print("  BẢNG TỔNG HỢP ABLATION STUDY — HỒ NÚI CỐC")
    print("  Chỉ số: RMSE (m) / MAE (m) / NSE")
    print(sep)

    # Header dòng 1: tên mô hình
    header = f"{'Khoảng':>{label_w}} |"
    for name in MODEL_NAMES:
        header += f" {name:^{col_w}} |"
    print(header)

    # Header dòng 2: tên chỉ số
    sub_hdr = f"{' ':>{label_w}} |"
    for _ in MODEL_NAMES:
        sub_hdr += f" {'RMSE':>5} {'MAE':>5} {'NSE':>4} |"
    print(sub_hdr)
    print("-" * len(sep))

    # Dữ liệu từng khoảng dự báo
    for h in horizons:
        row = f"  t+{h:>2}d   |"
        for model_name in MODEL_NAMES:
            metrics = all_results[h].get(model_name)
            if metrics is None:
                row += f" {'N/A':>5} {'N/A':>5} {'N/A':>4} |"
            else:
                rmse = metrics.get("rmse", np.nan)
                mae  = metrics.get("mae",  np.nan)
                nse  = metrics.get("nse",  np.nan)
                rmse_s = f"{rmse:.3f}" if not np.isnan(rmse) else "N/A"
                mae_s  = f"{mae:.3f}"  if not np.isnan(mae)  else "N/A"
                nse_s  = f"{nse:.3f}"  if not np.isnan(nse)  else "N/A"
                row   += f" {rmse_s:>5} {mae_s:>5} {nse_s:>4} |"
        print(row)

    print(sep)

    # In mô hình tốt nhất (NSE cao nhất) cho từng khoảng
    print("\n  ★ Mô hình tốt nhất theo NSE:")
    for h in horizons:
        best_model = None
        best_nse   = -np.inf
        for model_name in MODEL_NAMES:
            metrics = all_results[h].get(model_name)
            if metrics is not None:
                nse = metrics.get("nse", -np.inf)
                if nse is not None and not np.isnan(nse) and nse > best_nse:
                    best_nse   = nse
                    best_model = model_name
        tag = "✓ Tốt" if best_nse >= 0.75 else ("Khá" if best_nse >= 0.60 else "Yếu")
        print(f"    t+{h:>2}h → {best_model or 'N/A':<15} NSE={best_nse:.4f}  [{tag}]")
    print()


# ============================================================
# IX. HÀM MAIN — ĐIỀU PHỐI TOÀN BỘ ABLATION STUDY
# ============================================================
def main() -> None:
    """
    Hàm chính điều phối toàn bộ ablation study.

    Luồng thực thi:
        1. Load df_train, df_val, df_test từ data/final/
        2. Kiểm tra feature set (18 cols hoặc fallback 12 cols)
        3. Loop qua FORECAST_HOURS [1, 3, 6, 12, 24]:
            a. SARIMA rolling forecast
            b. LSTM đơn chiều (train từ đầu)
            c. Bi-LSTM đơn giản (train từ đầu)
            d. Load Bi-LSTM+Attention từ models/bilstm_t{h}h.keras
        4. Tổng hợp kết quả → dict all_results
        5. Vẽ biểu đồ so sánh → plot_comparison_table()
        6. Lưu JSON + in bảng → save_results_json()
    """
    logger.info("=" * 65)
    logger.info("  BƯỚC 6b: ABLATION STUDY / BASELINE COMPARISON")
    logger.info("  Thời gian: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 65)

    # ── 1. Load dữ liệu ────────────────────────────────────────
    logger.info("\n[Load] Đọc bộ dữ liệu từ data/final/ ...")
    df_train = load_dataset_csv("data/final/dataset_train.csv")
    df_val   = load_dataset_csv("data/final/dataset_val.csv")
    df_test  = load_dataset_csv("data/final/dataset_test.csv")

    logger.info(
        "  Train : %d bản ghi (%s → %s)",
        len(df_train),
        df_train.index.min().date(), df_train.index.max().date(),
    )
    logger.info(
        "  Val   : %d bản ghi (%s → %s)",
        len(df_val),
        df_val.index.min().date(), df_val.index.max().date(),
    )
    logger.info(
        "  Test  : %d bản ghi (%s → %s)",
        len(df_test),
        df_test.index.min().date(), df_test.index.max().date(),
    )

    # ── 2. Kiểm tra và chọn feature set ───────────────────────
    missing_q = [c for c in FEATURE_COLS if c not in df_train.columns]

    if missing_q:
        raise KeyError(
            f"Thiếu {len(missing_q)} cột đặc trưng trong dataset: {missing_q}\n"
            "Hãy chạy lại '05_integrate.py' để tạo đúng bộ dữ liệu."
        )
    else:
        logger.info("✓ Đủ 21 features ngày (v3.0).")
        feature_cols = FEATURE_COLS

    n_features = len(feature_cols)
    logger.info("  Số features sử dụng: %d", n_features)

    # Chuẩn bị sequences cho validation (dùng chung cho mọi baseline)
    # Sequences cho test sẽ tạo bên trong loop theo từng target_col

    # ── 3. Loop qua từng khoảng dự báo ────────────────────────
    # all_results[horizon_d][model_name] = {'rmse':…, 'mae':…, 'nse':…} | None
    all_results = {h: {} for h in FORECAST_DAYS}

    for h in FORECAST_DAYS:
        target_col = f"target_t{h}d"
        logger.info("\n%s", "─" * 60)
        logger.info("  DỰ BÁO t+%dd", h)
        logger.info("─" * 60)

        # Kiểm tra cột target tồn tại
        for split_name, split_df in [
            ("train", df_train), ("val", df_val), ("test", df_test)
        ]:
            if target_col not in split_df.columns:
                raise KeyError(
                    f"Cột '{target_col}' không có trong tập '{split_name}'.\n"
                    f"Các target hiện có: "
                    f"{[c for c in split_df.columns if c.startswith('target')]}"
                )

        # Tạo sequences cho Keras models
        X_train, y_train, _        = create_sequences(
            df_train, feature_cols, target_col, WINDOW_SIZE
        )
        X_val, y_val, _            = create_sequences(
            df_val,   feature_cols, target_col, WINDOW_SIZE
        )
        X_test, y_test, ts_test    = create_sequences(
            df_test,  feature_cols, target_col, WINDOW_SIZE
        )

        # ── 3a. SARIMA ──────────────────────────────────────────
        logger.info("\n  → [1/4] SARIMA baseline...")
        sarima_result = run_sarima_baseline(df_train, df_test, h)
        if sarima_result is not None:
            y_true_s, y_pred_s = sarima_result
            all_results[h]["SARIMA"] = evaluate_metrics(
                y_true_s, y_pred_s, label=f"SARIMA t+{h}d"
            )
        else:
            logger.warning("  [SARIMA] Bỏ qua t+%dd.", h)
            all_results[h]["SARIMA"] = None

        # ── 3b. LSTM đơn chiều ──────────────────────────────────
        logger.info("\n  → [2/4] LSTM đơn chiều...")
        lstm_model = build_lstm_unidirectional(
            input_shape=(WINDOW_SIZE, n_features)
        )
        train_keras_model(
            model=lstm_model,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            model_name="lstm_uni",
            horizon_d=h,
        )
        y_pred_lstm = lstm_model.predict(X_test, verbose=0).flatten()
        all_results[h]["LSTM"] = evaluate_metrics(
            y_test, y_pred_lstm, label=f"LSTM t+{h}d"
        )

        # ── 3c. Bi-LSTM đơn giản (không Attention) ─────────────
        logger.info("\n  → [3/4] Bi-LSTM đơn giản (không Attention)...")
        bilstm_model = build_bilstm_simple(
            input_shape=(WINDOW_SIZE, n_features)
        )
        train_keras_model(
            model=bilstm_model,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            model_name="bilstm_simple",
            horizon_d=h,
        )
        y_pred_bilstm = bilstm_model.predict(X_test, verbose=0).flatten()
        all_results[h]["Bi-LSTM"] = evaluate_metrics(
            y_test, y_pred_bilstm, label=f"Bi-LSTM t+{h}d"
        )

        # ── 3d. Bi-LSTM + Attention (load từ bước 06) ──────────
        attn_model_path = f"models/bilstm_t{h}d.keras"
        logger.info("\n  → [4/4] Bi-LSTM+Attention — load từ: %s", attn_model_path)

        if not os.path.exists(attn_model_path):
            logger.warning(
                "  CẢNH BÁO: Chưa tìm thấy '%s'.\n"
                "  → Hãy chạy '06_bilstm_model.py' trước để huấn luyện.\n"
                "  → Bỏ qua Bi-LSTM+Attn cho t+%dd.",
                attn_model_path, h,
            )
            all_results[h]["Bi-LSTM+Attn"] = None
        else:
            try:
                attn_model  = load_model(attn_model_path)
                y_pred_attn = attn_model.predict(X_test, verbose=0).flatten()
                all_results[h]["Bi-LSTM+Attn"] = evaluate_metrics(
                    y_test, y_pred_attn, label=f"Bi-LSTM+Attn t+{h}d"
                )
            except Exception as exc:
                logger.error(
                    "  Lỗi khi load/predict '%s': %s — Bỏ qua.",
                    attn_model_path, exc,
                )
                all_results[h]["Bi-LSTM+Attn"] = None

    # ── 4. Tổng hợp và xuất kết quả ───────────────────────────
    logger.info("\n%s", "=" * 65)
    logger.info("  TỔNG HỢP KẾT QUẢ ABLATION STUDY")
    logger.info("=" * 65)

    plot_comparison_table(all_results)
    save_results_json(all_results)

    logger.info(
        "\n✓ Hoàn thành Ablation Study!\n"
        "  Biểu đồ NSE  : results/baseline_comparison_nse.png\n"
        "  Biểu đồ RMSE : results/baseline_comparison_rmse.png\n"
        "  Dữ liệu JSON  : results/ablation_study.json"
    )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

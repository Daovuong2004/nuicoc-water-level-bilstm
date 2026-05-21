"""
run_all.py — Chạy toàn bộ pipeline từ đầu đến cuối
=====================================================
Thứ tự thực thi:
  Bước 1 : Thu thập dữ liệu khí tượng NASA POWER (có retry + checkpoint)
  Bước 2 : Xử lý GEE Sentinel-2 → mực nước qua đường cong A-H
  Bước 3+4: Trích xuất sự kiện lũ + vận hành cửa xả từ báo chí
  Bước 5 : Tích hợp dữ liệu (Kalman Filter, Q_out, chuẩn hóa, chia split)
  Bước 6 : Huấn luyện Bi-LSTM + Self-Attention + MC Dropout + SHAP
  Bước 6b: Ablation Study — so sánh SARIMA / LSTM / Bi-LSTM / Bi-LSTM+Attn
  Bước 7 : Phân tích Q_out và phát hiện xả đột ngột (tùy chọn)
  Bước 8 : Khởi động FastAPI Inference Server (tùy chọn)

Cài đặt phụ thuộc:
  pip install pandas numpy scikit-learn tensorflow keras requests
              matplotlib scipy joblib statsmodels fastapi uvicorn shap

Ghi chú GEE:
  - Bước 2 cần tài khoản Google Earth Engine
  - Nếu chưa có, đặt file CSV tại data/raw/gee_export.csv để chạy offline
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# CHẠY TỪNG BƯỚC
# ============================================================
def run_step(script: str, step_name: str,
             optional: bool = False) -> bool:
    """
    Chạy một script Python và theo dõi kết quả.

    Parameters
    ----------
    script : str
        Tên file script cần chạy.
    step_name : str
        Tên bước hiển thị trong log.
    optional : bool
        Nếu True: bước thất bại sẽ in cảnh báo và tiếp tục (không dừng pipeline).

    Returns
    -------
    bool
        True nếu thành công, False nếu thất bại.
    """
    if not os.path.exists(script):
        logger.warning("Bỏ qua '%s': file không tồn tại.", script)
        return False

    print(f"\n{'='*62}")
    print(f"  CHẠY: {step_name}")
    print(f"  Script: {script}")
    print(f"  Thời gian: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*62}")

    t0     = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        if optional:
            logger.warning(
                "✗ [Tùy chọn] %s thất bại sau %.1fs — tiếp tục pipeline.",
                step_name, elapsed,
            )
            return False
        else:
            logger.error(
                "✗ LỖI tại bước bắt buộc: %s (%.1fs)", step_name, elapsed
            )
            logger.error("  Kiểm tra lại script '%s' và dừng pipeline.", script)
            sys.exit(1)
    else:
        logger.info("✓ Hoàn thành: %s (%.1fs)", step_name, elapsed)
        return True


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 62)
    print("  PIPELINE DỰ BÁO MỰC NƯỚC HỒ NÚI CỐC")
    print("  Mô hình: Bi-LSTM + Self-Attention (v3.0)")
    print(f"  Bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # Tạo thư mục cần thiết
    for d in ["data/raw", "data/raw/checkpoints", "data/final",
              "models", "results"]:
        os.makedirs(d, exist_ok=True)

    # ── Danh sách các bước ────────────────────────────────────
    # Format: (script, tên bước, optional)
    # optional=False → pipeline dừng nếu bước này thất bại
    # optional=True  → cảnh báo và tiếp tục nếu thất bại
    pipeline_steps = [
        # --- Bắt buộc ---
        ("01_nasa_power.py",
         "Bước 1: Thu thập dữ liệu khí tượng NASA POWER",
         False),

        ("02_gee_sentinel2.py",
         "Bước 2: Xử lý ảnh vệ tinh GEE Sentinel-2",
         False),

        ("03_04_cua_xa_bao_chi.py",
         "Bước 3+4: Trích xuất sự kiện lũ và cửa xả từ báo chí",
         False),

        ("05_integrate.py",
         "Bước 5: Tích hợp dữ liệu (Kalman + Q_out + Normalize)",
         False),

        ("06_bilstm_model.py",
         "Bước 6: Huấn luyện Bi-LSTM + Self-Attention + MC Dropout + SHAP",
         False),

        # --- Tùy chọn (chạy thêm nếu có thể) ---
        ("06b_baseline_comparison.py",
         "Bước 6b: Ablation Study — SARIMA vs LSTM vs Bi-LSTM vs Bi-LSTM+Attn",
         True),

        ("07_infer_qout_1.py",
         "Bước 7: Phân tích Q_out và phát hiện xả đột ngột (visualize)",
         True),
    ]

    # Chạy từng bước
    results_summary = {}
    for script, name, optional in pipeline_steps:
        ok = run_step(script, name, optional=optional)
        results_summary[name] = "✓ OK" if ok else ("⚠ Bỏ qua" if optional else "✗ Lỗi")

    # ── Tóm tắt kết quả ───────────────────────────────────────
    print(f"\n{'='*62}")
    print("  TỔNG KẾT PIPELINE")
    print(f"  Hoàn thành lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}")
    for name, status in results_summary.items():
        # Rút gọn tên cho đẹp
        short = name.split(":")[0].strip()
        print(f"  {status}  {short}")

    print(f"\n{'='*62}")
    print("  OUTPUT:")
    print("    models/bilstm_t*.keras       ← Mô hình Bi-LSTM+Attention")
    print("    models/feature_scaler.pkl    ← Scaler chuẩn hóa")
    print("    results/plot_t*.png          ← Biểu đồ dự báo + CI95%")
    print("    results/shap_importance_*.png← SHAP Feature Importance")
    print("    results/metrics_summary.json ← RMSE / MAE / NSE")
    print("    results/ablation_study.json  ← So sánh 4 mô hình")
    print(f"{'='*62}")
    print()
    print("  BƯỚC TIẾP THEO:")
    print("    Khởi động API: uvicorn 08_api_serve:app --reload --port 8000")
    print("    Swagger UI   : http://localhost:8000/docs")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()

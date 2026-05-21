# ==============================================================
# NOTEBOOK: Trích xuất mực nước hồ Núi Cốc từ GEE Sentinel-2
# Phiên bản: 4.0 — Dùng Export.table.toDrive() tránh quota lỗi
# Môi trường: Google Colab
# ==============================================================
#
# TẠI SAO PHIÊN BẢN CŨ BỊ LỖI?
#   Lỗi "Too many concurrent aggregations":
#     Hàm cũ gọi reduceRegion() 3 lần / ảnh × 60 ảnh = 180 tác vụ song song
#     → Vượt giới hạn GEE free tier → HTTP 429 → thất bại toàn bộ.
#
#   Giải pháp: Export.table.toDrive()
#     GEE xử lý tất cả ảnh BÊN TRONG server của họ (asynchronous batch),
#     không gửi từng kết quả về Python → không bị giới hạn concurrent.
#     Kết quả được ghi thẳng vào Google Drive khi xong.
#
# LUỒNG THỰC THI:
#   Cell 1: Cài đặt + Mount Drive
#   Cell 2: Xác thực GEE
#   Cell 3: Cấu hình (ngưỡng mây, NDWI, A-H)
#   Cell 4: Định nghĩa hàm GEE (đơn giản hóa: 1 reduceRegion/ảnh)
#   Cell 5: Khởi động Export Task (không blocking, chạy nền)
#   Cell 6: Theo dõi tiến độ Export Task
#   Cell 7: Đọc CSV đã export + áp đường cong A-H + visualize
#   Cell 8: Lưu file cuối về Drive sẵn sàng cho pipeline
# ==============================================================


# %% ─────────────────────────────────────────────────────────────
# CELL 1: Cài đặt & Mount Drive
# ─────────────────────────────────────────────────────────────────
# Chạy cell này ĐẦU TIÊN và chờ Drive mount xong trước khi tiếp tục.

# Bỏ comment nếu cần cài thêm:
# !pip install earthengine-api scipy --quiet

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── ĐƯỜNG DẪN — ĐỔI NẾU CẦN ──────────────────────────────────
DRIVE_ROOT      = "/content/drive/MyDrive/DATN"
DRIVE_DATA_DIR  = f"{DRIVE_ROOT}/data/raw"
CHECKPOINT_DIR  = f"{DRIVE_ROOT}/data/raw/checkpoints_gee"
OUTPUT_CSV      = f"{DRIVE_DATA_DIR}/gee_water_level.csv"

# FIX LỖI CŨ: Tạo thư mục NGAY TẠI ĐÂY, không phụ thuộc vào Drive được
# mount hoàn toàn hay chưa. Đảm bảo tồn tại trước mọi thao tác ghi file.
for _d in [DRIVE_DATA_DIR, CHECKPOINT_DIR, f"{DRIVE_ROOT}/results"]:
    os.makedirs(_d, exist_ok=True)
    assert os.path.isdir(_d), f"KHÔNG thể tạo thư mục: {_d}"

print("=" * 55)
print("✓ Google Drive đã mount và thư mục đã tạo.")
print(f"  Drive root    : {DRIVE_ROOT}")
print(f"  Checkpoint    : {CHECKPOINT_DIR}")
print(f"  Output CSV    : {OUTPUT_CSV}")
print("=" * 55)


# %% ─────────────────────────────────────────────────────────────
# CELL 2: Xác thực GEE
# ─────────────────────────────────────────────────────────────────
import ee

# ── ĐỔI PROJECT ID TẠI ĐÂY ────────────────────────────────────
GEE_PROJECT = "datn-495501"   # ← Project ID Google Cloud của bạn

try:
    ee.Initialize(project=GEE_PROJECT)
    _ = ee.Number(1).getInfo()   # Kiểm tra kết nối thực sự
    print(f"✓ GEE đã khởi tạo (project: {GEE_PROJECT})")
except Exception:
    print("→ Cần xác thực, đang mở trình duyệt...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print(f"✓ GEE đã xác thực và khởi tạo (project: {GEE_PROJECT})")


# %% ─────────────────────────────────────────────────────────────
# CELL 3: Cấu hình
# ─────────────────────────────────────────────────────────────────

# ── PHẠM VI THỜI GIAN ─────────────────────────────────────────
START_YEAR = 2017   # Sentinel-2 SR Level-2A có từ 2017
END_YEAR   = 2025

# ── LỌC MÂY ───────────────────────────────────────────────────
# Dùng 80% để tối đa số ảnh. Cloud mask QA60 (trong Cell 4)
# sẽ loại từng pixel mây trước khi tính NDWI → kết quả vẫn chính xác.
MAX_CLOUD_PCT = 80

# ── NGƯỠNG NDWI ───────────────────────────────────────────────
NDWI_THRESHOLD = 0.0   # McFeeters (1996): NDWI > 0 → mặt nước

# ── BOUNDING BOX HỒ NÚI CỐC (WGS84) ──────────────────────────
LAKE_BOUNDS = {
    "lon_min": 105.68, "lat_min": 21.64,
    "lon_max": 105.78, "lat_max": 21.73,
}
REGION = ee.Geometry.Rectangle([
    LAKE_BOUNDS["lon_min"], LAKE_BOUNDS["lat_min"],
    LAKE_BOUNDS["lon_max"], LAKE_BOUNDS["lat_max"],
])

# ── THƯ MỤC GDRIVE NHẬN FILE EXPORT TỪ GEE ───────────────────
# GEE sẽ ghi file CSV vào đây khi Export Task hoàn thành.
# Đây là tên thư mục BÊN TRONG Google Drive (không phải đường dẫn đầy đủ).
GEE_EXPORT_FOLDER = "DATN_GEE_Export"   # Tự động tạo trong Drive

# ── ĐƯỜNG CONG A-H (Diện tích ha → Mực nước m) ────────────────
AH_CURVE = [
    ( 200, 36.00),
    ( 500, 38.00),
    ( 900, 40.00),
    (1400, 42.00),
    (2000, 44.00),
    (2500, 46.20),   # MNDBT
    (2700, 46.50),
    (2900, 46.90),
    (3050, 47.20),
    (3150, 47.50),
    (3200, 47.80),
    (3500, 48.25),
]
_areas  = [p[0] for p in AH_CURVE]
_levels = [p[1] for p in AH_CURVE]
ah_to_level = interp1d(_areas, _levels, kind="linear",
                        bounds_error=False,
                        fill_value=(_levels[0], _levels[-1]))

print("✓ Cấu hình hoàn tất.")
print(f"  Giai đoạn  : {START_YEAR} – {END_YEAR}")
print(f"  Ngưỡng mây : < {MAX_CLOUD_PCT}%  (QA60 mask)")
print(f"  NDWI > {NDWI_THRESHOLD}")
print(f"  Export → Drive folder: '{GEE_EXPORT_FOLDER}'")


# %% ─────────────────────────────────────────────────────────────
# CELL 4: Định nghĩa hàm GEE (đơn giản hóa: 1 reduceRegion/ảnh)
# ─────────────────────────────────────────────────────────────────
# FIX CHÍNH: Phiên bản cũ gọi 3 reduceRegion() / ảnh (water_area,
# total_pixels, valid_pixels) → với 60 ảnh = 180 aggregations song song
# → vượt quota GEE → "Too many concurrent aggregations".
#
# Phiên bản mới: CHỈ 1 reduceRegion() / ảnh.
# Thông tin % mây trong ROI sẽ tính từ cloud_scene (metadata)
# thay vì đếm pixel — đủ cho mục đích phân loại chất lượng.

def mask_clouds_qa60(image):
    """
    Loại bỏ pixel mây (bit 10) và cirrus (bit 11) từ band QA60.

    Phép AND bitwise với giá trị tại bit cần kiểm tra:
      Bit 10 = 1024 → mây dày (opaque clouds)
      Bit 11 = 2048 → mây mỏng cao tầng (cirrus)
    Giữ pixel khi CẢ HAI bit = 0 (không bị che).
    Scale giá trị × 0.0001 (Sentinel-2 SR lưu × 10000).
    """
    qa = image.select("QA60")
    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
          .And(qa.bitwiseAnd(1 << 11).eq(0))
    )
    return image.updateMask(mask).divide(10000)


def compute_water_area(image):
    """
    Tính diện tích mặt nước (ha) cho một ảnh Sentinel-2.

    Đơn giản hóa so với phiên bản cũ: CHỈ 1 reduceRegion() thay vì 3.
    Bỏ tính cloud_roi_pct từ server (tốn 2 reduceRegion thêm).
    Thay bằng cloud_scene (metadata sẵn có, 0 chi phí tính toán).

    Kết quả:
      date          : ngày chụp ảnh (YYYY-MM-dd)
      water_area_ha : diện tích mặt nước sau QA60 mask (ha)
      cloud_scene   : % mây toàn cảnh từ metadata Sentinel-2
    """
    # Bước 1: Áp QA60 cloud mask
    img_clean = mask_clouds_qa60(image)

    # Bước 2: NDWI = (Green B3 - NIR B8) / (Green B3 + NIR B8)
    ndwi = img_clean.normalizedDifference(["B3", "B8"]).rename("NDWI")

    # Bước 3: Phân loại pixel mặt nước
    water = ndwi.gt(NDWI_THRESHOLD)

    # Bước 4: Tính diện tích (1 reduceRegion duy nhất)
    # pixel_area (m²) × water_mask → tổng m² mặt nước → ÷ 10000 → ha
    area_image = water.multiply(ee.Image.pixelArea()).divide(10000)

    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=REGION,
        scale=10,            # Độ phân giải Sentinel-2 Band 3, 8 = 10m
        maxPixels=1e13,      # Đặt lớn để tránh lỗi "too many pixels"
        bestEffort=True,     # Tự tăng scale nếu vượt maxPixels
    )

    return ee.Feature(None, {
        "date":          ee.Date(image.get("system:time_start")).format("YYYY-MM-dd"),
        "water_area_ha": stats.get("NDWI"),
        "cloud_scene":   image.get("CLOUDY_PIXEL_PERCENTAGE"),
        "img_id":        image.id(),
    })


print("✓ Hàm GEE đã định nghĩa (1 reduceRegion/ảnh — tối ưu quota).")


# %% ─────────────────────────────────────────────────────────────
# CELL 5: Khởi động Export Tasks (không blocking — chạy nền GEE)
# ─────────────────────────────────────────────────────────────────
# ĐÂY LÀ GIẢI PHÁP CHÍNH cho lỗi "Too many concurrent aggregations".
#
# Export.table.toDrive() hoạt động như thế nào:
#   1. Python gửi "mô tả công việc" lên GEE server → task.start()
#   2. GEE xử lý hoàn toàn bên trong hệ thống của họ (không blocking)
#   3. Khi xong, GEE tự ghi file CSV vào Google Drive
#   4. Python có thể monitor trạng thái qua task.status()
#
# Ưu điểm:
#   - Không bị giới hạn concurrent aggregations (GEE quản lý hàng đợi)
#   - Colab có thể tắt trong khi task đang chạy — dữ liệu vẫn được lưu
#   - Xử lý toàn bộ 2017–2025 trong 1 task duy nhất hoặc nhiều task nhỏ

def start_export_task(year: int, description: str = None) -> object:
    """
    Tạo và khởi động 1 GEE Export Task cho một năm.

    Kết quả: file CSV tên '{description}.csv' xuất hiện trong
    Google Drive → Folder '{GEE_EXPORT_FOLDER}'.

    Parameters
    ----------
    year : int
        Năm cần export.
    description : str, optional
        Tên task (hiển thị trong GEE Task Manager).

    Returns
    -------
    ee.batch.Task
        Task object để theo dõi trạng thái.
    """
    if description is None:
        description = f"nuicoc_water_{year}"

    # Lọc ảnh Sentinel-2 cho năm này
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(REGION)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .select(["B3", "B8", "QA60"])
    )

    # Ánh xạ hàm tính diện tích → FeatureCollection
    fc = ee.FeatureCollection(s2.map(compute_water_area))

    # Tạo Export Task
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=GEE_EXPORT_FOLDER,      # Thư mục trong Drive (tự tạo nếu chưa có)
        fileNamePrefix=description,    # Tên file: nuicoc_water_2020.csv
        fileFormat="CSV",
        selectors=["date", "water_area_ha", "cloud_scene", "img_id"],
    )

    task.start()
    return task


# Khởi động tất cả tasks (2017–2025) — mỗi năm 1 task riêng
print("=" * 55)
print("  KHỞI ĐỘNG GEE EXPORT TASKS")
print(f"  Giai đoạn: {START_YEAR} – {END_YEAR}")
print(f"  Folder  : Google Drive → '{GEE_EXPORT_FOLDER}'")
print("=" * 55)

all_tasks = {}
for yr in range(START_YEAR, END_YEAR + 1):
    task_name = f"nuicoc_water_{yr}"
    try:
        task = start_export_task(yr, description=task_name)
        all_tasks[yr] = task
        print(f"  ✓ [{yr}] Task đã khởi động: {task_name}")
    except Exception as exc:
        print(f"  ✗ [{yr}] Không thể tạo task: {exc}")
        all_tasks[yr] = None

# Lưu danh sách task ID để theo dõi sau
task_ids = {}
for yr, task in all_tasks.items():
    if task is not None:
        try:
            task_ids[yr] = task.id
        except Exception:
            task_ids[yr] = "unknown"

with open(f"{DRIVE_ROOT}/gee_task_ids.json", "w") as f:
    json.dump(task_ids, f, indent=2)

print(f"\n✓ Đã lưu task IDs: {DRIVE_ROOT}/gee_task_ids.json")
print()
print("THEO DÕI TIẾN ĐỘ tại:")
print("  https://code.earthengine.google.com/tasks")
print()
print("⏳ Thời gian ước tính: 10–30 phút / năm tùy tải server GEE.")
print("   Bạn có thể CHẠY CELL TIẾP THEO để theo dõi, hoặc đợi.")


# %% ─────────────────────────────────────────────────────────────
# CELL 6: Theo dõi tiến độ Export Tasks
# ─────────────────────────────────────────────────────────────────
# Chạy cell này để xem trạng thái hiện tại của các tasks.
# Chạy lại nhiều lần cho đến khi tất cả "COMPLETED".
# Trạng thái: READY → RUNNING → COMPLETED (hoặc FAILED)

def check_all_tasks(tasks: dict) -> dict:
    """Kiểm tra trạng thái tất cả tasks, in bảng tóm tắt."""
    print(f"\n{'─'*55}")
    print(f"  {'Năm':>6} | {'Trạng thái':>12} | {'Tiến độ':>10} | {'Ghi chú'}")
    print(f"{'─'*55}")

    summary = {"COMPLETED": 0, "RUNNING": 0, "READY": 0, "FAILED": 0}

    for yr in sorted(tasks.keys()):
        task = tasks[yr]
        if task is None:
            print(f"  {yr:>6} | {'SKIP':>12} | {'–':>10} | Không tạo được task")
            continue

        try:
            status = task.status()
            state  = status.get("state", "UNKNOWN")
            pct    = status.get("progress", 0) * 100
            msg    = status.get("error_message", "")[:40]

            icon = {"COMPLETED": "✓", "RUNNING": "⟳",
                    "READY": "…", "FAILED": "✗"}.get(state, "?")
            print(f"  {yr:>6} | {icon} {state:>10} | {pct:>8.0f}% | {msg}")

            if state in summary:
                summary[state] += 1

        except Exception as exc:
            print(f"  {yr:>6} | {'ERROR':>12} | {'–':>10} | {exc}")

    print(f"{'─'*55}")
    total = len([t for t in tasks.values() if t is not None])
    print(f"  TỔNG: {total} tasks | "
          f"✓ {summary['COMPLETED']} hoàn thành | "
          f"⟳ {summary['RUNNING']} đang chạy | "
          f"… {summary['READY']} chờ | "
          f"✗ {summary['FAILED']} thất bại")

    all_done = (summary["RUNNING"] == 0 and summary["READY"] == 0
                and summary["FAILED"] == 0)
    if all_done and summary["COMPLETED"] > 0:
        print("\n  ✅ TẤT CẢ TASKS HOÀN THÀNH! Chạy Cell 7 để đọc kết quả.")
    else:
        print("\n  ⏳ Chạy lại cell này để cập nhật tiến độ...")

    return summary


summary = check_all_tasks(all_tasks)


# %% ─────────────────────────────────────────────────────────────
# CELL 7: Đọc CSV từ Drive + Áp đường cong A-H + Visualize
# ─────────────────────────────────────────────────────────────────
# Chạy cell này SAU KHI tất cả Export Tasks ở Cell 6 đều "COMPLETED".
# GEE đã ghi các file CSV vào: Google Drive → GEE_EXPORT_FOLDER/

def find_exported_csvs() -> list:
    """
    Tìm các file CSV đã được GEE export vào Drive.
    Trả về danh sách đường dẫn tuyệt đối.
    """
    # GEE export vào thư mục gốc Drive hoặc subfolder tùy cấu hình
    # Tìm ở nhiều vị trí có thể
    search_dirs = [
        f"/content/drive/MyDrive/{GEE_EXPORT_FOLDER}",
        f"/content/drive/MyDrive",
        DRIVE_DATA_DIR,
    ]

    found = []
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith("nuicoc_water_") and fname.endswith(".csv"):
                found.append(os.path.join(d, fname))

    return sorted(found)


def load_and_merge_gee_csvs(csv_paths: list) -> pd.DataFrame:
    """
    Đọc và gộp tất cả file CSV đã export từ GEE.

    Parameters
    ----------
    csv_paths : list of str
        Danh sách đường dẫn file CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame đã gộp, sắp xếp theo ngày.
    """
    dfs = []
    for path in csv_paths:
        try:
            df_y = pd.read_csv(path)
            year = os.path.basename(path).replace("nuicoc_water_", "").replace(".csv", "")
            print(f"  ✓ {os.path.basename(path)}: {len(df_y)} bản ghi (năm {year})")
            dfs.append(df_y)
        except Exception as exc:
            print(f"  ✗ {os.path.basename(path)}: {exc}")

    if not dfs:
        raise RuntimeError(
            "Không tìm thấy file CSV nào!\n"
            f"Kiểm tra Google Drive → '{GEE_EXPORT_FOLDER}' đã có file chưa.\n"
            "Nếu chưa: các Export Tasks (Cell 5) có thể chưa hoàn thành."
        )

    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch + ánh xạ diện tích → mực nước qua đường cong A-H.

    Phân cấp chất lượng dựa trên cloud_scene (% mây toàn cảnh):
      good : cloud_scene < 30%  → ảnh thoáng, kết quả tin cậy cao
      fair : cloud_scene 30–60% → có mây nhưng QA60 mask đã lọc
      low  : cloud_scene 60–80% → nên kiểm tra thủ công
    """
    df = df.copy()
    df["water_area_ha"] = pd.to_numeric(df["water_area_ha"], errors="coerce")
    df["cloud_scene"]   = pd.to_numeric(df["cloud_scene"],   errors="coerce")

    n_before = len(df)
    df = df[df["water_area_ha"] > 100].dropna(subset=["water_area_ha"])
    print(f"  Loại {n_before - len(df)} bản ghi (diện tích ≤ 100 ha hoặc NaN)")

    # Áp đường cong A-H (vectorized)
    df["water_level_m"] = ah_to_level(df["water_area_ha"].values)

    # Phân cấp chất lượng dựa trên % mây toàn cảnh
    df["quality"] = np.select(
        [df["cloud_scene"] < 30, df["cloud_scene"] < 60, df["cloud_scene"] <= 80],
        ["good",                  "fair",                  "low"],
        default="low",
    )

    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "water_area_ha", "water_level_m", "cloud_scene", "quality"]]


# ── Đọc và xử lý ─────────────────────────────────────────────
print("=" * 55)
print("  ĐỌC KẾT QUẢ TỪ GEE EXPORT")
print("=" * 55)

csv_paths = find_exported_csvs()
print(f"Tìm thấy {len(csv_paths)} file CSV:\n")

df_raw   = load_and_merge_gee_csvs(csv_paths)
df_final = postprocess(df_raw)

print(f"\n{'─'*55}")
print(f"✓ Tổng bản ghi hợp lệ : {len(df_final)}")
print(f"  Phân bố chất lượng:")
for q, cnt in df_final["quality"].value_counts().items():
    pct = cnt / len(df_final) * 100
    bar = "█" * max(1, int(pct / 2.5))
    print(f"    {q:5s}: {cnt:4d} ({pct:5.1f}%) {bar}")
print(f"  Mực nước: {df_final['water_level_m'].min():.2f}m "
      f"– {df_final['water_level_m'].max():.2f}m")
print(f"  Giai đoạn: {df_final['date'].min().date()} → {df_final['date'].max().date()}")

# ── Visualize ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(15, 9))
fig.suptitle(
    f"Mực nước hồ Núi Cốc — GEE Sentinel-2 | {START_YEAR}–{END_YEAR}\n"
    f"QA60 mask | Mây < {MAX_CLOUD_PCT}% | NDWI > {NDWI_THRESHOLD} | "
    f"Đường cong A-H",
    fontsize=12, fontweight="bold",
)

color_map = {"good": "royalblue", "fair": "orange", "low": "tomato"}

ax1 = axes[0]
for q, grp in df_final.groupby("quality"):
    ax1.scatter(grp["date"], grp["water_level_m"],
                c=color_map[q], label=f"{q} ({len(grp)})",
                alpha=0.75, s=25, zorder=3)
ax1.plot(df_final["date"], df_final["water_level_m"],
         color="gray", alpha=0.25, linewidth=0.7, zorder=1)
ax1.axhline(46.20, color="green",  ls="--", lw=1.3, label="MNDBT 46.20m")
ax1.axhline(47.40, color="red",    ls="--", lw=1.3, label="Lũ TK 47.40m")
ax1.set_ylabel("Mực nước (m)")
ax1.set_title("Chuỗi mực nước suy luận từ đường cong A-H")
ax1.legend(fontsize=9, ncol=4)
ax1.grid(alpha=0.3)

ax2 = axes[1]
for q, grp in df_final.groupby("quality"):
    ax2.scatter(grp["date"], grp["water_area_ha"],
                c=color_map[q], label=q, alpha=0.75, s=25)
ax2.axhline(2500, color="green", ls="--", lw=1.3, label="MNDBT ~2500 ha")
ax2.set_ylabel("Diện tích mặt nước (ha)")
ax2.set_title("Diện tích mặt nước theo thời gian")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_path = f"{DRIVE_ROOT}/results/gee_result.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✓ Biểu đồ đã lưu: {plot_path}")


# %% ─────────────────────────────────────────────────────────────
# CELL 8: Lưu file cuối cùng vào Drive
# ─────────────────────────────────────────────────────────────────

df_final.to_csv(OUTPUT_CSV, index=False)
df_raw.to_csv(OUTPUT_CSV.replace(".csv", "_raw.csv"), index=False)

print("=" * 55)
print("✓ ĐÃ LƯU VÀO GOOGLE DRIVE:")
print(f"  Chính : {OUTPUT_CSV}")
print(f"  Thô   : {OUTPUT_CSV.replace('.csv', '_raw.csv')}")
print(f"  Biểu đồ: {plot_path}")
print("=" * 55)
print()
print("BƯỚC TIẾP THEO (trên máy tính cá nhân):")
print(f"  1. Tải file: {OUTPUT_CSV}")
print("  2. Đặt vào: DATN/data/raw/gee_water_level.csv")
print("  3. Chạy: python 05_integrate.py")
print("=" * 55)

display(df_final.head(10))
print(f"\nThống kê:\n{df_final.describe().round(3)}")

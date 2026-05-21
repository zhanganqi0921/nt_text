"""
合并 master 索引与端点标注，生成训练用数据表 `dataset_350-700.csv`。

主要功能：
1) 读取配置的数据根目录下的 `labels/master.csv` 获取图像路径、尺寸与病例元信息；
2) 读取配置的数据根目录下的 `labels/endpoints.csv` 获取 P1/P2 端点坐标；
3) 按 `image_id` 合并两张表，并默认写入 `quality_label=2`；
4) 计算端点距离 `D_gt` 与方向角 `angle_gt`；
5) 若历史输出表中已填写 `nt_thickness_mm`，自动保留并计算 `scale_s`。

输入与输出：
- 输入主索引：`MASTER_CSV`
- 输入端点表：`ENDPOINTS_CSV`
- 输出数据表：`DATASET_350_700_CSV`

输出字段：
- `image_id`, `image_name`, `image_path_raw_png`, `image_path_clean`, `image_path_train_clean`, `mask_path`, `width`, `height`
- `patient_id`, `exam_id`, `original_date`, `quality_label`
- `x1`, `y1`, `x2`, `y2`
- `nt_thickness_mm`, `D_gt`, `angle_gt`, `scale_s`

约束与策略：
- `endpoints.csv` 中缺少 `image_id` 或坐标无法解析的行会跳过；
- `master.csv` 中找不到匹配 `image_id` 的端点行会跳过并在终端汇总；
- `nt_thickness_mm` 首次构建为空，后续重建时会尽量从既有输出表保留。

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from dataset_paths import CLEAN_DIR, DATASET_350_700_CSV, DATASET_DIR, ENDPOINTS_CSV, MASK_DIR, MASTER_CSV, TRAIN_CLEAN_DIR


OUTPUT_CSV = DATASET_350_700_CSV


OUTPUT_FIELDS = [
    "image_id",
    "image_name",
    "image_path_raw_png",
    "image_path_clean",
    "image_path_train_clean",
    "mask_path",
    "width",
    "height",
    "patient_id",
    "exam_id",
    "original_date",
    "quality_label",
    "x1",
    "y1",
    "x2",
    "y2",
    "nt_thickness_mm",
    "D_gt",
    "angle_gt",
    "scale_s",
]


def _parse_float(value: str) -> float | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_master_rows() -> dict[str, dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    with MASTER_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                rows_by_id[image_id] = row
    return rows_by_id


def load_existing_nt_values() -> dict[str, str]:
    nt_by_id: dict[str, str] = {}
    if not OUTPUT_CSV.is_file():
        return nt_by_id
    with OUTPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            nt_by_id[image_id] = (row.get("nt_thickness_mm") or "").strip()
    return nt_by_id


def _relative_to_dataset(path: Path) -> str:
    return path.relative_to(DATASET_DIR).as_posix()


def build_dataset() -> tuple[int, int, int]:
    master_rows = load_master_rows()
    existing_nt_by_id = load_existing_nt_values()
    merged_rows: list[dict[str, str]] = []
    skipped_missing_master = 0
    skipped_bad_or_missing_endpoint = 0
    d_gt_bad = 0

    with ENDPOINTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            if str(row.get("labeled", "0")).strip() != "1":
                skipped_bad_or_missing_endpoint += 1
                continue
            master = master_rows.get(image_id)
            if master is None:
                skipped_missing_master += 1
                continue

            x1 = _parse_float(row.get("x1", ""))
            y1 = _parse_float(row.get("y1", ""))
            x2 = _parse_float(row.get("x2", ""))
            y2 = _parse_float(row.get("y2", ""))
            if None in (x1, y1, x2, y2):
                skipped_bad_or_missing_endpoint += 1
                continue

            # 预留字段：首次构建为空；若历史文件已填值则自动保留。
            nt_thickness_mm = existing_nt_by_id.get(image_id, "")
            d_gt = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if d_gt <= 0:
                d_gt_bad += 1
                continue
            angle_gt = math.atan2(y2 - y1, x2 - x1)

            scale_s = ""
            nt_value = _parse_float(nt_thickness_mm)
            if nt_value is not None:
                scale_s = f"{nt_value / d_gt:.8f}"

            merged_rows.append(
                {
                    "image_id": image_id,
                    "image_name": (master.get("image_name") or f"{image_id}.png").strip(),
                    "image_path_raw_png": (master.get("image_path_raw_png") or "").strip(),
                    "image_path_clean": _relative_to_dataset(CLEAN_DIR / f"{image_id}_clean.png"),
                    "image_path_train_clean": _relative_to_dataset(
                        TRAIN_CLEAN_DIR / f"{image_id}_train_clean.png"
                    ),
                    "mask_path": _relative_to_dataset(MASK_DIR / f"{image_id}_mask.png"),
                    "width": (master.get("width") or "").strip(),
                    "height": (master.get("height") or "").strip(),
                    "patient_id": (master.get("patient_id") or "").strip(),
                    "exam_id": (master.get("exam_id") or "").strip(),
                    "original_date": (master.get("original_date") or "").strip(),
                    "quality_label": (master.get("quality_label") or "2").strip(),
                    "x1": str(int(round(x1))),
                    "y1": str(int(round(y1))),
                    "x2": str(int(round(x2))),
                    "y2": str(int(round(y2))),
                    "nt_thickness_mm": nt_thickness_mm,
                    "D_gt": f"{d_gt:.6f}",
                    "angle_gt": f"{angle_gt:.6f}",
                    "scale_s": scale_s,
                }
            )

    merged_rows.sort(key=lambda r: r["image_id"])
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(merged_rows)

    return len(merged_rows), skipped_missing_master + skipped_bad_or_missing_endpoint, d_gt_bad


def main() -> None:
    count, skipped, d_gt_bad = build_dataset()
    print(f"写入完成: {OUTPUT_CSV}")
    print(f"合并行数: {count}")
    print(f"缺失/未标注/无法合并数量: {skipped}")
    print(f"D_gt 异常数量: {d_gt_bad}")


if __name__ == "__main__":
    main()

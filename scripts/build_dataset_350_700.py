from __future__ import annotations

import csv
import math
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parents[1]
LABELS_DIR = DATASET_DIR / "labels"
MASTER_CSV = LABELS_DIR / "master.csv"
ENDPOINTS_CSV = LABELS_DIR / "endpoints.csv"
OUTPUT_CSV = LABELS_DIR / "dataset_350-700.csv"


OUTPUT_FIELDS = [
    "image_id",
    "image_path_raw_png",
    "width",
    "height",
    "patient_id",
    "exam_id",
    "original_date",
    "x1",
    "y1",
    "x2",
    "y2",
    "quality_label",
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


def build_dataset() -> tuple[int, int]:
    master_rows = load_master_rows()
    existing_nt_by_id = load_existing_nt_values()
    merged_rows: list[dict[str, str]] = []
    skipped_missing_master = 0

    with ENDPOINTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
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
                continue

            # 预留字段：首次构建为空；若历史文件已填值则自动保留。
            nt_thickness_mm = existing_nt_by_id.get(image_id, "")
            d_gt = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            angle_gt = math.atan2(y2 - y1, x2 - x1)

            scale_s = ""
            nt_value = _parse_float(nt_thickness_mm)
            if nt_value is not None and d_gt != 0:
                scale_s = f"{nt_value / d_gt:.8f}"

            merged_rows.append(
                {
                    "image_id": image_id,
                    "image_path_raw_png": (master.get("image_path_raw_png") or "").strip(),
                    "width": (master.get("width") or "").strip(),
                    "height": (master.get("height") or "").strip(),
                    "patient_id": (master.get("patient_id") or "").strip(),
                    "exam_id": (master.get("exam_id") or "").strip(),
                    "original_date": (master.get("original_date") or "").strip(),
                    "x1": str(int(round(x1))),
                    "y1": str(int(round(y1))),
                    "x2": str(int(round(x2))),
                    "y2": str(int(round(y2))),
                    "quality_label": "2",
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

    return len(merged_rows), skipped_missing_master


def main() -> None:
    count, skipped = build_dataset()
    print(f"写入完成: {OUTPUT_CSV}")
    print(f"合并行数: {count}")
    print(f"缺少 master 匹配而跳过: {skipped}")


if __name__ == "__main__":
    main()

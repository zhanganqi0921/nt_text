"""
生成数据主索引 `master.csv`，为后续标注校验与统计提供统一元信息表。

主要功能：
1) 扫描 `dataset/images_raw_png` 下全部 PNG；
2) 读取每张图的宽高信息；
3) 基于文件名（如 `N0001_A0001_19860113_1.png`）解析 patient/exam/date；
4) 汇总写入 `dataset/labels/master.csv`。

输出字段：
- `image_id`, `image_name`, `image_path_raw_png`, `width`, `height`
- `patient_id`, `exam_id`, `original_date`

约束与策略：
- 文件名不匹配解析规则时，不中断流程，对应解析字段留空；
- 若输入目录不存在或为空，会直接报错退出，避免写出空索引。

文档约定（`dataset/scripts/`）：
- 新建或在本目录新增脚本时，应在文件最顶部编写与本模块相同结构的说明字符串：一句话概述、主要功能（编号列表）、输入与输出路径、字段或产物说明、约束与策略、命令行用法示例（若脚本支持参数）。
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image

DATASET_DIR = Path(__file__).resolve().parents[1]
PNG_DIR = DATASET_DIR / "images_raw_png"
LABELS_DIR = DATASET_DIR / "labels"
MASTER_CSV = LABELS_DIR / "master.csv"

# N0001_A0001_19860113_1.png（仅解析 stem；不符合时相关字段留空，不中断）
STEM_PARSE_RE = re.compile(
    r"^(?P<patient_id>N\d+)_(?P<exam_id>A\d+)_(?P<original_date>\d{8})_(?P<tail>.+)$"
)

FIELDNAMES = [
    "image_id",
    "image_name",
    "image_path_raw_png",
    "width",
    "height",
    "patient_id",
    "exam_id",
    "original_date",
]


def parse_stem(stem: str) -> tuple[str, str, str]:
    """返回 (patient_id, exam_id, original_date)，无法解析则为空字符串。"""
    m = STEM_PARSE_RE.match(stem)
    if not m:
        return "", "", ""
    d = m.groupdict()
    return d["patient_id"], d["exam_id"], d["original_date"]


def row_for_png(png_path: Path) -> dict[str, str]:
    image_name = png_path.name
    image_id = png_path.stem
    rel = png_path.relative_to(DATASET_DIR).as_posix()
    patient_id, exam_id, original_date = parse_stem(image_id)
    with Image.open(png_path) as im:
        w, h = im.size
    return {
        "image_id": image_id,
        "image_name": image_name,
        "image_path_raw_png": rel,
        "width": str(w),
        "height": str(h),
        "patient_id": patient_id,
        "exam_id": exam_id,
        "original_date": original_date,
    }


def main() -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if not PNG_DIR.is_dir():
        raise SystemExit(f"未找到目录: {PNG_DIR}")
    pngs = sorted(PNG_DIR.glob("*.png"))
    if not pngs:
        raise SystemExit(f"{PNG_DIR} 下没有 PNG，请先运行 convert_bmp_to_png.py")
    rows = [row_for_png(p) for p in pngs]
    with MASTER_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"已写入 {len(rows)} 条记录 -> {MASTER_CSV}")


if __name__ == "__main__":
    main()

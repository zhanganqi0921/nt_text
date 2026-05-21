"""
生成数据主索引 `master.csv`，为后续标注校验与统计提供统一元信息表。

主要功能：
1) 扫描配置的 PNG 子目录（默认 `/Users/zaq/Desktop/dataset/images_raw_png/2`）下全部 PNG；
2) 读取每张图的宽高信息；
3) 基于文件名（如 `N0001_A0001_19860113_1.png`）解析 patient/exam/date；
4) 汇总写入配置的数据根目录下的 `labels/master.csv`。

输出字段：
- `image_id`, `image_name`, `image_path_raw_png`, `width`, `height`
- `patient_id`, `exam_id`, `original_date`, `quality_label`

约束与策略：
- 文件名不匹配解析规则时，不中断流程，对应解析字段留空；
- 若输入目录不存在或为空，会直接报错退出，避免写出空索引。

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。

文档约定（`scripts/`）：
- 新建或在本目录新增脚本时，应在文件最顶部编写与本模块相同结构的说明字符串：一句话概述、主要功能（编号列表）、输入与输出路径、字段或产物说明、约束与策略、命令行用法示例（若脚本支持参数）。
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image

from dataset_paths import DATASET_DIR, LABELS_DIR, MASTER_CSV, PNG_DIR

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
    "quality_label",
]


def parse_stem(stem: str) -> tuple[str, str, str, bool]:
    """返回 (patient_id, exam_id, original_date, 是否成功解析)。"""
    m = STEM_PARSE_RE.match(stem)
    if not m:
        return "", "", "", False
    d = m.groupdict()
    return d["patient_id"], d["exam_id"], d["original_date"], True


def row_for_png(png_path: Path) -> tuple[dict[str, str], bool]:
    image_name = png_path.name
    image_id = png_path.stem
    rel = png_path.relative_to(DATASET_DIR).as_posix()
    patient_id, exam_id, original_date, parsed = parse_stem(image_id)
    with Image.open(png_path) as im:
        w, h = im.size
    return (
        {
            "image_id": image_id,
            "image_name": image_name,
            "image_path_raw_png": rel,
            "width": str(w),
            "height": str(h),
            "patient_id": patient_id,
            "exam_id": exam_id,
            "original_date": original_date,
            "quality_label": "2",
        },
        parsed,
    )


def _format_size_stats(rows: list[dict[str, str]]) -> str:
    widths = [int(r["width"]) for r in rows]
    heights = [int(r["height"]) for r in rows]
    unique_sizes = sorted({(w, h) for w, h in zip(widths, heights, strict=True)})
    return (
        f"width min/max={min(widths)}/{max(widths)}, "
        f"height min/max={min(heights)}/{max(heights)}, "
        f"unique_sizes={len(unique_sizes)}"
    )


def main() -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if not PNG_DIR.is_dir():
        raise SystemExit(f"未找到目录: {PNG_DIR}")
    pngs = sorted(PNG_DIR.glob("*.png"))
    if not pngs:
        raise SystemExit(f"{PNG_DIR} 下没有 PNG，请先运行 convert_bmp_to_png.py")
    rows: list[dict[str, str]] = []
    unparsed = 0
    for png_path in pngs:
        row, parsed = row_for_png(png_path)
        rows.append(row)
        if not parsed:
            unparsed += 1
            print(f"warning: 文件名无法解析 patient/exam/date: {png_path.name}")
    with MASTER_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"总 PNG 数量: {len(pngs)}")
    print(f"master.csv 行数: {len(rows)}")
    print(f"宽高统计: {_format_size_stats(rows)}")
    print(f"无法解析文件名数量: {unparsed}")
    print(f"已写入: {MASTER_CSV}")


if __name__ == "__main__":
    main()

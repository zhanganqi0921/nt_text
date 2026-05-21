"""
批量将原始 BMP 图像转换为 PNG，作为后续标注与建表流程的标准输入。

主要功能：
1) 扫描 `dataset/images_raw_bmp` 下全部 `.bmp` 文件；
2) 逐个转换为同 stem 的 `.png`，输出到 `dataset/images_raw_png`；
3) 若目标 PNG 已存在则跳过，避免重复覆盖。

输入输出：
- 输入目录：`dataset/images_raw_bmp`
- 输出目录：`dataset/images_raw_png`
- 返回统计：新转换数量、已存在跳过数量

使用方式：
- 直接运行：`python dataset/scripts/convert_bmp_to_png.py`
- 常见顺序：先执行本脚本，再执行 `create_master_csv.py`。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

DATASET_DIR = Path(__file__).resolve().parents[1]
BMP_DIR = DATASET_DIR / "images_raw_bmp"
PNG_DIR = DATASET_DIR / "images_raw_png"


def convert_all() -> tuple[int, int]:
    """返回 (新转换数量, 跳过数量)。"""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    for bmp_path in sorted(BMP_DIR.glob("*.bmp")):
        out_path = PNG_DIR / (bmp_path.stem + ".png")
        if out_path.is_file():
            skipped += 1
            continue
        with Image.open(bmp_path) as im:
            im.save(out_path, format="PNG")
        converted += 1
    return converted, skipped


def main() -> None:
    if not BMP_DIR.is_dir():
        raise SystemExit(f"未找到目录: {BMP_DIR}")
    converted, skipped = convert_all()
    print(f"BMP -> PNG 新转换: {converted} 个，已存在跳过: {skipped} 个，输出目录: {PNG_DIR}")


if __name__ == "__main__":
    main()

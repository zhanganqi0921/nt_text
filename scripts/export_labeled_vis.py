"""
根据 `endpoints.csv` 批量导出端点可视化图，帮助进行标注复查与质检。

主要功能：
1) 读取配置的数据根目录下 `labels/endpoints.csv` 中 `labeled=1` 的记录；
2) 按 `image_path_raw_png` 回读原图；
3) 在图上绘制 P1/P2、两点连线与坐标文字；
4) 输出到配置的数据子目录下的 `images_labeled_vis/2/{image_id}_labeled.png`。

容错策略：
- 坐标缺失/无效、图像路径缺失、图像不存在或不可读时跳过并计数；
- 处理完成后打印“成功生成/跳过”统计。

典型用途：
- 当手工修改了 `endpoints.csv`，或想统一重建复查图时执行本脚本。

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。
"""
from __future__ import annotations

from pathlib import Path

import label_endpoints as le


def main() -> None:
    if not le.ENDPOINTS_CSV.is_file():
        raise SystemExit(f"未找到: {le.ENDPOINTS_CSV}")
    rows = le.load_endpoints()
    if not rows:
        raise SystemExit(f"{le.ENDPOINTS_CSV} 中尚无带 image_id 的记录。")

    n_ok = 0
    n_skip = 0
    for iid, r in sorted(rows.items(), key=lambda kv: kv[0]):
        if int(r.get("labeled", 0) or 0) != 1:
            continue
        try:
            x1 = int(float(r.get("x1", "")))
            y1 = int(float(r.get("y1", "")))
            x2 = int(float(r.get("x2", "")))
            y2 = int(float(r.get("y2", "")))
        except ValueError:
            print(f"跳过（坐标无效）: {iid}")
            n_skip += 1
            continue
        rel = (r.get("image_path_raw_png") or r.get("image_path") or "").strip()
        if not rel:
            print(f"跳过（无 image_path_raw_png）: {iid}")
            n_skip += 1
            continue
        png_path = le.DATASET_DIR / Path(rel)
        if not png_path.is_file():
            print(f"跳过（图像不存在）: {iid} -> {png_path}")
            n_skip += 1
            continue
        bgr = le.imread_unicode(png_path)
        if bgr is None:
            print(f"跳过（无法读取）: {png_path}")
            n_skip += 1
            continue
        out = le.write_labeled_review_png(iid, bgr, x1, y1, x2, y2)
        print(out.relative_to(le.DATASET_DIR).as_posix())
        n_ok += 1

    print(f"完成: 生成 {n_ok} 张，跳过 {n_skip} 条。")


if __name__ == "__main__":
    main()

"""
对标注结果做快速体检，输出可直接用于人工复核的质量统计。

主要功能：
1) 读取 `labels/master.csv` 与 `labels/endpoints.csv`；
2) 统计总图像数、已标注数、未标注数；
3) 检查已标注坐标是否越界（是否落在图像尺寸范围外）；
4) 检查 `master` 与 `endpoints` 中记录的图像路径是否真实存在。

输出说明：
- `坐标越界数量` 包含坐标解析失败和越界两类异常；
- `图像路径缺失数量` 按 `image_id` 去重统计，避免重复计数。

使用场景：
- 每轮批量标注后运行一次，快速判断是否需要回到标注环节修正。
"""
from __future__ import annotations

import csv
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
LABELS_DIR = DATASET_DIR / "labels"
MASTER_CSV = LABELS_DIR / "master.csv"
ENDPOINTS_CSV = LABELS_DIR / "endpoints.csv"


def load_master() -> list[dict[str, str]]:
    if not MASTER_CSV.is_file():
        return []
    with MASTER_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_endpoints() -> list[dict[str, str]]:
    if not ENDPOINTS_CSV.is_file():
        return []
    with ENDPOINTS_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_under_dataset(rel: str) -> Path:
    p = Path(rel.strip())
    if p.is_absolute():
        return p
    return DATASET_DIR / p


def main() -> None:
    master_rows = load_master()
    ep_rows = load_endpoints()

    total = len(master_rows)
    master_by_id = {
        (r.get("image_id") or "").strip(): r
        for r in master_rows
        if (r.get("image_id") or "").strip()
    }

    labeled_ids: set[str] = set()
    for r in ep_rows:
        iid = (r.get("image_id") or "").strip()
        if not iid:
            continue
        try:
            labeled = int(str(r.get("labeled", "0")).strip() or "0")
        except ValueError:
            labeled = 0
        if labeled == 1:
            labeled_ids.add(iid)

    labeled_count = len(labeled_ids)
    unlabeled = max(0, total - labeled_count)

    oob_count = 0
    for r in ep_rows:
        if str(r.get("labeled", "0")).strip() != "1":
            continue
        iid = (r.get("image_id") or "").strip()
        try:
            x1 = int(float(r.get("x1", "")))
            y1 = int(float(r.get("y1", "")))
            x2 = int(float(r.get("x2", "")))
            y2 = int(float(r.get("y2", "")))
        except ValueError:
            oob_count += 1
            continue
        m = master_by_id.get(iid)
        if not m:
            continue
        try:
            w = int(m["width"])
            h = int(m["height"])
        except (KeyError, ValueError):
            continue
        bad = any(x < 0 or y < 0 or x >= w or y >= h for x, y in ((x1, y1), (x2, y2)))
        if bad:
            oob_count += 1

    bad_path_ids: set[str] = set()
    for r in master_rows:
        iid = (r.get("image_id") or "").strip()
        rel = (r.get("image_path_raw_png") or "").strip()
        if not iid:
            continue
        if not rel or not resolve_under_dataset(rel).is_file():
            bad_path_ids.add(iid)
    for r in ep_rows:
        if str(r.get("labeled", "0")).strip() != "1":
            continue
        iid = (r.get("image_id") or "").strip()
        rel = (r.get("image_path") or "").strip()
        if not iid:
            continue
        if not rel or not resolve_under_dataset(rel).is_file():
            bad_path_ids.add(iid)

    missing_count = len(bad_path_ids)

    print(f"总图像数量: {total}")
    print(f"已标注端点数量: {labeled_count}")
    print(f"未标注数量: {unlabeled}")
    print(f"坐标越界数量: {oob_count}")
    print(f"图像路径缺失数量: {missing_count}")


if __name__ == "__main__":
    main()

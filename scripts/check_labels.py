"""
训练前数据完整性检查。

主要功能：
1) 读取 `labels/dataset_350-700.csv`；
2) 检查 raw / clean / train_clean / marker mask 路径是否存在；
3) 检查原图端点坐标、D_gt、angle_gt、scale_s 逻辑；
4) 检查 split 是否为 train/val/test，并统计 train/val/test 数量；
5) 检查 patient_id / exam_id 是否跨 split；
6) 统计 quality_label 分布；
7) 对 train_clean 中疑似残留 NT/Dist/设备文字区域生成可视化，供人工复查。

输出：
- 终端打印汇总统计；
- 详细异常列表写入 `logs/check_labels_detail.log`；
- 疑似文字残留可视化写入 `final_check_vis/{set}/text_artifacts/`。

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dataset_paths import DATASET_350_700_CSV, DATASET_DIR, FINAL_CHECK_VIS_DIR, LOG_DIR, MASK_DIR


VALID_SPLITS = {"train", "val", "test"}


def load_dataset_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise SystemExit(f"缺少训练 CSV: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_under_dataset(rel: str) -> Path:
    p = Path((rel or "").strip())
    if p.is_absolute():
        return p
    return DATASET_DIR / p


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix.lower() or ".png", img)
    if not ok or buf is None:
        return False
    buf.tofile(str(path))
    return True


def parse_float(value: str) -> float | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_int(value: str) -> int | None:
    v = parse_float(value)
    return None if v is None else int(round(v))


def mask_path_for_row(row: dict[str, str]) -> Path:
    image_id = (row.get("image_id") or "").strip()
    rel = (row.get("mask_path") or row.get("image_path_mask") or "").strip()
    if rel:
        return resolve_under_dataset(rel)
    return MASK_DIR / f"{image_id}_mask.png"


def check_file(path_value: str, fallback: Path | None = None) -> bool:
    if path_value:
        return resolve_under_dataset(path_value).is_file()
    return fallback is not None and fallback.is_file()


def is_coord_oob(row: dict[str, str], coords: tuple[float, float, float, float]) -> bool:
    w = parse_int(row.get("width", ""))
    h = parse_int(row.get("height", ""))
    if w is None or h is None or w <= 0 or h <= 0:
        return True
    x1, y1, x2, y2 = coords
    return any(x < 0 or y < 0 or x >= w or y >= h for x, y in ((x1, y1), (x2, y2)))


def find_cross_split_ids(rows: list[dict[str, str]], key: str) -> dict[str, set[str]]:
    by_key: dict[str, set[str]] = {}
    for row in rows:
        value = (row.get(key) or "").strip()
        split = (row.get("split") or "").strip()
        if not value or split not in VALID_SPLITS:
            continue
        by_key.setdefault(value, set()).add(split)
    return {value: splits for value, splits in by_key.items() if len(splits) > 1}


def build_text_suspect_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Heuristic mask for obvious NT/Dist/device text remnants in train_clean."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Colored remnants: NT/Pctl text often yellow/red; device overlays may be green/blue.
    yellow = cv2.inRange(hsv, (15, 35, 70), (45, 255, 255))
    green = cv2.inRange(hsv, (45, 45, 65), (95, 255, 255))
    blue = cv2.inRange(hsv, (90, 55, 65), (135, 255, 255))
    red1 = cv2.inRange(hsv, (0, 45, 65), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 45, 65), (180, 255, 255))
    color = cv2.bitwise_or(yellow, cv2.bitwise_or(green, cv2.bitwise_or(blue, cv2.bitwise_or(red1, red2))))

    # White/gray text remnants near image edges and corners.
    edge_roi = np.zeros((h, w), dtype=np.uint8)
    edge_roi[: int(round(h * 0.12)), :] = 255
    edge_roi[int(round(h * 0.88)) :, :] = 255
    edge_roi[:, : int(round(w * 0.12))] = 255
    edge_roi[:, int(round(w * 0.88)) :] = 255

    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((7, 7), dtype=np.uint8))
    local_bright = cv2.bitwise_and(cv2.inRange(top_hat, 24, 255), cv2.inRange(gray, 90, 255))
    gray_edge = cv2.bitwise_and(local_bright, edge_roi)

    mask = cv2.bitwise_or(color, gray_edge)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    max_area = int(h * w * 0.004)
    for idx in range(1, n):
        x, y, bw, bh, area = stats[idx]
        if area < 3 or area > max_area:
            continue
        if bw > int(w * 0.45) or bh > int(h * 0.25):
            continue
        out[labels == idx] = 255
    if np.any(out):
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return out


def make_text_artifact_vis(img_bgr: np.ndarray, suspect_mask: np.ndarray) -> np.ndarray:
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = suspect_mask
    overlay = cv2.addWeighted(img_bgr, 0.75, red, 0.35, 0.0)
    mask_bgr = cv2.cvtColor(suspect_mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([img_bgr, mask_bgr, overlay])


def assess_split_counts(split_counts: Counter[str], total: int) -> list[str]:
    warnings: list[str] = []
    for split in ("train", "val", "test"):
        if split_counts.get(split, 0) == 0:
            warnings.append(f"{split} 数量为 0")
    if total > 0 and split_counts.get("train", 0) < split_counts.get("val", 0):
        warnings.append("train 数量少于 val")
    if total > 0 and split_counts.get("train", 0) < split_counts.get("test", 0):
        warnings.append("train 数量少于 test")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="训练前数据完整性检查")
    parser.add_argument("--csv", type=Path, default=DATASET_350_700_CSV, help="训练 CSV 路径")
    parser.add_argument("--text-vis-samples", type=int, default=30, help="疑似文字残留可视化最多输出数量")
    parser.add_argument("--no-text-vis", action="store_true", help="不生成疑似文字残留可视化")
    args = parser.parse_args()

    rows = load_dataset_rows(args.csv.resolve())
    total = len(rows)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    detail_log = LOG_DIR / "check_labels_detail.log"
    text_vis_dir = FINAL_CHECK_VIS_DIR / "text_artifacts"

    missing_raw: list[str] = []
    missing_clean: list[str] = []
    missing_train_clean: list[str] = []
    missing_mask: list[str] = []
    empty_coord: list[str] = []
    coord_oob: list[str] = []
    d_gt_bad: list[str] = []
    angle_bad: list[str] = []
    scale_s_bad: list[str] = []
    split_bad: list[str] = []
    split_missing: list[str] = []
    text_suspects: list[tuple[str, int, Path]] = []

    quality_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for row in rows:
        image_id = (row.get("image_id") or "").strip()
        if not image_id:
            continue

        raw_rel = (row.get("image_path_raw_png") or "").strip()
        clean_rel = (row.get("image_path_clean") or "").strip()
        train_clean_rel = (row.get("image_path_train_clean") or "").strip()
        if not check_file(raw_rel):
            missing_raw.append(image_id)
        if not check_file(clean_rel):
            missing_clean.append(image_id)
        if not check_file(train_clean_rel):
            missing_train_clean.append(image_id)
        if not mask_path_for_row(row).is_file():
            missing_mask.append(image_id)

        coord_values = [parse_float(row.get(field, "")) for field in ("x1", "y1", "x2", "y2")]
        if any(v is None for v in coord_values):
            empty_coord.append(image_id)
        else:
            coords = tuple(float(v) for v in coord_values if v is not None)
            if is_coord_oob(row, coords):
                coord_oob.append(image_id)

        d_gt = parse_float(row.get("D_gt", ""))
        if d_gt is None or d_gt <= 0:
            d_gt_bad.append(image_id)

        angle = parse_float(row.get("angle_gt", ""))
        if angle is None:
            angle_bad.append(image_id)

        nt = parse_float(row.get("nt_thickness_mm", ""))
        scale_s = parse_float(row.get("scale_s", ""))
        if nt is not None and (scale_s is None or scale_s <= 0):
            scale_s_bad.append(image_id)

        quality_counts[(row.get("quality_label") or "<empty>").strip() or "<empty>"] += 1

        if "split" in row:
            split = (row.get("split") or "").strip()
            if not split:
                split_missing.append(image_id)
            elif split not in VALID_SPLITS:
                split_bad.append(image_id)
                split_counts[split] += 1
            else:
                split_counts[split] += 1
        else:
            split_missing.append(image_id)

        if not args.no_text_vis and train_clean_rel:
            train_clean_path = resolve_under_dataset(train_clean_rel)
            img = imread_unicode(train_clean_path)
            if img is not None:
                suspect_mask = build_text_suspect_mask(img)
                score = int(np.count_nonzero(suspect_mask))
                if score > 0:
                    text_suspects.append((image_id, score, train_clean_path))

    patient_cross = find_cross_split_ids(rows, "patient_id") if "split" in (rows[0] if rows else {}) else {}
    exam_cross = find_cross_split_ids(rows, "exam_id") if "split" in (rows[0] if rows else {}) else {}
    split_warnings = assess_split_counts(split_counts, total) if "split" in (rows[0] if rows else {}) else ["CSV 缺少 split 列"]

    text_suspects.sort(key=lambda x: x[1], reverse=True)
    n_text_vis = 0
    if not args.no_text_vis and args.text_vis_samples > 0:
        text_vis_dir.mkdir(parents=True, exist_ok=True)
        for image_id, score, path in text_suspects[: args.text_vis_samples]:
            img = imread_unicode(path)
            if img is None:
                continue
            suspect_mask = build_text_suspect_mask(img)
            vis = make_text_artifact_vis(img, suspect_mask)
            out_path = text_vis_dir / f"{image_id}_text_artifact_check.png"
            if imwrite_unicode(out_path, vis):
                n_text_vis += 1

    with detail_log.open("w", encoding="utf-8") as f:
        f.write(f"# check_labels run {datetime.now(timezone.utc).isoformat()} csv={args.csv.resolve()}\n")
        groups: list[tuple[str, list[str]]] = [
            ("raw 缺失", missing_raw),
            ("clean 缺失", missing_clean),
            ("train_clean 缺失", missing_train_clean),
            ("mask 缺失", missing_mask),
            ("坐标为空/无效", empty_coord),
            ("坐标越界", coord_oob),
            ("D_gt 异常", d_gt_bad),
            ("angle_gt 异常", angle_bad),
            ("scale_s 异常", scale_s_bad),
            ("split 缺失", split_missing),
            ("split 非法", split_bad),
            ("疑似文字残留", [f"{iid} score={score}" for iid, score, _ in text_suspects]),
        ]
        for title, ids in groups:
            f.write(f"\n## {title} ({len(ids)})\n")
            for item in ids:
                f.write(f"{item}\n")
        f.write(f"\n## patient_id 跨 split ({len(patient_cross)})\n")
        for value, splits in sorted(patient_cross.items()):
            f.write(f"{value}: {sorted(splits)}\n")
        f.write(f"\n## exam_id 跨 split ({len(exam_cross)})\n")
        for value, splits in sorted(exam_cross.items()):
            f.write(f"{value}: {sorted(splits)}\n")
        f.write(f"\n## split 数量提示 ({len(split_warnings)})\n")
        for warning in split_warnings:
            f.write(f"{warning}\n")

    print(f"总样本数: {total}")
    if "split" in (rows[0] if rows else {}):
        print(f"train: {split_counts.get('train', 0)}")
        print(f"val: {split_counts.get('val', 0)}")
        print(f"test: {split_counts.get('test', 0)}")
    else:
        print("train: 0（CSV 缺少 split 列）")
        print("val: 0（CSV 缺少 split 列）")
        print("test: 0（CSV 缺少 split 列）")
    for label, count in sorted(quality_counts.items()):
        print(f"quality_label={label}: {count}")
    print(f"raw 缺失: {len(missing_raw)}")
    print(f"clean 缺失: {len(missing_clean)}")
    print(f"train_clean 缺失: {len(missing_train_clean)}")
    print(f"mask 缺失: {len(missing_mask)}")
    print(f"坐标为空/无效: {len(empty_coord)}")
    print(f"坐标越界: {len(coord_oob)}")
    print(f"D_gt 异常: {len(d_gt_bad)}")
    print(f"angle_gt 异常: {len(angle_bad)}")
    print(f"scale_s 异常: {len(scale_s_bad)}")
    print(f"split 缺失: {len(split_missing)}")
    print(f"split 非法: {len(split_bad)}")
    print(f"split 数量提示: {len(split_warnings)}")
    print(f"patient/exam 跨 split: {len(patient_cross) + len(exam_cross)}")
    print(f"文字残留候选图像（启发式排序，非硬错误）: {len(text_suspects)}")
    print(f"文字残留候选可视化 Top {n_text_vis}: {text_vis_dir}")
    print(f"详细日志: {detail_log}")


if __name__ == "__main__":
    main()

"""
根据 dataset CSV 中的端点批量生成 Marker Mask 与 Clean Image。

- Marker Mask：P1/P2 为半径 circle_radius 的圆，连线线宽 line_thickness；背景 0、标记 255。
- Clean Image：对 mask 区域做 OpenCV inpaint（默认 TELEA），供训练使用（非 labeled_vis）。

输入：配置的数据根目录下的 labels CSV（默认 dataset_350-700.csv），需含 image_path_raw_png、x1,y1,x2,y2。
输出：配置的数据子目录下的 masks_marker/2/{image_id}_mask.png，images_clean/2/{image_id}_clean.png；
默认生成：clean_vis/2/{image_id}_vis.png（原图 | mask | clean | 原图叠加 mask 和 P1/P2）。

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from dataset_paths import CLEAN_DIR, CLEAN_VIS_DIR, DATASET_DIR, LABELS_DIR, LOG_DIR, MASK_DIR


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok or buf is None:
        return False
    buf.tofile(str(path))
    return True


def _parse_int(s: str) -> int | None:
    t = (s or "").strip()
    if not t:
        return None
    try:
        return int(round(float(t)))
    except ValueError:
        return None


def _line_thickness_from_span(
    x1: int, y1: int, x2: int, y2: int, lo: int, hi: int
) -> int:
    d = math.hypot(x2 - x1, y2 - y1)
    if d <= 1e-6:
        return lo
    t = int(round(d / 80.0))
    return max(lo, min(hi, t))


def build_marker_mask(
    h: int,
    w: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    circle_radius: int,
    line_thickness: int | None,
    lo_thick: int,
    hi_thick: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    thick = (
        line_thickness
        if line_thickness is not None
        else _line_thickness_from_span(x1, y1, x2, y2, lo_thick, hi_thick)
    )
    cv2.line(mask, (x1, y1), (x2, y2), 255, thick, lineType=cv2.LINE_AA)
    cv2.circle(mask, (x1, y1), circle_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (x2, y2), circle_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    return mask


def _mask_bgr(mask_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def _raw_overlay_bgr(
    orig_bgr: np.ndarray,
    mask_u8: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
) -> np.ndarray:
    overlay = orig_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = mask_u8
    overlay = cv2.addWeighted(overlay, 0.75, red, 0.25, 0.0)
    cv2.circle(overlay, p1, 8, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, p2, 8, (255, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(overlay, "P1", (p1[0] + 10, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(overlay, "P2", (p2[0] + 10, p2[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return overlay


def make_check_vis(
    orig_bgr: np.ndarray,
    mask_u8: np.ndarray,
    clean_bgr: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
) -> np.ndarray:
    overlay = _raw_overlay_bgr(orig_bgr, mask_u8, p1, p2)
    return np.hstack([orig_bgr, _mask_bgr(mask_u8), clean_bgr, overlay])


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成 marker mask 与 clean 图")
    parser.add_argument(
        "--csv",
        type=Path,
        default=LABELS_DIR / "dataset_350-700.csv",
        help="标注表路径（相对 dataset/ 或绝对路径）",
    )
    parser.add_argument("--circle-radius", type=int, default=10, help="端点圆半径（像素）")
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=4,
        help="连线线宽（像素）",
    )
    parser.add_argument("--line-thickness-min", type=int, default=3)
    parser.add_argument("--line-thickness-max", type=int, default=5)
    parser.add_argument("--inpaint-radius", type=int, default=3, help="cv2.inpaint 邻域半径")
    parser.add_argument(
        "--inpaint",
        choices=("telea", "ns"),
        default="telea",
        help="inpaint 算法：telea 或 ns",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="不写入 clean_vis 检查图",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=LOG_DIR / "marker_clean.log",
        help="运行日志路径",
    )
    args = parser.parse_args()

    def resolve_csv(p: Path) -> Path:
        if p.is_absolute():
            return p.resolve()
        cand = (DATASET_DIR / p).resolve()
        if cand.is_file():
            return cand
        cand2 = (LABELS_DIR / p.name).resolve()
        if cand2.is_file():
            return cand2
        return cand

    csv_path = resolve_csv(args.csv)
    if not csv_path.is_file():
        print(f"找不到 CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_vis:
        CLEAN_VIS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = args.log.resolve() if args.log.is_absolute() else (DATASET_DIR / args.log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    inpaint_flag = cv2.INPAINT_TELEA if args.inpaint == "telea" else cv2.INPAINT_NS

    n_ok = 0
    n_skip = 0
    issues: list[str] = []

    def log_line(fp: TextIO, msg: str, echo: bool = False) -> None:
        fp.write(msg + "\n")
        fp.flush()
        if echo or msg.startswith("SKIP") or msg.startswith("WARN"):
            print(msg, file=sys.stderr if msg.startswith("SKIP") or msg.startswith("WARN") else sys.stdout)

    with log_path.open("w", encoding="utf-8") as logf:
        log_line(
            logf,
            f"# marker_clean run {datetime.now(timezone.utc).isoformat()} csv={csv_path}",
            echo=True,
        )

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = (row.get("image_id") or "").strip()
                rel = (row.get("image_path_raw_png") or "").strip()
                if not image_id or not rel:
                    msg = f"SKIP missing image_id or image_path_raw_png: {row!r}"
                    issues.append(msg)
                    log_line(logf, msg)
                    n_skip += 1
                    continue

                x1 = _parse_int(row.get("x1", ""))
                y1 = _parse_int(row.get("y1", ""))
                x2 = _parse_int(row.get("x2", ""))
                y2 = _parse_int(row.get("y2", ""))
                if None in (x1, y1, x2, y2):
                    msg = f"SKIP bad coords image_id={image_id}"
                    issues.append(msg)
                    log_line(logf, msg)
                    n_skip += 1
                    continue

                img_path = (DATASET_DIR / rel.replace("\\", "/")).resolve()

                if not img_path.is_file():
                    msg = f"SKIP missing image: {image_id} path={img_path}"
                    issues.append(msg)
                    log_line(logf, msg)
                    n_skip += 1
                    continue

                bgr = imread_unicode(img_path)
                if bgr is None:
                    msg = f"SKIP unreadable image: {image_id} path={img_path}"
                    issues.append(msg)
                    log_line(logf, msg)
                    n_skip += 1
                    continue

                h, w = bgr.shape[:2]
                oob = (
                    x1 < 0
                    or y1 < 0
                    or x2 < 0
                    or y2 < 0
                    or x1 >= w
                    or y1 >= h
                    or x2 >= w
                    or y2 >= h
                )
                cx1 = int(np.clip(x1, 0, w - 1))
                cy1 = int(np.clip(y1, 0, h - 1))
                cx2 = int(np.clip(x2, 0, w - 1))
                cy2 = int(np.clip(y2, 0, h - 1))
                if oob:
                    log_line(
                        logf,
                        f"WARN clipped coords image_id={image_id} ({x1},{y1})({x2},{y2}) -> ({cx1},{cy1})({cx2},{cy2}) size={w}x{h}",
                    )

                mask = build_marker_mask(
                    h,
                    w,
                    cx1,
                    cy1,
                    cx2,
                    cy2,
                    args.circle_radius,
                    args.line_thickness,
                    args.line_thickness_min,
                    args.line_thickness_max,
                )

                mask_path = MASK_DIR / f"{image_id}_mask.png"
                if not imwrite_unicode(mask_path, mask):
                    msg = f"SKIP mask write failed: {mask_path}"
                    issues.append(msg)
                    log_line(logf, msg)
                    n_skip += 1
                    continue

                clean = cv2.inpaint(bgr, mask, args.inpaint_radius, inpaint_flag)
                clean_path = CLEAN_DIR / f"{image_id}_clean.png"
                if not imwrite_unicode(clean_path, clean):
                    msg = f"SKIP clean write failed: {clean_path}"
                    issues.append(msg)
                    log_line(logf, msg)
                    n_skip += 1
                    continue

                if not args.no_vis:
                    vis = make_check_vis(bgr, mask, clean, (cx1, cy1), (cx2, cy2))
                    vis_path = CLEAN_VIS_DIR / f"{image_id}_vis.png"
                    if not imwrite_unicode(vis_path, vis):
                        log_line(logf, f"WARN vis write failed: {vis_path}")

                log_line(logf, f"OK {image_id}", echo=False)
                n_ok += 1

        log_line(logf, f"# done ok={n_ok} skip={n_skip}", echo=True)

    if issues:
        print(f"共 {len(issues)} 条异常/跳过，详见 {log_path}", file=sys.stderr)
    print(f"完成: mask={MASK_DIR} clean={CLEAN_DIR} clean_vis={CLEAN_VIS_DIR} log={log_path}")


if __name__ == "__main__":
    main()

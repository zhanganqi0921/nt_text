"""
训练前最终可视化抽查。

默认随机抽查 30 张样本，输出到 `final_check_vis/{set}/`。
每张检查图包含 6 个 panel：
1) raw image
2) clean image
3) train_clean image
4) marker mask
5) train_clean_resized + heatmap overlay
6) train_clean_resized + heatmap overlay + P1/P2 坐标点

输入：
- `labels/dataset_350-700.csv`
- `images_clean/{set}/`
- `images_train_clean/{set}/`
- `images_train_clean_resized/{set}/`
- `masks_marker/{set}/`
- `heatmaps/{target_size}/{image_id}.npy`（若不存在则从 CSV 动态生成）

输出：
- `final_check_vis/{set}/{image_id}_final_check.png`

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dataset_paths import (
    CLEAN_DIR,
    DATASET_350_700_CSV,
    DATASET_DIR,
    FINAL_CHECK_VIS_DIR,
    HEATMAP_DIR,
    LOG_DIR,
    MASK_DIR,
    TRAIN_CLEAN_DIR,
    TRAIN_CLEAN_RESIZED_DIR,
)
from generate_endpoint_heatmaps import heatmaps_from_row


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


def _parse_int(value: str) -> int | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return int(round(float(s)))
    except ValueError:
        return None


def _path_from_row(row: dict[str, str], field: str, fallback: Path) -> Path:
    rel = (row.get(field) or "").strip()
    return DATASET_DIR / rel if rel else fallback


def fit_to_panel(img: np.ndarray, panel_size: int) -> np.ndarray:
    """Letterbox any image to panel_size x panel_size for visual comparison."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(panel_size / w, panel_size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    x0 = (panel_size - nw) // 2
    y0 = (panel_size - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def annotate(panel: np.ndarray, title: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heatmap_to_color(heatmaps: np.ndarray) -> np.ndarray:
    """Colorize [2,H,W] heatmap: P1=green, P2=red."""
    if heatmaps.shape[0] != 2:
        raise ValueError(f"expected heatmap shape [2,H,W], got {heatmaps.shape}")
    p1 = np.clip(heatmaps[0], 0.0, 1.0)
    p2 = np.clip(heatmaps[1], 0.0, 1.0)
    color = np.zeros((heatmaps.shape[1], heatmaps.shape[2], 3), dtype=np.uint8)
    color[:, :, 1] = (p1 * 255).astype(np.uint8)
    color[:, :, 2] = (p2 * 255).astype(np.uint8)
    return color


def overlay_heatmaps(resized_bgr: np.ndarray, heatmaps: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    heat_color = heatmap_to_color(heatmaps)
    if heat_color.shape[:2] != resized_bgr.shape[:2]:
        heat_color = cv2.resize(heat_color, (resized_bgr.shape[1], resized_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    intensity = np.max(heat_color, axis=2)
    overlay = resized_bgr.copy()
    mask = intensity > 0
    blended = cv2.addWeighted(resized_bgr, 1.0 - alpha, heat_color, alpha, 0.0)
    overlay[mask] = blended[mask]
    return overlay


def draw_points(panel: np.ndarray, row: dict[str, str]) -> np.ndarray:
    out = panel.copy()
    x1 = _parse_int(row.get("x1_resized", ""))
    y1 = _parse_int(row.get("y1_resized", ""))
    x2 = _parse_int(row.get("x2_resized", ""))
    y2 = _parse_int(row.get("y2_resized", ""))
    if None in (x1, y1, x2, y2):
        return out

    h, w = out.shape[:2]
    p1 = (max(0, min(w - 1, x1)), max(0, min(h - 1, y1)))
    p2 = (max(0, min(w - 1, x2)), max(0, min(h - 1, y2)))
    cv2.line(out, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, p1, 6, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(out, p2, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(out, p1, 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, p2, 8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"P1 {p1}", (p1[0] + 8, max(18, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"P2 {p2}", (p2[0] + 8, min(h - 8, p2[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def load_heatmaps(row: dict[str, str], target_size: int, sigma: float) -> np.ndarray | None:
    image_id = (row.get("image_id") or "").strip()
    heatmap_path = HEATMAP_DIR / str(target_size) / f"{image_id}.npy"
    if heatmap_path.is_file():
        heatmaps = np.load(heatmap_path)
        if heatmaps.shape == (2, target_size, target_size):
            return heatmaps.astype(np.float32, copy=False)
    return heatmaps_from_row(row, target_size=target_size, sigma=sigma)


def make_final_check(row: dict[str, str], *, panel_size: int, target_size: int, sigma: float) -> np.ndarray | None:
    image_id = (row.get("image_id") or "").strip()
    if not image_id:
        return None

    raw_path = _path_from_row(row, "image_path_raw_png", DATASET_DIR / f"images_raw_png/2/{image_id}.png")
    clean_path = _path_from_row(row, "image_path_clean", CLEAN_DIR / f"{image_id}_clean.png")
    train_clean_path = _path_from_row(row, "image_path_train_clean", TRAIN_CLEAN_DIR / f"{image_id}_train_clean.png")
    resized_path = _path_from_row(
        row,
        "image_path_train_clean_resized",
        TRAIN_CLEAN_RESIZED_DIR / f"{image_id}_train_clean_resized.png",
    )
    marker_mask_path = MASK_DIR / f"{image_id}_mask.png"

    raw = imread_unicode(raw_path)
    clean = imread_unicode(clean_path)
    train_clean = imread_unicode(train_clean_path)
    resized = imread_unicode(resized_path)
    marker_mask = imread_unicode(marker_mask_path, cv2.IMREAD_GRAYSCALE)
    heatmaps = load_heatmaps(row, target_size=target_size, sigma=sigma)

    if any(x is None for x in (raw, clean, train_clean, resized, marker_mask)) or heatmaps is None:
        return None

    resized_panel = fit_to_panel(resized, panel_size)
    if resized_panel.shape[:2] != heatmaps.shape[1:]:
        heatmaps_for_panel = np.stack(
            [
                cv2.resize(heatmaps[0], (panel_size, panel_size), interpolation=cv2.INTER_LINEAR),
                cv2.resize(heatmaps[1], (panel_size, panel_size), interpolation=cv2.INTER_LINEAR),
            ],
            axis=0,
        )
    else:
        heatmaps_for_panel = heatmaps

    heat_overlay = overlay_heatmaps(resized_panel, heatmaps_for_panel)
    heat_points = draw_points(heat_overlay, row)

    panels = [
        annotate(fit_to_panel(raw, panel_size), "1 raw image"),
        annotate(fit_to_panel(clean, panel_size), "2 clean image"),
        annotate(fit_to_panel(train_clean, panel_size), "3 train_clean image"),
        annotate(fit_to_panel(marker_mask, panel_size), "4 marker mask"),
        annotate(heat_overlay, "5 resized + heatmap overlay"),
        annotate(heat_points, "6 P1/P2 coordinate points"),
    ]
    return np.hstack(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练前最终可视化抽查")
    parser.add_argument("--csv", type=Path, default=DATASET_350_700_CSV, help="训练 CSV 路径")
    parser.add_argument("--output-dir", type=Path, default=FINAL_CHECK_VIS_DIR, help="输出目录")
    parser.add_argument("--target-size", type=int, default=512, help="resize/heatmap 尺寸，默认 512")
    parser.add_argument("--panel-size", type=int, default=512, help="每个 panel 的显示尺寸，默认 512")
    parser.add_argument("--samples", type=int, default=30, help="随机抽查数量，默认 30")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，默认 0")
    parser.add_argument("--sigma", type=float, default=3.0, help="动态生成 heatmap 时使用的 sigma，默认 3")
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit(f"--samples 必须为正数，得到 {args.samples}")
    if args.target_size <= 0 or args.panel_size <= 0:
        raise SystemExit("--target-size 和 --panel-size 必须为正数")

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"未找到 CSV: {csv_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "generate_final_check_vis.log"

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if (row.get("image_id") or "").strip()]
    if not rows:
        raise SystemExit(f"CSV 中没有可抽查行: {csv_path}")

    rng = random.Random(args.seed)
    sample_rows = rng.sample(rows, min(args.samples, len(rows)))

    n_ok = 0
    n_skip = 0
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(
            f"# generate_final_check_vis run {datetime.now(timezone.utc).isoformat()} "
            f"samples={len(sample_rows)} target={args.target_size} output={output_dir}\n"
        )
        for row in sample_rows:
            image_id = (row.get("image_id") or "").strip()
            vis = make_final_check(row, panel_size=args.panel_size, target_size=args.target_size, sigma=args.sigma)
            if vis is None:
                n_skip += 1
                logf.write(f"SKIP {image_id}\n")
                continue
            out_path = output_dir / f"{image_id}_final_check.png"
            if not imwrite_unicode(out_path, vis):
                n_skip += 1
                logf.write(f"SKIP write_failed {out_path}\n")
                continue
            n_ok += 1
            logf.write(f"OK {image_id}\n")

        logf.write(f"# done ok={n_ok} skip={n_skip}\n")

    print(f"完成: final_check_vis={output_dir} ok={n_ok} skip={n_skip} log={log_path}")


if __name__ == "__main__":
    main()

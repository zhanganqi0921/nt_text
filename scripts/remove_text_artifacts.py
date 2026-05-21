"""
去除 clean image 中残留的设备信息、NT 字样和屏幕标尺，生成最终训练用图。

输入：配置数据子目录下的 images_clean/2/{image_id}_clean.png。
输出：images_train_clean/2/{image_id}_train_clean.png；
检查图：train_clean_vis/2/{image_id}_vis.png（clean | artifact mask | train clean | mask overlay）。

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dataset_paths import CLEAN_DIR, LOG_DIR, TRAIN_CLEAN_DIR, TRAIN_CLEAN_VIS_DIR


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


def image_id_from_clean_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_clean")] if stem.endswith("_clean") else stem


def build_edge_roi(h: int, w: int, top: float, bottom: float, left: float, right: float) -> np.ndarray:
    roi = np.zeros((h, w), dtype=np.uint8)
    if top > 0:
        roi[: int(round(h * top)), :] = 255
    if bottom > 0:
        roi[int(round(h * (1.0 - bottom))) :, :] = 255
    if left > 0:
        roi[:, : int(round(w * left))] = 255
    if right > 0:
        roi[:, int(round(w * (1.0 - right))) :] = 255
    return roi


def build_corner_roi(
    h: int,
    w: int,
    height: float,
    width: float,
    *,
    top_left: bool = False,
    top_right: bool = False,
    bottom_left: bool = False,
    bottom_right: bool = False,
) -> np.ndarray:
    roi = np.zeros((h, w), dtype=np.uint8)
    hh = max(0, min(h, int(round(h * height))))
    ww = max(0, min(w, int(round(w * width))))
    if hh == 0 or ww == 0:
        return roi
    if top_left:
        roi[:hh, :ww] = 255
    if top_right:
        roi[:hh, w - ww :] = 255
    if bottom_left:
        roi[h - hh :, :ww] = 255
    if bottom_right:
        roi[h - hh :, w - ww :] = 255
    return roi


def build_top_right_roi(h: int, w: int, height: float, width: float) -> np.ndarray:
    return build_corner_roi(h, w, height, width, top_right=True)


def build_all_corner_roi(h: int, w: int, height: float, width: float) -> np.ndarray:
    return build_corner_roi(
        h,
        w,
        height,
        width,
        top_left=True,
        top_right=True,
        bottom_left=True,
        bottom_right=True,
    )


def keep_text_like_components(binary: np.ndarray, *, max_area_ratio: float) -> np.ndarray:
    h, w = binary.shape[:2]
    max_area = int(h * w * max_area_ratio)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for idx in range(1, n):
        x, y, bw, bh, area = stats[idx]
        if area < 2 or area > max_area:
            continue
        # Keep small text, tick marks, and thin ruler lines; reject large anatomy-like blobs.
        thin = bw <= max(8, int(w * 0.015)) or bh <= max(8, int(h * 0.015))
        text_sized = bw <= int(w * 0.35) and bh <= int(h * 0.18)
        if thin or text_sized:
            out[labels == idx] = 255
    return out


def build_gray_text_mask(gray: np.ndarray, roi: np.ndarray, *, cluster_mode: str = "horizontal") -> np.ndarray:
    """Detect white/anti-aliased gray text strokes via local contrast.

    Text strokes are 1–3 px wide and much brighter than their immediate
    background, so a small-kernel top-hat plus an adaptive-threshold filter
    isolates them well. Fetal anatomy may produce similar local contrast on
    bony interfaces, so we additionally require (a) a mostly dark surrounding
    background and (b) text-like clustering — top/bottom labels line up in
    horizontal rows, while left-side device labels stack multiple short rows in
    a narrow vertical column. Isolated fetal speckle does neither.
    """
    if not np.any(roi):
        return np.zeros_like(gray)

    # 1) Small-kernel top-hat: highlights local bright peaks (text strokes).
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((7, 7), dtype=np.uint8))
    tophat_hit = cv2.inRange(top_hat, 25, 255)

    # 2) Adaptive threshold: pixels >= local mean + 25.
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, -25
    )

    text_candidate = cv2.bitwise_and(tophat_hit, adaptive)
    # Require a minimum absolute brightness so we do not chase dark fetal
    # speckle that happens to be locally brighter than its neighbours.
    text_candidate = cv2.bitwise_and(text_candidate, cv2.inRange(gray, 95, 255))
    text_candidate = cv2.bitwise_and(text_candidate, roi)

    # Require somewhat dark surroundings: fetal anatomy is on a continuously
    # gray background so its local dark-pixel ratio is low; device text sits
    # on or close to the black margin even when characters span a few px of
    # gray. The horizontal clustering filter below provides the stronger
    # protection against fetal speckle.
    dark_pixels = cv2.inRange(gray, 0, 70)
    dark_ratio = cv2.blur(dark_pixels, (25, 25))
    dark_bg = cv2.inRange(dark_ratio, int(round(255 * 0.30)), 255)
    text_candidate = cv2.bitwise_and(text_candidate, dark_bg)

    h, w = gray.shape[:2]
    if cluster_mode == "vertical":
        # Left-side device text is a stack of short horizontal rows. Link rows
        # vertically in a narrow column instead of requiring each row to be wide.
        clustered = cv2.dilate(
            text_candidate, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)), iterations=1
        )
    else:
        # Top/bottom labels are horizontal strings; require clusters that are
        # reasonably wide and significantly wider than tall.
        clustered = cv2.dilate(
            text_candidate, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)), iterations=1
        )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(clustered, connectivity=8)
    cluster_keep = np.zeros_like(text_candidate)
    min_cluster_width = max(28, int(round(w * 0.06)))
    min_cluster_height = max(18, int(round(h * 0.08)))
    for idx in range(1, n):
        x, y, bw, bh, area = stats[idx]
        if cluster_mode == "vertical":
            if bh < min_cluster_height:
                continue
            if bw > max(32, int(round(w * 0.16))):
                continue
        else:
            if bw < min_cluster_width:
                continue
            if bw < max(bh * 2, bh + 12):
                continue
        cluster_keep[labels == idx] = 255
    text_candidate = cv2.bitwise_and(text_candidate, cluster_keep)

    # Keep only thin/small components: fetal anatomy chunks are larger blobs
    # than text strokes even when they happen to satisfy local contrast.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(text_candidate, connectivity=8)
    out = np.zeros_like(text_candidate)
    max_dim = max(6, int(round(min(h, w) * 0.04)))
    max_area = int(h * w * 0.0015)
    for idx in range(1, n):
        x, y, bw, bh, area = stats[idx]
        if area < 2 or area > max_area:
            continue
        if min(bw, bh) > max_dim:
            continue
        out[labels == idx] = 255
    return out


def build_corner_text_mask(
    gray: np.ndarray,
    color_mask: np.ndarray,
    roi: np.ndarray,
    *,
    dilate: int,
) -> np.ndarray:
    # Color annotations (yellow NT/Pctl, green/blue device overlays) are highly
    # specific; we trust them anywhere they pass the strict HSV thresholds,
    # whether or not they sit on a black background.
    gray_text = build_gray_text_mask(gray, roi)
    mask = cv2.bitwise_or(color_mask, gray_text)
    mask = keep_text_like_components(mask, max_area_ratio=0.012)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def build_bottom_left_measure_mask(gray: np.ndarray) -> np.ndarray:
    """Catch tiny bottom-left measurement strings such as '+ Dist. 0.203 cm'."""
    h, w = gray.shape[:2]
    roi = np.zeros_like(gray)
    y0 = int(round(h * 0.92))
    x1 = int(round(w * 0.46))
    roi[y0:, :x1] = 255

    # These labels are low and small; top-hat alone can miss anti-aliased parts,
    # so combine absolute brightness with dark local context.
    bright = cv2.bitwise_and(cv2.inRange(gray, 105, 255), roi)
    dark_pixels = cv2.inRange(gray, 0, 75)
    dark_ratio = cv2.blur(dark_pixels, (21, 21))
    dark_context = cv2.inRange(dark_ratio, int(round(255 * 0.40)), 255)
    candidate = cv2.bitwise_and(bright, dark_context)

    linked = cv2.dilate(candidate, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 2)), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
    keep = np.zeros_like(gray)
    min_width = max(18, int(round(w * 0.045)))
    for idx in range(1, n):
        x, y, bw, bh, area = stats[idx]
        if bw < min_width:
            continue
        if bh > max(18, int(round(h * 0.12))):
            continue
        keep[labels == idx] = 255
    return cv2.bitwise_and(candidate, keep)


def build_top_right_text_mask(
    gray: np.ndarray,
    color_mask: np.ndarray,
    *,
    height: float,
    width: float,
    dilate: int,
) -> np.ndarray:
    roi = build_top_right_roi(gray.shape[0], gray.shape[1], height, width)
    return build_corner_text_mask(gray, color_mask, roi, dilate=dilate)


def build_artifact_mask(
    img_bgr: np.ndarray,
    *,
    edge_top: float,
    edge_bottom: float,
    edge_left: float,
    edge_right: float,
    dilate: int,
    corner_height: float,
    corner_width: float,
    corner_dilate: int,
    corners: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Colored ultrasound annotations: yellow NT text, green/blue device overlays.
    # Saturation low enough to catch anti-aliased character edges but well above
    # the near-zero saturation of grayscale anatomy.
    yellow = cv2.inRange(hsv, (15, 40, 80), (45, 255, 255))
    green = cv2.inRange(hsv, (45, 50, 70), (95, 255, 255))
    blue = cv2.inRange(hsv, (90, 60, 70), (135, 255, 255))
    color_mask = cv2.bitwise_or(yellow, cv2.bitwise_or(green, blue))

    tl, tr, bl, br = corners
    corner_roi = build_corner_roi(
        h,
        w,
        corner_height,
        corner_width,
        top_left=tl,
        top_right=tr,
        bottom_left=bl,
        bottom_right=br,
    )
    edge_roi = build_edge_roi(h, w, edge_top, edge_bottom, 0.0, edge_right)
    left_roi = build_edge_roi(h, w, 0.0, 0.0, edge_left, 0.0)
    text_roi = cv2.bitwise_or(corner_roi, edge_roi)

    # Top/bottom and corner labels are mostly horizontal; left device labels
    # are often a narrow vertical stack of short rows, so detect that separately.
    horizontal_text = build_gray_text_mask(gray, text_roi, cluster_mode="horizontal")
    left_text = build_gray_text_mask(gray, left_roi, cluster_mode="vertical")
    bottom_left_measure = build_bottom_left_measure_mask(gray)
    gray_text = cv2.bitwise_or(cv2.bitwise_or(horizontal_text, left_text), bottom_left_measure)
    corner_text = cv2.bitwise_or(color_mask, gray_text)
    corner_text = keep_text_like_components(corner_text, max_area_ratio=0.012)
    corner_text = cv2.morphologyEx(corner_text, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if corner_dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * corner_dilate + 1, 2 * corner_dilate + 1))
        corner_text = cv2.dilate(corner_text, kernel, iterations=1)

    mask = cv2.bitwise_or(color_mask, corner_text)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if dilate > 0:
        dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, dkernel, iterations=1)
    return mask


def remove_artifacts(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int,
) -> np.ndarray:
    cleaned = cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Text on black margins is cleaner when set back to black instead of hallucinated by inpaint.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    black_pixels = cv2.inRange(gray, 0, 55)
    black_ratio = cv2.blur(black_pixels, (21, 21))
    black_context = cv2.inRange(black_ratio, 170, 255)
    black_mask = cv2.bitwise_and(mask, black_context)
    cleaned[black_mask > 0] = (0, 0, 0)
    return cleaned


def make_vis(img_bgr: np.ndarray, mask: np.ndarray, cleaned_bgr: np.ndarray) -> np.ndarray:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = mask
    overlay = cv2.addWeighted(img_bgr, 0.75, red, 0.25, 0.0)
    return np.hstack([img_bgr, mask_bgr, cleaned_bgr, overlay])


def main() -> None:
    parser = argparse.ArgumentParser(description="去除设备文字和 NT 信息，生成最终训练 clean 图")
    parser.add_argument("--input-dir", type=Path, default=CLEAN_DIR, help="输入 clean image 目录")
    parser.add_argument("--output-dir", type=Path, default=TRAIN_CLEAN_DIR, help="输出训练 clean image 目录")
    parser.add_argument("--vis-dir", type=Path, default=TRAIN_CLEAN_VIS_DIR, help="输出复查图目录")
    parser.add_argument("--edge-top", type=float, default=0.10, help="顶部边缘文字检测带高度比例（黑色背景过滤保护胎儿）")
    parser.add_argument("--edge-bottom", type=float, default=0.10, help="底部边缘文字检测带高度比例（黑色背景过滤保护胎儿）")
    parser.add_argument("--edge-left", type=float, default=0.08, help="左侧边缘文字检测带宽度比例")
    parser.add_argument("--edge-right", type=float, default=0.0, help="右侧边缘文字检测带宽度比例")
    parser.add_argument("--dilate", type=int, default=2, help="artifact mask 膨胀半径")
    parser.add_argument("--corner-height", type=float, default=0.22, help="四角专用检测区域高度比例")
    parser.add_argument("--corner-width", type=float, default=0.35, help="四角专用检测区域宽度比例")
    parser.add_argument("--corner-dilate", type=int, default=3, help="四角文字 mask 额外膨胀半径")
    parser.add_argument("--no-top-left", action="store_true", help="不检测左上角文字")
    parser.add_argument("--no-top-right", action="store_true", help="不检测右上角文字")
    parser.add_argument("--no-bottom-left", action="store_true", help="不检测左下角文字")
    parser.add_argument("--no-bottom-right", action="store_true", help="不检测右下角文字")
    parser.add_argument("--inpaint-radius", type=int, default=3, help="cv2.inpaint 半径")
    parser.add_argument("--no-vis", action="store_true", help="不生成 train_clean_vis 复查图")
    args = parser.parse_args()

    corners = (
        not args.no_top_left,
        not args.no_top_right,
        not args.no_bottom_left,
        not args.no_bottom_right,
    )

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    vis_dir = args.vis_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"未找到输入目录: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "remove_text_artifacts.log"

    n_ok = 0
    n_skip = 0
    clean_paths = sorted(input_dir.glob("*_clean.png"))
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# remove_text_artifacts run {datetime.now(timezone.utc).isoformat()} input={input_dir}\n")
        for clean_path in clean_paths:
            image_id = image_id_from_clean_path(clean_path)
            img = imread_unicode(clean_path)
            if img is None:
                n_skip += 1
                logf.write(f"SKIP unreadable {clean_path}\n")
                continue

            mask = build_artifact_mask(
                img,
                edge_top=args.edge_top,
                edge_bottom=args.edge_bottom,
                edge_left=args.edge_left,
                edge_right=args.edge_right,
                dilate=args.dilate,
                corner_height=args.corner_height,
                corner_width=args.corner_width,
                corner_dilate=args.corner_dilate,
                corners=corners,
            )
            train_clean = remove_artifacts(
                img,
                mask,
                args.inpaint_radius,
            )

            out_path = output_dir / f"{image_id}_train_clean.png"
            if not imwrite_unicode(out_path, train_clean):
                n_skip += 1
                logf.write(f"SKIP write_failed {out_path}\n")
                continue

            if not args.no_vis:
                vis = make_vis(img, mask, train_clean)
                vis_path = vis_dir / f"{image_id}_vis.png"
                if not imwrite_unicode(vis_path, vis):
                    logf.write(f"WARN vis_write_failed {vis_path}\n")

            n_ok += 1
            logf.write(f"OK {image_id}\n")
        logf.write(f"# done ok={n_ok} skip={n_skip}\n")

    print(f"完成: train_clean={output_dir} train_clean_vis={vis_dir} ok={n_ok} skip={n_skip} log={log_path}")


if __name__ == "__main__":
    main()


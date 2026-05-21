"""
按指定尺寸对 train_clean 图进行 Letterbox/Stretch resize，并同步端点坐标。

主要功能：
1) 读取配置的数据根目录下的训练 CSV（默认 `dataset_350-700.csv`），逐行处理；
2) 读取 `images_train_clean/{set}/{image_id}_train_clean.png`，按目标尺寸做 resize；
3) Letterbox（默认）：保持纵横比，按最长边缩放，剩余区域补黑边，输出方形；
4) Stretch：直接拉伸到目标尺寸，纵横比可能变形；
5) 同步端点坐标到 resize 后图像坐标系；
6) 在原 CSV 上新增列写回。

输入：
- 训练 CSV：`labels/dataset_350-700.csv`（含 `image_path_train_clean`、`width`、`height`、`x1..y2`）
- 训练 clean 图：`images_train_clean/{set}/`

输出：
- Resize 图目录：`images_train_clean_resized/{set}/{image_id}_train_clean_resized.png`
- 更新后的训练 CSV，新增列：
  `image_path_train_clean_resized`、`resize_target`、`resize_mode`、`resize_scale`、
  `pad_x`、`pad_y`、`x1_resized`、`y1_resized`、`x2_resized`、`y2_resized`
注意事项：
- 不修改原 train_clean 图；resize 结果写到新目录。
- 端点坐标会按 letterbox 公式同步并 clamp 到 `[0, target_size - 1]`。
- 多次以不同 `--target-size` 运行时，CSV 中相应列会被新值覆盖。

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dataset_paths import (
    DATASET_350_700_CSV,
    DATASET_DIR,
    LOG_DIR,
    TRAIN_CLEAN_DIR,
    TRAIN_CLEAN_RESIZED_DIR,
)


NEW_FIELDS = [
    "image_path_train_clean_resized",
    "resize_target",
    "resize_mode",
    "resize_scale",
    "pad_x",
    "pad_y",
    "x1_resized",
    "y1_resized",
    "x2_resized",
    "y2_resized",
]


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


def _parse_float(value: str) -> float | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _relative_to_dataset(path: Path) -> str:
    return path.resolve().relative_to(DATASET_DIR.resolve()).as_posix()


def letterbox_params(src_w: int, src_h: int, target: int) -> tuple[float, int, int, int, int]:
    """Return (scale, new_w, new_h, pad_x, pad_y) for a letterbox resize."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source size: {src_w}x{src_h}")
    scale = min(target / src_w, target / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def resize_letterbox(img: np.ndarray, target: int) -> tuple[np.ndarray, float, int, int]:
    """Resize to target×target with letterbox padding; return (out, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale, new_w, new_h, pad_x, pad_y = letterbox_params(w, h, target)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        out = np.zeros((target, target, img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((target, target), dtype=img.dtype)
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return out, scale, pad_x, pad_y


def resize_stretch(img: np.ndarray, target: int) -> tuple[np.ndarray, tuple[float, float], int, int]:
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid source size: {w}x{h}")
    sx = target / w
    sy = target / h
    out = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    return out, (sx, sy), 0, 0


def transform_point(
    x: float,
    y: float,
    *,
    mode: str,
    scale: float | tuple[float, float],
    pad_x: int,
    pad_y: int,
    target: int,
) -> tuple[int, int]:
    if mode == "letterbox":
        assert isinstance(scale, float)
        nx = x * scale + pad_x
        ny = y * scale + pad_y
    else:
        sx, sy = scale  # type: ignore[misc]
        nx = x * sx
        ny = y * sy
    nx_i = int(round(max(0, min(target - 1, nx))))
    ny_i = int(round(max(0, min(target - 1, ny))))
    return nx_i, ny_i


def update_csv(
    csv_path: Path,
    updates: dict[str, dict[str, str]],
) -> tuple[list[str], int]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    new_fields = [f for f in NEW_FIELDS if f not in existing_fields]
    final_fields = existing_fields + new_fields

    updated = 0
    for row in rows:
        image_id = (row.get("image_id") or "").strip()
        upd = updates.get(image_id)
        if upd is None:
            for f in NEW_FIELDS:
                row.setdefault(f, "")
            continue
        for f in NEW_FIELDS:
            row[f] = upd.get(f, "")
        updated += 1

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(rows)

    return final_fields, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Letterbox/Stretch resize train_clean 图并同步端点坐标")
    parser.add_argument("--csv", type=Path, default=DATASET_350_700_CSV, help="训练 CSV 路径")
    parser.add_argument("--input-dir", type=Path, default=TRAIN_CLEAN_DIR, help="train_clean 图输入目录")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRAIN_CLEAN_RESIZED_DIR,
        help="resize 后图输出目录",
    )
    parser.add_argument("--target-size", type=int, default=512, help="目标边长（默认 512）")
    parser.add_argument(
        "--mode",
        choices=["letterbox", "stretch"],
        default="letterbox",
        help="resize 模式：letterbox（默认，保持纵横比补黑边）或 stretch（直接拉伸）",
    )
    parser.add_argument("--no-csv-update", action="store_true", help="不写回 CSV，仅生成图片")
    args = parser.parse_args()

    target = int(args.target_size)
    if target <= 0:
        raise SystemExit(f"--target-size 必须为正数，得到 {target}")
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"未找到训练 CSV: {csv_path}")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "resize_train_clean.log"

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"训练 CSV 为空: {csv_path}")

    updates: dict[str, dict[str, str]] = {}
    n_ok = 0
    n_skip_image = 0
    n_skip_coord = 0

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(
            f"# resize_train_clean run {datetime.now(timezone.utc).isoformat()} "
            f"target={target} mode={args.mode} input={input_dir} output={output_dir}\n"
        )
        for row in rows:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                logf.write("SKIP no_image_id\n")
                continue

            rel_tc = (row.get("image_path_train_clean") or "").strip()
            tc_path = (DATASET_DIR / rel_tc) if rel_tc else (input_dir / f"{image_id}_train_clean.png")
            img = imread_unicode(tc_path)
            if img is None:
                n_skip_image += 1
                logf.write(f"SKIP unreadable {image_id} -> {tc_path}\n")
                continue

            try:
                if args.mode == "letterbox":
                    resized, scale, pad_x, pad_y = resize_letterbox(img, target)
                    scale_repr = f"{scale:.8f}"
                else:
                    resized, (sx, sy), pad_x, pad_y = resize_stretch(img, target)
                    scale = (sx, sy)
                    scale_repr = f"{sx:.8f},{sy:.8f}"
            except ValueError as e:
                n_skip_image += 1
                logf.write(f"SKIP bad_size {image_id} {e}\n")
                continue

            out_path = output_dir / f"{image_id}_train_clean_resized.png"
            if not imwrite_unicode(out_path, resized):
                n_skip_image += 1
                logf.write(f"SKIP write_failed {out_path}\n")
                continue

            x1 = _parse_float(row.get("x1", ""))
            y1 = _parse_float(row.get("y1", ""))
            x2 = _parse_float(row.get("x2", ""))
            y2 = _parse_float(row.get("y2", ""))
            if None in (x1, y1, x2, y2):
                n_skip_coord += 1
                logf.write(f"WARN missing_coord {image_id}\n")
                p1 = p2 = None
                coord_strs = {"x1_resized": "", "y1_resized": "", "x2_resized": "", "y2_resized": ""}
            else:
                p1 = transform_point(
                    x1, y1, mode=args.mode, scale=scale, pad_x=pad_x, pad_y=pad_y, target=target
                )
                p2 = transform_point(
                    x2, y2, mode=args.mode, scale=scale, pad_x=pad_x, pad_y=pad_y, target=target
                )
                coord_strs = {
                    "x1_resized": str(p1[0]),
                    "y1_resized": str(p1[1]),
                    "x2_resized": str(p2[0]),
                    "y2_resized": str(p2[1]),
                }

            updates[image_id] = {
                "image_path_train_clean_resized": _relative_to_dataset(out_path),
                "resize_target": str(target),
                "resize_mode": args.mode,
                "resize_scale": scale_repr,
                "pad_x": str(pad_x),
                "pad_y": str(pad_y),
                **coord_strs,
            }

            n_ok += 1
            logf.write(f"OK {image_id}\n")

        logf.write(
            f"# done ok={n_ok} skip_image={n_skip_image} skip_coord={n_skip_coord}\n"
        )

    csv_updated_count = 0
    if not args.no_csv_update and updates:
        _, csv_updated_count = update_csv(csv_path, updates)

    print(
        "完成: "
        f"target={target} mode={args.mode} "
        f"resized={output_dir} "
        f"ok={n_ok} skip_image={n_skip_image} skip_coord={n_skip_coord} "
        f"csv_updated={csv_updated_count} log={log_path}"
    )


if __name__ == "__main__":
    main()

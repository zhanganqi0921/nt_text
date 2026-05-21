"""
生成 P1/P2 两通道 Gaussian heatmap，用于 U-Net-Heatmap / 关键点定位训练。

主要功能：
1) 从 `dataset_350-700.csv` 读取 resize 后端点坐标：
   `x1_resized,y1_resized,x2_resized,y2_resized`；
2) 为每张 512x512（或指定尺寸）图生成 shape = [2, H, W] 的 heatmap；
3) 可在 PyTorch Dataset 中 import `make_endpoint_heatmaps()` 动态生成；
4) 也可批量保存 `.npy` 到 `heatmaps/{target_size}/{image_id}.npy`。

默认参数：
- target-size: 512
- sigma: 3.0
- dtype: float32

注意事项：
- 坐标会 clip 到 `[0, target_size - 1]`。
- 不修改原 train_clean 或 resized 图像。
- 若 CSV 缺少 resize 后坐标或目标尺寸不匹配，会跳过并记录日志。

路径配置：
- 数据根目录与 0/1/2 子集在 `scripts/dataset_paths.py` 中修改。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dataset_paths import DATASET_350_700_CSV, HEATMAP_DIR, LOG_DIR


RESIZED_COORD_FIELDS = ["x1_resized", "y1_resized", "x2_resized", "y2_resized"]


def _parse_float(value: str) -> float | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def clip_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Clip a point into image bounds, keeping float precision for Gaussian center."""
    cx = max(0.0, min(float(width - 1), float(x)))
    cy = max(0.0, min(float(height - 1), float(y)))
    return cx, cy


def gaussian_heatmap(
    x: float,
    y: float,
    *,
    height: int,
    width: int,
    sigma: float,
    truncate: float = 3.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Generate a single 2D Gaussian heatmap with peak value 1.0."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid heatmap size: {width}x{height}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    cx, cy = clip_point(x, y, width, height)
    heatmap = np.zeros((height, width), dtype=dtype)

    radius = max(1, int(round(truncate * sigma)))
    x0 = max(0, int(np.floor(cx)) - radius)
    x1 = min(width - 1, int(np.ceil(cx)) + radius)
    y0 = max(0, int(np.floor(cy)) - radius)
    y1 = min(height - 1, int(np.ceil(cy)) + radius)

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    patch = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(dtype)
    heatmap[y0 : y1 + 1, x0 : x1 + 1] = patch
    return heatmap


def make_endpoint_heatmaps(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    height: int = 512,
    width: int = 512,
    sigma: float = 3.0,
    truncate: float = 3.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return heatmaps with shape [2, H, W], channel 0=P1 and channel 1=P2."""
    p1 = gaussian_heatmap(x1, y1, height=height, width=width, sigma=sigma, truncate=truncate, dtype=dtype)
    p2 = gaussian_heatmap(x2, y2, height=height, width=width, sigma=sigma, truncate=truncate, dtype=dtype)
    return np.stack([p1, p2], axis=0).astype(dtype, copy=False)


def heatmaps_from_row(
    row: dict[str, str],
    *,
    target_size: int = 512,
    sigma: float = 3.0,
    truncate: float = 3.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray | None:
    """Create [2, target_size, target_size] heatmaps from one CSV row."""
    coords = [_parse_float(row.get(field, "")) for field in RESIZED_COORD_FIELDS]
    if any(v is None for v in coords):
        return None
    x1, y1, x2, y2 = (float(v) for v in coords if v is not None)
    return make_endpoint_heatmaps(
        x1,
        y1,
        x2,
        y2,
        height=target_size,
        width=target_size,
        sigma=sigma,
        truncate=truncate,
        dtype=dtype,
    )


def _row_matches_target(row: dict[str, str], target_size: int) -> bool:
    resize_target = (row.get("resize_target") or "").strip()
    return not resize_target or resize_target == str(target_size)


def save_heatmaps(
    *,
    csv_path: Path,
    output_dir: Path,
    target_size: int,
    sigma: float,
    truncate: float,
) -> tuple[int, int, int]:
    """Save .npy heatmaps. Returns (ok, skipped_bad_coord, skipped_target_mismatch)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "generate_endpoint_heatmaps.log"

    ok = 0
    skipped_bad_coord = 0
    skipped_target_mismatch = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f, log_path.open("w", encoding="utf-8") as logf:
        reader = csv.DictReader(f)
        missing_fields = [field for field in RESIZED_COORD_FIELDS if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise SystemExit(f"CSV 缺少字段: {missing_fields}. 请先运行 resize_train_clean.py")

        logf.write(
            f"# generate_endpoint_heatmaps run {datetime.now(timezone.utc).isoformat()} "
            f"target={target_size} sigma={sigma} output={output_dir}\n"
        )
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                skipped_bad_coord += 1
                logf.write("SKIP no_image_id\n")
                continue
            if not _row_matches_target(row, target_size):
                skipped_target_mismatch += 1
                logf.write(
                    f"SKIP target_mismatch {image_id} csv_target={row.get('resize_target', '')} "
                    f"requested={target_size}\n"
                )
                continue

            heatmaps = heatmaps_from_row(row, target_size=target_size, sigma=sigma, truncate=truncate)
            if heatmaps is None:
                skipped_bad_coord += 1
                logf.write(f"SKIP bad_coord {image_id}\n")
                continue

            out_path = output_dir / f"{image_id}.npy"
            np.save(out_path, heatmaps)
            ok += 1
            logf.write(f"OK {image_id}\n")

        logf.write(
            f"# done ok={ok} skipped_bad_coord={skipped_bad_coord} "
            f"skipped_target_mismatch={skipped_target_mismatch}\n"
        )

    return ok, skipped_bad_coord, skipped_target_mismatch


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 P1/P2 两通道 Gaussian endpoint heatmap")
    parser.add_argument("--csv", type=Path, default=DATASET_350_700_CSV, help="训练 CSV 路径")
    parser.add_argument("--target-size", type=int, default=512, help="heatmap 边长（默认 512）")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma（默认 3）")
    parser.add_argument("--truncate", type=float, default=3.0, help="Gaussian 截断半径倍数（默认 3）")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录，默认 heatmaps/{target-size}")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只验证 CSV 和生成函数，不保存 .npy",
    )
    args = parser.parse_args()

    target_size = int(args.target_size)
    if target_size <= 0:
        raise SystemExit(f"--target-size 必须为正数，得到 {target_size}")
    if args.sigma <= 0:
        raise SystemExit(f"--sigma 必须为正数，得到 {args.sigma}")
    if args.truncate <= 0:
        raise SystemExit(f"--truncate 必须为正数，得到 {args.truncate}")

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"未找到 CSV: {csv_path}")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else HEATMAP_DIR / str(target_size)

    if args.no_save:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
        if row is None:
            raise SystemExit(f"CSV 为空: {csv_path}")
        heatmaps = heatmaps_from_row(row, target_size=target_size, sigma=args.sigma, truncate=args.truncate)
        if heatmaps is None:
            raise SystemExit("CSV 第一行缺少有效 resize 后端点坐标")
        print(
            f"验证通过: shape={heatmaps.shape} dtype={heatmaps.dtype} "
            f"min={heatmaps.min():.6f} max={heatmaps.max():.6f}"
        )
        return

    ok, skipped_bad_coord, skipped_target_mismatch = save_heatmaps(
        csv_path=csv_path,
        output_dir=output_dir,
        target_size=target_size,
        sigma=args.sigma,
        truncate=args.truncate,
    )
    print(
        f"完成: heatmaps={output_dir} ok={ok} "
        f"skipped_bad_coord={skipped_bad_coord} skipped_target_mismatch={skipped_target_mismatch}"
    )


if __name__ == "__main__":
    main()

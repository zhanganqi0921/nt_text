"""
端点标注主流程脚本：调用 LabelMe 标注，并将结果标准化为 CSV 与复查图。

整体流程：
1) 打开 `dataset/images_raw_png` 进行标注（可用 `--sync-only` 跳过此步骤）；
2) 将 LabelMe 可能落在图片目录下的 JSON 统一归档到 `dataset/labels/labelme_json`；
3) 解析 JSON 中的两端点（支持 `P1/P2` point 或单条两点 line/linestrip）；
4) 写入/更新 `dataset/labels/endpoints.csv`；
5) 同步生成 `dataset/images_labeled_vis/{image_id}_labeled.png` 用于人工复查。

输入与输出：
- 输入图像目录：`dataset/images_raw_png`
- 输入标注目录：`dataset/labels/labelme_json`
- 输出标注表：`dataset/labels/endpoints.csv`
- 输出复查图：`dataset/images_labeled_vis/*_labeled.png`

关键约定：
- 优先读取标签为 `P1` / `P2` 的 point；
- 若无显式标签，可回退为“恰好两个未命名 point”；
- 若不满足两端点规则，记录为跳过并在汇总中按原因分组提示。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

DATASET_DIR = Path(__file__).resolve().parents[1]
PNG_DIR = DATASET_DIR / "images_raw_png"
VIS_DIR = DATASET_DIR / "images_labeled_vis"
LABELS_DIR = DATASET_DIR / "labels"
LABELME_JSON_DIR = LABELS_DIR / "labelme_json"
ENDPOINTS_CSV = LABELS_DIR / "endpoints.csv"

FIELDNAMES = [
    "image_id",
    "image_path",
    "x1",
    "y1",
    "x2",
    "y2",
    "labeled",
]


def imread_unicode(path: Path) -> np.ndarray | None:
    """Windows 下支持含中文路径的图像读取。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path: Path, img_bgr: np.ndarray) -> bool:
    """Windows 下支持含中文路径的 PNG 写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok or buf is None:
        return False
    buf.tofile(str(path))
    return True


def load_endpoints() -> dict[str, dict[str, Any]]:
    """image_id -> 行字典（含整数坐标）。"""
    if not ENDPOINTS_CSV.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with ENDPOINTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        for row in reader:
            iid = (row.get("image_id") or "").strip()
            if not iid:
                continue
            try:
                labeled = int(row.get("labeled", "0") or "0")
            except ValueError:
                labeled = 0
            out[iid] = {
                "image_id": iid,
                "image_path": (row.get("image_path") or "").strip(),
                "x1": row.get("x1", ""),
                "y1": row.get("y1", ""),
                "x2": row.get("x2", ""),
                "y2": row.get("y2", ""),
                "labeled": labeled,
            }
    return out


def write_endpoints(rows_by_id: dict[str, dict[str, Any]]) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows_by_id.values(), key=lambda r: r["image_id"])
    with ENDPOINTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            line = {
                "image_id": r["image_id"],
                "image_path": r.get("image_path", ""),
                "x1": str(r.get("x1", "")),
                "y1": str(r.get("y1", "")),
                "x2": str(r.get("x2", "")),
                "y2": str(r.get("y2", "")),
                "labeled": str(int(r.get("labeled", 0))),
            }
            w.writerow(line)


def draw_labeled_review_image(
    base_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """
    用于 images_labeled_vis 的复查图：P1/P2、连线、坐标文字（无交互提示条）。
    """
    vis = base_bgr.copy()
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.line(vis, p1, p2, (0, 200, 0), 2, cv2.LINE_AA)
    items: list[tuple[tuple[int, int], str, tuple[int, int, int], tuple[int, int]]] = [
        (p1, "P1", (0, 255, 255), (12, -8)),
        (p2, "P2", (255, 0, 255), (12, 28)),
    ]
    for (px, py), tag, color, (ox, oy) in items:
        cv2.circle(vis, (px, py), 8, color, -1)
        cv2.circle(vis, (px, py), 9, (0, 0, 0), 1)
        text = f"{tag} ({px},{py})"
        cv2.putText(
            vis,
            text,
            (px + ox, py + oy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


def write_labeled_review_png(
    image_id: str,
    base_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Path:
    """写入 dataset/images_labeled_vis/{image_id}_labeled.png。"""
    out_path = VIS_DIR / f"{image_id}_labeled.png"
    vis = draw_labeled_review_image(base_bgr, x1, y1, x2, y2)
    if not imwrite_unicode(out_path, vis):
        raise OSError(f"无法写入复查图: {out_path}")
    return out_path


def draw_overlay(
    base_bgr: np.ndarray,
    points: list[tuple[int, int]],
) -> np.ndarray:
    vis = base_bgr.copy()
    if len(points) == 2:
        cv2.line(vis, points[0], points[1], (0, 200, 0), 2, cv2.LINE_AA)
    labels = ["P1", "P2"]
    for i, (px, py) in enumerate(points):
        cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
        cv2.circle(vis, (px, py), 7, (0, 0, 0), 1)
        tag = labels[i] if i < len(labels) else f"P{i + 1}"
        text = f"{tag} ({px},{py})"
        cv2.putText(
            vis,
            text,
            (px + 10, py - 10 - i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    help_lines = [
        "LClick: point | s:save | r:reset | n:skip | q:quit",
    ]
    y0 = vis.shape[0] - 28
    for j, h in enumerate(help_lines):
        cv2.putText(
            vis,
            h,
            (8, y0 + j * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return vis


def label_one_image(
    png_path: Path,
    rows_by_id: dict[str, dict[str, Any]],
) -> bool:
    """
    交互标注一张图。返回 True 表示用户按 q 退出，False 表示继续下一张。
    """
    image_id = png_path.stem
    rel_path = png_path.relative_to(DATASET_DIR).as_posix()
    bgr = imread_unicode(png_path)
    if bgr is None:
        print(f"无法读取图像，跳过: {png_path}")
        return False

    points: list[tuple[int, int]] = []
    win = "NT endpoint labeling"

    def redraw() -> None:
        frame = draw_overlay(bgr, points)
        cv2.imshow(win, frame)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(points) >= 2:
            return
        points.append((int(x), int(y)))
        redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(win)
            return True
        if key == ord("r"):
            points.clear()
            redraw()
        elif key == ord("s"):
            if len(points) != 2:
                print("需要恰好 2 个端点后再按 s 保存。")
                continue
            (x1, y1), (x2, y2) = points
            rows_by_id[image_id] = {
                "image_id": image_id,
                "image_path": rel_path,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "labeled": 1,
            }
            write_endpoints(rows_by_id)
            out_vis = write_labeled_review_png(image_id, bgr, x1, y1, x2, y2)
            print(f"已保存: {image_id} | 复查图: {out_vis.relative_to(DATASET_DIR).as_posix()}")
            cv2.destroyWindow(win)
            return False
        elif key == ord("n"):
            cv2.destroyWindow(win)
            return False


def _to_xy(pt: Any) -> tuple[int, int]:
    if not isinstance(pt, list) or len(pt) < 2:
        raise ValueError("point 格式错误")
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _extract_points_from_shapes(shapes: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    p1: tuple[int, int] | None = None
    p2: tuple[int, int] | None = None
    unlabeled_points: list[tuple[int, int]] = []

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip().lower()
        shape_type = str(shape.get("shape_type", "")).strip().lower()
        points = shape.get("points")
        if not isinstance(points, list) or not points:
            continue

        if shape_type == "point":
            xy = _to_xy(points[0])
            if label == "p1":
                p1 = xy
            elif label == "p2":
                p2 = xy
            else:
                unlabeled_points.append(xy)
            continue

        if shape_type in {"line", "linestrip"} and len(points) >= 2 and p1 is None and p2 is None:
            p1 = _to_xy(points[0])
            p2 = _to_xy(points[1])

    if p1 is None and p2 is None and len(unlabeled_points) == 2:
        p1 = unlabeled_points[0]
        p2 = unlabeled_points[1]

    if p1 is None or p2 is None:
        raise ValueError("未找到两端点，请使用 P1/P2 两个 point 或一条两点线段")
    return p1[0], p1[1], p2[0], p2[1]


def _resolve_image_path(json_path: Path, payload: dict[str, Any]) -> Path:
    raw = str(payload.get("imagePath", "")).strip()
    if raw:
        p = Path(raw)
        candidates = [p]
        if not p.is_absolute():
            candidates = [json_path.parent / p, DATASET_DIR / p, PNG_DIR / p.name]
        for c in candidates:
            if c.is_file():
                return c.resolve()
    fallback = PNG_DIR / f"{json_path.stem}.png"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"找不到原图: {json_path}")


def parse_labelme_json(json_path: Path) -> tuple[str, str, int, int, int, int]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("JSON 缺少 shapes")
    x1, y1, x2, y2 = _extract_points_from_shapes(shapes)
    image_path_abs = _resolve_image_path(json_path, payload)
    return (
        image_path_abs.stem,
        image_path_abs.relative_to(DATASET_DIR).as_posix(),
        x1,
        y1,
        x2,
        y2,
    )


def collect_json_files() -> list[Path]:
    return sorted(LABELME_JSON_DIR.glob("*.json"))


def relocate_labelme_json_from_png_dir() -> int:
    """
    LabelMe 默认会把 JSON 写到图片同目录，这里统一搬运到 labels/labelme_json。
    """
    moved = 0
    LABELME_JSON_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(PNG_DIR.glob("*.json")):
        dst = LABELME_JSON_DIR / src.name
        shutil.move(str(src), str(dst))
        moved += 1
    return moved


def _print_sync_issue_report(issues: list[tuple[str, str]]) -> None:
    if not issues:
        return
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, reason in issues:
        grouped[reason].append(name)
    print("以下 JSON 需要回到 LabelMe 检查：")
    for reason, names in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"- {reason} ({len(names)} 个)")
        for n in sorted(names):
            print(f"  - {n}")


def sync_from_labelme_json() -> tuple[int, int, list[tuple[str, str]]]:
    rows_by_id = load_endpoints()
    n_ok = 0
    n_bad = 0
    issues: list[tuple[str, str]] = []
    for json_path in collect_json_files():
        try:
            image_id, rel_path, x1, y1, x2, y2 = parse_labelme_json(json_path)
            rows_by_id[image_id] = {
                "image_id": image_id,
                "image_path": rel_path,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "labeled": 1,
            }
            bgr = imread_unicode(DATASET_DIR / rel_path)
            if bgr is not None:
                write_labeled_review_png(image_id, bgr, x1, y1, x2, y2)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_bad += 1
            reason = str(e).strip() or type(e).__name__
            issues.append((json_path.name, reason))
            print(f"跳过 JSON: {json_path.name} ({reason})")
    write_endpoints(rows_by_id)
    return n_ok, n_bad, issues


def launch_labelme() -> None:
    if not PNG_DIR.is_dir():
        raise SystemExit(f"未找到目录: {PNG_DIR}")
    print("即将打开 LabelMe，关闭后会自动同步 JSON -> endpoints.csv")
    print(f"JSON 将统一放到: {LABELME_JSON_DIR}")
    print("建议标注方式：P1/P2 两个 point，或一条两点 line/linestrip。")
    try:
        # 优先使用当前 Python 解释器启动，避免 Windows 下 PATH 未包含 Scripts 导致找不到 labelme 命令。
        subprocess.run(
            [
                sys.executable,
                "-m",
                "labelme",
                str(PNG_DIR),
                "--output",
                str(LABELME_JSON_DIR),
                "--autosave",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            "启动 LabelMe 失败。请确认当前解释器已安装 labelme：python -m pip install labelme"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(description="LabelMe 端点标注同步")
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="仅同步 JSON 到 endpoints.csv，不打开 LabelMe",
    )
    args = parser.parse_args()

    LABELME_JSON_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.sync_only:
        launch_labelme()

    moved = relocate_labelme_json_from_png_dir()
    if moved:
        print(f"已归档 JSON 到 labelme_json: {moved} 个")
    n_ok, n_bad, issues = sync_from_labelme_json()
    _print_sync_issue_report(issues)
    print(f"同步完成: 成功 {n_ok}，跳过 {n_bad}。")


if __name__ == "__main__":
    main()

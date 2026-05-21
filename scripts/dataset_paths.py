"""
集中管理数据集路径。

代码保存在当前仓库，图片与标注数据保存在外部数据目录。
如果以后要更换数据路径或切换 images_raw_png 下的 0/1/2 子集，只需要修改本文件顶部的配置。
"""
from __future__ import annotations

from pathlib import Path

# 要换整个数据集目录时，改这里。
DATASET_DIR = Path("/Users/zaq/Desktop/dataset")

# 当前只使用 images_raw_png/2 这一批数据；要换 0/1/2 子目录时，改这里。
RAW_IMAGE_SET = "2"

BMP_DIR = DATASET_DIR / "images_raw_bmp" / RAW_IMAGE_SET
PNG_DIR = DATASET_DIR / "images_raw_png" / RAW_IMAGE_SET
VIS_DIR = DATASET_DIR / "vis_images_labeled" / RAW_IMAGE_SET
CLEAN_DIR = DATASET_DIR / "images_clean" / RAW_IMAGE_SET
MASK_DIR = DATASET_DIR / "masks_marker" / RAW_IMAGE_SET
CLEAN_VIS_DIR = DATASET_DIR / "vis_clean" / RAW_IMAGE_SET
TRAIN_CLEAN_DIR = DATASET_DIR / "images_train_clean" / RAW_IMAGE_SET
TRAIN_CLEAN_VIS_DIR = DATASET_DIR / "vis_train_clean" / RAW_IMAGE_SET
TRAIN_CLEAN_RESIZED_DIR = DATASET_DIR / "images_train_clean_resized" / RAW_IMAGE_SET
FINAL_CHECK_VIS_DIR = DATASET_DIR / "final_check_vis" / RAW_IMAGE_SET
HEATMAP_DIR = DATASET_DIR / "heatmaps"
LOG_DIR = DATASET_DIR / "logs"

LABELS_DIR = DATASET_DIR / "labels"
LABELME_JSON_DIR = LABELS_DIR / "labelme_json"
MASTER_CSV = LABELS_DIR / "master.csv"
ENDPOINTS_CSV = LABELS_DIR / "endpoints.csv"
DATASET_350_700_CSV = LABELS_DIR / "dataset_350-700.csv"

# NT 数据集第一阶段脚本与输出说明

路径配置统一在 `scripts/dataset_paths.py` 中维护。当前数据根目录为 `/Users/zaq/Desktop/dataset`，当前处理子集为 `2`，对应输入图像目录为 `/Users/zaq/Desktop/dataset/images_raw_png/2`。

## 脚本与输出

| 脚本 | 功能 | 输出内容 | 作用 / 内容说明 |
| --- | --- | --- | --- |
| `scripts/dataset_paths.py` | 集中管理数据路径。 | 不直接生成数据文件。 | 统一定义 `DATASET_DIR`、`RAW_IMAGE_SET` 以及各输入输出目录。以后换数据根目录或切换 `0/1/2` 子集，主要改这个文件。 |
| `scripts/convert_bmp_to_png.py` | 将原始 BMP 图像批量转换为 PNG。 | `/Users/zaq/Desktop/dataset/images_raw_png/2/*.png` | 生成后续标注、建表、训练预处理使用的标准 PNG 输入图像。当前阶段 BMP 转 PNG 已完成。 |
| `scripts/create_master_csv.py` | 扫描 PNG 图像并生成主索引表。 | `/Users/zaq/Desktop/dataset/labels/master.csv` | 每张 PNG 一行，包含 `image_id`、`image_name`、`image_path_raw_png`、`width`、`height`、`patient_id`、`exam_id`、`original_date`、`quality_label`。其中 `quality_label` 当前统一为 `2`。 |
| `scripts/label_endpoints.py` | 进行端点标注，并同步 LabelMe JSON 到端点 CSV。 | `/Users/zaq/Desktop/dataset/labels/endpoints.csv` | 记录每张已标注图片的 P1/P2 坐标，字段包括 `image_id`、`image_path_raw_png`、`x1`、`y1`、`x2`、`y2`、`labeled`。`labeled=1` 表示该图已完成端点标注。 |
| `scripts/label_endpoints.py` | 每标完或同步一张图时生成端点复查图。 | `/Users/zaq/Desktop/dataset/images_labeled_vis/2/*_labeled.png` | 人工复查用图片，会在原图上显示 `P1 (x1,y1)`、`P2 (x2,y2)` 和 P1-P2 连接线。不作为模型输入。 |
| `scripts/export_labeled_vis.py` | 根据已有 `endpoints.csv` 批量重建端点复查图。 | `/Users/zaq/Desktop/dataset/images_labeled_vis/2/*_labeled.png` | 当手动修改端点 CSV 或需要重新生成可视化结果时使用，输出内容与标注脚本生成的复查图一致。 |
| `scripts/check_labels.py` | 训练前数据完整性检查。 | 终端统计、`/Users/zaq/Desktop/dataset/logs/check_labels_detail.log`、`/Users/zaq/Desktop/dataset/final_check_vis/2/text_artifacts/*_text_artifact_check.png` | 检查 `dataset_350-700.csv` 中 raw/clean/train_clean/mask 路径、端点坐标范围、`D_gt`、`angle_gt`、`scale_s`、`split`、patient/exam 跨 split、`quality_label` 分布，并输出 train_clean 文字残留候选可视化供人工复查。 |
| `scripts/build_dataset_350_700.py` | 合并 master 表和 endpoints 表，生成训练用 CSV。 | `/Users/zaq/Desktop/dataset/labels/dataset_350-700.csv` | 只保留已有端点标注的样本，包含图像元信息、`image_path_clean`、`image_path_train_clean`、`mask_path`、质量标签、P1/P2 坐标、`nt_thickness_mm`、`D_gt`、`angle_gt`、`scale_s`。后续训练统一读取 `image_path_train_clean`，marker mask 可从 `mask_path` 读取。 |
| `scripts/generate_marker_mask_clean.py` | 根据训练 CSV 生成医生标记 mask。 | `/Users/zaq/Desktop/dataset/masks_marker/2/*_mask.png` | 黑白 mask，尺寸与原图一致。背景为 `0`，P1/P2 圆点和两点连线区域为 `255`。默认端点圆半径为 `10 px`，连线宽度为 `4 px`。 |
| `scripts/generate_marker_mask_clean.py` | 根据 mask 对原图做 inpaint，生成去标记图。 | `/Users/zaq/Desktop/dataset/images_clean/2/*_clean.png` | 使用 `cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)` 去除医生端点和连线。后续模型训练应使用 clean image，不使用 raw image 或 labeled_vis。 |
| `scripts/generate_marker_mask_clean.py` | 生成 clean 结果检查图。 | `/Users/zaq/Desktop/dataset/clean_vis/2/*_vis.png` | 人工检查去标记效果用，包含原图 raw、marker mask、clean image、原图叠加 mask 和 P1/P2。 |
| `scripts/remove_text_artifacts.py` | 在 `images_clean/2` 基础上继续去除设备信息、NT 字样、右侧标尺和边缘文字。 | `/Users/zaq/Desktop/dataset/images_train_clean/2/*_train_clean.png` | 最终训练用 clean 图。脚本会检测彩色文字和边缘白色设备信息，生成 artifact mask 后做 inpaint；黑色边缘上的文字会直接恢复为黑底。 |
| `scripts/remove_text_artifacts.py` | 生成去文字结果检查图。 | `/Users/zaq/Desktop/dataset/train_clean_vis/2/*_vis.png` | 人工复查用，包含 `images_clean` 输入、artifact mask、最终 train clean、mask overlay。 |
| `scripts/resize_train_clean.py` | 将 train_clean 图按目标尺寸做 Letterbox/Stretch resize，并同步端点坐标。 | `/Users/zaq/Desktop/dataset/images_train_clean_resized/2/*_train_clean_resized.png` 和更新后的 `dataset_350-700.csv` | 默认 `512x512` letterbox。CSV 新增字段：`image_path_train_clean_resized`、`resize_target`、`resize_mode`、`resize_scale`、`pad_x`、`pad_y`、`x1_resized`、`y1_resized`、`x2_resized`、`y2_resized`。原 train_clean 不变。 |
| `scripts/generate_endpoint_heatmaps.py` | 根据 resize 后端点坐标生成 P1/P2 两通道 Gaussian heatmap。 | `/Users/zaq/Desktop/dataset/heatmaps/512/{image_id}.npy` | 默认 `512x512`、`sigma=3`，输出 shape 为 `[2, 512, 512]`，通道 0 为 P1、通道 1 为 P2。脚本也提供 `make_endpoint_heatmaps()` / `heatmaps_from_row()` 供 PyTorch Dataset 动态生成。 |
| `scripts/generate_final_check_vis.py` | 训练前随机抽查最终数据链路。 | `/Users/zaq/Desktop/dataset/final_check_vis/2/*_final_check.png` | 默认随机抽 30 张，每张包含 `raw image | clean image | train_clean image | marker mask | train_clean_resized + heatmap overlay | train_clean_resized + heatmap overlay + P1/P2`。 |

## 推荐运行顺序

1. `python scripts/create_master_csv.py`
2. `python scripts/label_endpoints.py` 或 `python scripts/label_endpoints.py --sync-only`
3. `python scripts/export_labeled_vis.py`
4. `python scripts/check_labels.py`
5. `python scripts/build_dataset_350_700.py`
6. `python scripts/generate_marker_mask_clean.py`
7. `python scripts/remove_text_artifacts.py`
8. `python scripts/resize_train_clean.py`（默认 `--target-size 512 --mode letterbox`，可改为 `1024` 等）
9. `python scripts/generate_endpoint_heatmaps.py`（默认读取 resize 后 512 坐标，保存到 `heatmaps/512`）
10. `python scripts/generate_final_check_vis.py`（默认随机抽查 30 张，输出训练前最终检查图）

如果当前环境的 `python` 不可用，可以使用实际可用解释器运行，例如 `python3` 或指定 Python 路径。

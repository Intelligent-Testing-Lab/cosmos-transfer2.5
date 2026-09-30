from ultralytics import YOLO
import os
import cv2
import numpy as np
from collections import defaultdict
import json

WEIGHTS = os.getenv(
    "YOLO_WEIGHTS",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/models/yolo26x-sem.torchscript",
)
INPUT_COSMOS_RGB = os.getenv(
    "INPUT_COSMOS_RGB",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/guided_generation.mp4",
)
INPUT_RAW_RGB = os.getenv(
    "INPUT_RAW_RGB",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/sensor_recordings/front_rgb/front_rgb.mp4",
)

OUTPUT_COSMOS_SEG = os.getenv(
    "OUTPUT_COSMOS_SEG",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/yolo26_seg_cosmos.mp4",
)

OUTPUT_RAW_SEG = os.getenv(
    "OUTPUT_RAW_SEG",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/yolo26_seg.mp4",
)

OUTPUT_RAW = os.getenv(
    "OUTPUT_IOU_RAW",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/iou_raw.npy",
)
RESULTS_FILE = os.getenv(
    "RESULTS_FILE", "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/results.json"
)

# YOLO26 / Cityscapes train-ID classes 0–18
CITYSCAPES_CLASSES = [
    {"id": 0, "name": "road", "color": (128, 64, 128)},
    {"id": 1, "name": "sidewalk", "color": (244, 35, 232)},
    {"id": 2, "name": "building", "color": (70, 70, 70)},
    {"id": 3, "name": "wall", "color": (102, 102, 156)},
    {"id": 4, "name": "fence", "color": (190, 153, 153)},
    {"id": 5, "name": "pole", "color": (153, 153, 153)},
    {"id": 6, "name": "traffic light", "color": (250, 170, 30)},
    {"id": 7, "name": "traffic sign", "color": (220, 220, 0)},
    {"id": 8, "name": "vegetation", "color": (107, 142, 35)},
    {"id": 9, "name": "terrain", "color": (152, 251, 152)},
    {"id": 10, "name": "sky", "color": (70, 130, 180)},
    {"id": 11, "name": "person", "color": (220, 20, 60)},
    {"id": 12, "name": "rider", "color": (255, 0, 0)},
    {"id": 13, "name": "car", "color": (0, 0, 142)},
    {"id": 14, "name": "truck", "color": (0, 0, 70)},
    {"id": 15, "name": "bus", "color": (0, 60, 100)},
    {"id": 16, "name": "train", "color": (0, 80, 100)},
    {"id": 17, "name": "motorcycle", "color": (0, 0, 230)},
    {"id": 18, "name": "bicycle", "color": (119, 11, 32)},
]

IGNORE_LABEL = 255

# re-implement
# def append_metric(results: dict, metric: str, metric_buffer: np.ndarray, output: str) -> None:
#    results[metric] = calculate_metric_stats(metric_buffer)
#    save_json(output, results)


def load_json(path: str) -> dict:
    mode = "r"
    if not os.path.exists(path):
        return {}

    with open(path, mode, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, content: dict) -> None:
    mode = "w"
    if not os.path.exists(path):
        mode = "a"
    with open(path, mode, encoding="utf-8") as f:
        json.dump(content, f, indent=4)


def get_frame_buffer(target: str, frames: int = -1) -> np.ndarray:
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video at {target}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames == -1 or frames > frame_count:
        frames = frame_count
    elif frames < -1:
        frames = frame_count + frames
    elif frames == 0:
        raise ValueError("Cannot get zero frames from video")

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_buffer = np.empty((frames, h, w, 3), dtype=np.uint8)

    frame_index = 0
    while frame_index < frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer[frame_index] = frame
        frame_index += 1

    cap.release()
    return frame_buffer[:frame_index]


def compute_iou_per_class(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    num_classes: int,
    ignore_label: int = IGNORE_LABEL,
) -> np.ndarray:
    """
    Compute IoU per class between gt_mask and pred_mask, ignoring gt pixels with ignore_label.
    """
    ious = np.zeros(num_classes, dtype=np.float32)

    valid = gt_mask != ignore_label

    for cid in range(num_classes):
        gt = (gt_mask == cid) & valid
        pred = (pred_mask == cid) & valid

        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()

        if union == 0:
            ious[cid] = np.nan
        else:
            ious[cid] = intersection / union

    return ious


def append_metric(file_buffer: dict, metric: str, results: dict, output: str) -> None:
    file_buffer[metric] = results
    save_json(output, file_buffer)


def colorize_mask(pred_mask: np.ndarray) -> np.ndarray:
    """
    Convert a (H,W) class-id mask to an (H,W,3) BGR image using CITYSCAPES_CLASSES colors.
    """
    h, w = pred_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in CITYSCAPES_CLASSES:
        cid = cls["id"]
        r, g, b = cls["color"]
        color_img[pred_mask == cid] = (b, g, r)  # RGB -> BGR for OpenCV
    return color_img


def main():
    # Initialise YOLO26 semantic segmentation model
    model = YOLO(WEIGHTS)

    # Load Cosmos RGB frames and CARLA GT segmentation
    cosmos_rgb = get_frame_buffer(INPUT_COSMOS_RGB)
    raw_rgb = get_frame_buffer(INPUT_RAW_RGB)

    num_frames_rgb, h_rgb, w_rgb, _ = cosmos_rgb.shape
    num_frames_gt, h_gt, w_gt, channels = raw_rgb.shape

    num_frames = min(num_frames_rgb, num_frames_gt)
    print(
        f"Cosmos frames: {num_frames_rgb}, GT frames: {num_frames_gt}, using: {num_frames}"
    )
    print(f"Cosmos resolution: {h_rgb}x{w_rgb}, GT resolution: {h_gt}x{w_gt}")

    if (h_rgb, w_rgb) != (h_gt, w_gt):
        raise ValueError("Cosmos RGB and GT segmentation resolutions do not match")

    # Set up video writer to save YOLO semantic output
    cap = cv2.VideoCapture(INPUT_COSMOS_RGB)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_cosmos = cv2.VideoWriter(OUTPUT_COSMOS_SEG, fourcc, fps, (w_rgb, h_rgb))
    out_raw = cv2.VideoWriter(OUTPUT_RAW_SEG, fourcc, fps, (w_rgb, h_rgb))

    per_class_ious = defaultdict(list)

    for t in range(num_frames):
        frame_cosmos = cosmos_rgb[t]
        frame_raw = raw_rgb[t]

        # Semantic segmentation inference
        results = [
            model.predict(frame_cosmos, verbose=False),
            model.predict(frame_raw, verbose=False),
        ]

        sem = results[0][
            0
        ].semantic_mask.data  # (H,W) class IDs [web:175][web:228][web:234]
        pred_mask = sem.cpu().numpy()

        raw_mask_carla = results[1][0].semantic_mask.data
        mask_carla = raw_mask_carla.cpu().numpy()

        # IoU per class for this frame
        ious_t = compute_iou_per_class(
            mask_carla,
            pred_mask,
            num_classes=len(CITYSCAPES_CLASSES),
            ignore_label=IGNORE_LABEL,
        )
        for cid, iou_c in enumerate(ious_t):
            per_class_ious[cid].append(iou_c)

        # Colorize prediction and overlay
        color_pred = colorize_mask(pred_mask)
        overlay = cv2.addWeighted(frame_cosmos, 0.5, color_pred, 0.5, 0)
        out_cosmos.write(overlay)

        color_raw = colorize_mask(mask_carla)
        overlay = cv2.addWeighted(frame_raw, 0.5, color_pred, 0.5, 0)
        out_raw.write(overlay)

        if (t + 1) % 10 == 0 or t == num_frames - 1:
            print(f"Processed frame {t + 1}/{num_frames}")

    out_raw.release()
    out_cosmos.release()

    print(f"Saved YOLO semantic output video to: {OUTPUT_RAW_SEG}")
    print(f"Saved YOLO semantic output video for COSMOS to: {OUTPUT_COSMOS_SEG}")

    result_dict = {
        "per_class_iou": [],
        "npy": OUTPUT_RAW,
        "target": {"input": INPUT_RAW_RGB, "output": OUTPUT_RAW_SEG},
        "prediction": {"input": INPUT_COSMOS_RGB, "output": OUTPUT_COSMOS_SEG},
    }

    print("\nPer-class IoU statistics (over frames):")
    for cls in CITYSCAPES_CLASSES:
        cid = cls["id"]
        name = cls["name"]
        vals = np.array(per_class_ious[cid], dtype=np.float32)
        valid = vals[~np.isnan(vals)]

        if valid.size == 0:
            print(f"{cid:2d} | {name:15s} | no valid pixels in any frame")
            continue

        mean_iou = float(valid.mean())
        std_iou = float(valid.std())
        min_iou = float(valid.min())
        max_iou = float(valid.max())

        # save the results in a dict
        vals[np.isnan(vals)] = -1
        
        result_dict["per_class_iou"].append(
            {
                "class_id": cid,
                "name": name,
                "mean_iou": mean_iou,
                "std_iou": std_iou,
                "min_iou": min_iou,
                "max_iou": max_iou,
                "frames": int(valid.size),
                "values": [float(value) for value in vals],
            }
        )

        print(
            f"{cid:2d} | {name:15s} | "
            f"mean IoU: {mean_iou:.3f}, std: {std_iou:.3f}, "
            f"min: {min_iou:.3f}, max: {max_iou:.3f}, "
            f"frames: {valid.size}"
        )

    append_metric(load_json(RESULTS_FILE), "iou", result_dict, RESULTS_FILE)
    np.save(OUTPUT_RAW, {cid: np.array(vals) for cid, vals in per_class_ious.items()})  # type: ignore


if __name__ == "__main__":
    main()

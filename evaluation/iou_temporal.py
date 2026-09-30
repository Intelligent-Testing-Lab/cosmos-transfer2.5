from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
from collections import defaultdict

WEIGHTS = os.getenv(
    "YOLO_WEIGHTS",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/models/yolo26x-sem.torchscript",
)
INPUT_RGB = os.getenv(
    "INPUT_RGB",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/guided_generation.mp4",
)
OUTPUT_FLOW_VIDEO = os.getenv(
    "OUTPUT_FLOW_VIDEO",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/yolo26_sem_flow_iou.mp4",
)
OUTPUT_FLOW_RAW = os.getenv(
    "OUTPUT_FLOW_RAW",
    "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/yolo26_sem_flow_iou.npy",
)

RESULTS_FILE = os.getenv(
    "RESULTS_FILE", "/data/cosmos/cosmos-transfer2.5/evaluation/data/test/results.json"
)

# YOLO26 / Cityscapes train-ID classes 0–18
CITYSCAPES_CLASSES = [
    {"id": 0,  "name": "road",          "color": (128, 64, 128)},
    {"id": 1,  "name": "sidewalk",      "color": (244, 35, 232)},
    {"id": 2,  "name": "building",      "color": (70,  70,  70)},
    {"id": 3,  "name": "wall",          "color": (102, 102, 156)},
    {"id": 4,  "name": "fence",         "color": (190, 153, 153)},
    {"id": 5,  "name": "pole",          "color": (153, 153, 153)},
    {"id": 6,  "name": "traffic light", "color": (250, 170, 30)},
    {"id": 7,  "name": "traffic sign",  "color": (220, 220, 0)},
    {"id": 8,  "name": "vegetation",    "color": (107, 142, 35)},
    {"id": 9,  "name": "terrain",       "color": (152, 251, 152)},
    {"id": 10, "name": "sky",           "color": (70,  130, 180)},
    {"id": 11, "name": "person",        "color": (220, 20,  60)},
    {"id": 12, "name": "rider",         "color": (255, 0,   0)},
    {"id": 13, "name": "car",           "color": (0,   0,   142)},
    {"id": 14, "name": "truck",         "color": (0,   0,   70)},
    {"id": 15, "name": "bus",           "color": (0,   60,  100)},
    {"id": 16, "name": "train",         "color": (0,   80,  100)},
    {"id": 17, "name": "motorcycle",    "color": (0,   0,   230)},
    {"id": 18, "name": "bicycle",       "color": (119, 11,  32)},
]

IGNORE_LABEL = 255


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


def compute_iou_per_class(mask_a: np.ndarray,
                          mask_b: np.ndarray,
                          num_classes: int,
                          ignore_label: int = IGNORE_LABEL) -> np.ndarray:
    """
    IoU per class between mask_a and mask_b, ignoring pixels with ignore_label in either mask.
    """
    ious = np.zeros(num_classes, dtype=np.float32)

    valid = (mask_a != ignore_label) & (mask_b != ignore_label)

    for cid in range(num_classes):
        a = (mask_a == cid) & valid
        b = (mask_b == cid) & valid

        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()

        if union == 0:
            ious[cid] = np.nan
        else:
            ious[cid] = intersection / union

    return ious

def save_json(path: str, content: dict) -> None:
    mode = "w"
    if not os.path.exists(path):
        mode = "a"
    with open(path, mode, encoding="utf-8") as f:
        json.dump(content, f, indent=4)

def load_json(path: str) -> dict:
    mode = "r"
    if not os.path.exists(path):
        return {}

    with open(path, mode, encoding="utf-8") as f:
        return json.load(f)

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


def warp_mask_with_flow(mask: np.ndarray,
                        flow: np.ndarray,
                        ignore_label: int = IGNORE_LABEL) -> np.ndarray:
    """
    Warp a class-ID mask forward using dense optical flow.

    mask: (H,W) uint8 class IDs
    flow: (H,W,2) float32, (dx, dy) per pixel
    """
    h, w = mask.shape
    # Create coordinate grid
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        mask.astype(np.float32),
        map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(ignore_label),
    ).astype(np.uint8)

    return warped


def main():
    # Initialise YOLO26 semantic segmentation model
    model = YOLO(WEIGHTS)

    # Load Cosmos RGB frames
    cosmos_rgb = get_frame_buffer(INPUT_RGB)
    num_frames, h, w, _ = cosmos_rgb.shape
    print(f"Cosmos frames: {num_frames}, resolution: {h}x{w}")

    # Get FPS for output video
    cap = cv2.VideoCapture(INPUT_RGB)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_FLOW_VIDEO, fourcc, fps, (w, h))

    # Store semantic masks
    sem_masks = []

    # Run YOLO semantic segmentation on all frames
    for t in range(num_frames):
        frame = cosmos_rgb[t]
        results = model.predict(frame, verbose=False)
        r = results[0]
        sem = r.semantic_mask.data  # (H,W) train-ID class map [web:175][web:184][web:242]
        pred_mask = sem.cpu().numpy()

        sem_masks.append(pred_mask)

        # For visualization, we’ll just write original for now; we’ll overlay later
        out.write(frame)

        if (t + 1) % 10 == 0 or t == num_frames - 1:
            print(f"Semantic segmentation: processed frame {t + 1}/{num_frames}")

    out.release()

    # Now compute optical flow IoU between consecutive frames
    per_class_flow_ious = defaultdict(list)

    flow_video_out = cv2.VideoWriter(
        OUTPUT_FLOW_VIDEO.replace(".mp4", "_flow_overlay.mp4"),
        fourcc, fps, (w, h)
    )

    for t in range(num_frames - 1):
        frame_t = cosmos_rgb[t]
        frame_t1 = cosmos_rgb[t + 1]

        # Optical flow on grayscale frames
        prev_gray = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        prev_mask = sem_masks[t]
        curr_mask = sem_masks[t + 1]

        warped_prev_mask = warp_mask_with_flow(prev_mask, flow, ignore_label=IGNORE_LABEL)

        ious_t = compute_iou_per_class(
            warped_prev_mask,
            curr_mask,
            num_classes=len(CITYSCAPES_CLASSES),
            ignore_label=IGNORE_LABEL,
        )

        for cid, iou_c in enumerate(ious_t):
            per_class_flow_ious[cid].append(iou_c)

        # Visualization: overlay current mask and warped previous mask
        color_curr = colorize_mask(curr_mask)
        color_warped = colorize_mask(warped_prev_mask)

        overlay = cv2.addWeighted(frame_t1, 0.5, color_curr, 0.3, 0)
        overlay = cv2.addWeighted(overlay, 0.8, color_warped, 0.2, 0)

        flow_video_out.write(overlay)

        if (t + 1) % 10 == 0 or t == num_frames - 2:
            print(f"Optical flow IoU: processed pair {t + 1}/{num_frames - 1}")

    flow_video_out.release()
    print(f"Saved optical-flow IoU overlay video to: {OUTPUT_FLOW_VIDEO.replace('.mp4', '_flow_overlay.mp4')}")
    
    # Aggregate and print IoU over time
    result_dict = {
          "per_class_iou": [],
          "npy": OUTPUT_FLOW_RAW,
          "target": {"input": INPUT_RGB, "output": OUTPUT_FLOW_VIDEO},
      }
    
    print("\nPer-class optical-flow IoU statistics (warped t -> t+1):")
    for cls in CITYSCAPES_CLASSES:
        cid = cls["id"]
        name = cls["name"]
        vals = np.array(per_class_flow_ious[cid], dtype=np.float32)
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            print(f"{cid:2d} | {name:15s} | no valid pixels in any frame pair")
            continue
        mean_iou = float(valid.mean())
        std_iou = float(valid.std())
        min_iou = float(valid.min())
        max_iou = float(valid.max())
        print(
            f"{cid:2d} | {name:15s} | "
            f"mean IoU: {mean_iou:.3f}, std: {std_iou:.3f}, "
            f"min: {min_iou:.3f}, max: {max_iou:.3f}, "
            f"pairs: {valid.size}"
        )
        
        # convert Nan values so they can be serialiased
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
        
    # Save full per-class optical-flow IoU distributions for later analysis
    flow_ious_dict = {
        cid: np.array(vals, dtype=np.float32)
        for cid, vals in per_class_flow_ious.items()
    }
    np.save(
        OUTPUT_FLOW_RAW,
        flow_ious_dict,
    ) # type: ignore
    append_metric(load_json(RESULTS_FILE), "iou_optical_flow", result_dict, RESULTS_FILE)
    print("Saved raw optical-flow IoU data to optical_flow_per_class_ious.npy")

if __name__ == "__main__":
    main()
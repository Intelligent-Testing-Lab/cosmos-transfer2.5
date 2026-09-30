from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from concurrent.futures import ThreadPoolExecutor   

import numpy as np
import os
import sys
import json
import cv2
import torch
import pyiqa

def get_frame_buffer(target: str, frames: int = -1) -> np.ndarray:
    """
    Returns a frames length buffer. If set to -1 its the total length of the video. 
    If frames is more than total framecount, return framecount 
    """
    
    cap = cv2.VideoCapture(target)
    
    if not cap.isOpened():
        print("Failed to open video: is the path correct?")
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames > frame_count or frames == -1:
        frames = frame_count
    elif frames < -1:
        frames = frame_count + frames
    elif frames == 0:
        print("Cannot get zero frames from video")
        
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    frame_buffer = np.ndarray((frames, h,w, 3), dtype=np.uint8)
    
    frame_index = 0
    while True:
        ret, frame = cap.read()
        
        frame_buffer[frame_index] = frame

        # exit loop if video finished or target frames reached    
        if not ret or frame_index == (frame_count - 1):
            break 
        else:
            frame_index += 1

    cap.release()
    return frame_buffer

def get_video_metadata(buffer: np.ndarray, path: str) -> dict:
    return {
        "path": path,
        "type": os.path.splitext(path)[1], # extension
        "frames": buffer.shape[0],
        "height": buffer.shape[1],
        "width": buffer.shape[2],
        "channels": 3    
    }
   
def calculate_outliers(metric_buffer: np.ndarray, multiplier: float=1.5) -> dict:
    q1, q3 = np.percentile(metric_buffer, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)
    outlier_mask = (metric_buffer < lower_bound) | (metric_buffer > upper_bound)
    outliers = []
    if outlier_mask.any():
        outliers = metric_buffer[outlier_mask]
    
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outliers": [float(outlier) for outlier in outliers],
    }
 
def calculate_metric_stats(metric_buffer: np.ndarray) -> dict:
    return {
            "mean": float(np.mean(metric_buffer)),
            "median": float(np.median(metric_buffer)),
            "std": float(np.std(metric_buffer)),
            "min": float(np.min(metric_buffer)),
            "max": float(np.max(metric_buffer)),
            "outliers": calculate_outliers(metric_buffer),
            "values": [float(value) for value in metric_buffer.flatten()],
    }
    
def append_metric(results: dict, metric: str, metric_buffer: np.ndarray, output: str) -> None:        
    results[metric] = calculate_metric_stats(metric_buffer)
    save_json(output, results)

def load_json(path: str) -> dict:
    mode="r"
    if not os.path.exists(path):
        return {}
    
    with open(path, mode, encoding='utf-8') as f:
        return json.load(f)

def save_json(path: str, content: dict) -> None:
    mode="w"
    if not os.path.exists(path):
        mode="a"
    with open(path, mode, encoding='utf-8') as f:
        json.dump(content, f, indent=4)

# replace with lpip function
def calculate_clipqa(
    metric,
    pred_chunk: np.ndarray,
) -> np.ndarray:
    
    # normalise both buffers to 0-1
    pred_chunk = pred_chunk / 255
    
    # reshape into (N, 3, H, W)
    frames = pred_chunk.shape[0]
    h = pred_chunk.shape[1]
    w = pred_chunk.shape[2]
    channels = 3
    
    pred_chunk = pred_chunk.reshape(frames, channels, h, w).astype(dtype=np.float32)    
    score_fr = metric(torch.from_numpy(pred_chunk))
    
    return score_fr.cpu().numpy()

# check both videos are equal in frames
def main():
    pred_buffer = get_frame_buffer(os.getenv("CLIPQA_PRED", ""))

    output = os.getenv("RESULTS_FILE", "")

    pred_shape = pred_buffer.shape
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    CHUNK_FRAME_SIZE = 10
    CHUNK_DIFF = int(pred_shape[0] / CHUNK_FRAME_SIZE)
    pred_chunks = np.array_split(pred_buffer, CHUNK_DIFF, axis=0)
    clipqa_func = pyiqa.create_metric("clipiqa+", device=device, as_loss=False)
    
    clipqa_chunked = []
    for chunk in range(0, CHUNK_DIFF):
        clipqa_chunked.append(calculate_clipqa(clipqa_func, pred_chunks[chunk]))
        print(f"Processed chunk {chunk}/{CHUNK_DIFF}")
    clipqa = np.concatenate(clipqa_chunked)
    
    append_metric(load_json(output), "clipqa", clipqa, output) # type: ignore
    
    carla_buffer = get_frame_buffer(os.getenv("CLIPQA_CARLA", ""))
    
    carla_shape = carla_buffer.shape

    CHUNK_FRAME_SIZE = 10
    CHUNK_DIFF = int(carla_shape[0] / CHUNK_FRAME_SIZE)
    carla_chunks = np.array_split(carla_buffer, CHUNK_DIFF, axis=0)
    
    clipqa_chunked = []
    for chunk in range(0, CHUNK_DIFF):
        clipqa_chunked.append(calculate_clipqa(clipqa_func, carla_chunks[chunk]))
        print(f"Processed chunk {chunk}/{CHUNK_DIFF}")
    clipqa = np.concatenate(clipqa_chunked)
    
    append_metric(load_json(output), "clipqa_carla", clipqa, output) # type: ignore
    
if __name__ == "__main__":
    main()

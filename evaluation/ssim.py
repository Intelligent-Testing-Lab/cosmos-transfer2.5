from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from concurrent.futures import ThreadPoolExecutor   

import numpy as np
import os
import sys
import json
import cv2

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
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers": list(outliers),
    }
 
def calculate_metric_stats(metric_buffer: np.ndarray) -> dict:
    return {
            "mean": np.mean(metric_buffer),
            "median": np.median(metric_buffer),
            "std": np.std(metric_buffer),
            "min": np.min(metric_buffer),
            "max": np.max(metric_buffer),
            "outliers": calculate_outliers(metric_buffer),
            "values": list(metric_buffer)
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

def calculate_chunk_ssim(
    true_chunk: np.ndarray,
    pred_chunk: np.ndarray,
) -> np.ndarray:
    return np.asarray([
        ssim(
            true_frame,
            pred_frame,
            data_range=255,
            channel_axis=-1,
        )
        for true_frame, pred_frame in zip(true_chunk, pred_chunk)
    ])

# check both videos are equal in frames
def main():
    pred_buffer = get_frame_buffer(os.getenv("SSIM_PRED", ""))
    true_buffer = get_frame_buffer(os.getenv("SSIM_TRUE", ""))
    output = os.getenv("RESULTS_FILE", "")

    pred_shape = pred_buffer.shape
    true_shape = true_buffer.shape
    
    if not (pred_shape[0] == true_shape[0]):
        print("Videos are not equal length in frames")
        sys.exit(1)

    # to match Wang Et Al paper, set
    # gaussian_weights to True
    # sigma to 1.5
    # use_sample_covariance to False
    # data_range should be inferred, its standard 0-255
    
    num_threads = int(os.getenv("SSIM_THREADS", 60)) # 600 frames total: 10 frames per thread
    pred_chunks = np.array_split(pred_buffer, num_threads, axis=0)
    true_chunks = np.array_split(true_buffer, num_threads, axis=0)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                calculate_chunk_ssim,
                true_chunks[i],
                pred_chunks[i],
            )
            for i in range(num_threads)
        ]
            
        ssim_buffer = np.concatenate(
            [future.result() for future in futures]
        )        
    append_metric(load_json(output), "ssim", ssim_buffer, output) # type: ignore
    
if __name__ == "__main__":
    main()





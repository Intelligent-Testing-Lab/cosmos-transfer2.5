import cv2
import os
import numpy as np

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


VIDEO = os.getenv("TARGET_VIDEO", "")
TARGET_FRAME = 104 # 1 to N
cv2.imwrite(f"frame_{TARGET_FRAME}.png", get_frame_buffer(VIDEO)[TARGET_FRAME-1])

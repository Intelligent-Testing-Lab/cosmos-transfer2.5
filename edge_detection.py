import cv2
import numpy as np
import os
from scipy.ndimage import binary_erosion, generate_binary_structure

# fill these in

USE_CANNY = False

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

    
def canny_edge(frame: np.ndarray, 
               min_threshold: int=10, 
               max_threshold: int=50, 
               kernel_size: int=3,
               gaussian_kernel: int=3,
               gaussian_sigma: float = 1.4,
               gradient_func: bool=True,
               blur: bool = True,
               adjust_image_features: bool = True,
               brightness: int =50,
               contrast: float =1.0
            ) -> np.ndarray:
    if adjust_image_features:
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    if blur:
        blurred = cv2.GaussianBlur(gray, (gaussian_kernel,gaussian_kernel), gaussian_sigma)
    return cv2.Canny(blurred if blur else gray, min_threshold, max_threshold, apertureSize=kernel_size, L2gradient=gradient_func)

def save_mp4(
    frames: np.ndarray,
    path: str,
    fps: float = 20.0,
) -> None:
    """Save uint8 BGR frames of shape (N, H, W, 3) as an MP4."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (N, H, W, 3)")

    frames = frames.astype(np.uint8, copy=False)
    _, height, width, _ = frames.shape

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),  # OpenCV expects (W, H), not (H, W),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    for frame in frames:
        writer.write(frame)

    writer.release()

def run_canny(target: str) -> np.ndarray:
    frame_buffer = get_frame_buffer(target)           
    return np.stack([cv2.cvtColor(canny_edge(f), cv2.COLOR_GRAY2BGR) for f in frame_buffer], axis=0)

raw_semantic = os.getenv("RAW_SEG_TARGET", "/data/cosmos/cosmos-transfer2.5/outputs/run1/3e330e41-21aa-4f5f-ad64-addd42c2705e/sensor_recordings/front_semantic_id/raw_frames.npy")
semantic_mask = os.getenv("EDGE_MASK_INPUT", "/data/cosmos/cosmos-transfer2.5/outputs/run1/3e330e41-21aa-4f5f-ad64-addd42c2705e/edge_mask.mp4")
target = os.getenv("CANNY_DETECTION_TARGET", "/data/cosmos/cosmos-transfer2.5/outputs/run1/3e330e41-21aa-4f5f-ad64-addd42c2705e/sensor_recordings/front_rgb/front_rgb.mp4")
output_dir = os.getenv("CANNY_DETECTION_OUTPUT", "/data/cosmos/cosmos-transfer2.5/outputs/run1/3e330e41-21aa-4f5f-ad64-addd42c2705e/detected_edges.mp4")

# fixed - now each border has a deterministic rule for pixel owernship
# vertical border - upper pixel owner
# horizontal border - left pixel owner
def run_edge_algorithm(
    semantic_data_path: str,
    return_mask: bool = False,
) -> np.ndarray:
    labels = np.load(semantic_data_path)  # (N, H, W)

    # One pixel owns each label transition.
    # Vertical transition between (y, x) and (y + 1, x):
    # assign edge to the upper pixel (y, x).
    vertical_change = labels[:, :-1, :] != labels[:, 1:, :]

    # Horizontal transition between (y, x) and (y, x + 1):
    # assign edge to the left pixel (y, x).
    horizontal_change = labels[:, :, :-1] != labels[:, :, 1:]

    edge_mask = np.zeros_like(labels, dtype=bool)
    edge_mask[:, :-1, :] |= vertical_change
    edge_mask[:, :, :-1] |= horizontal_change

    if return_mask:
        return edge_mask

    detected_edges = np.zeros((*labels.shape, 3), dtype=np.uint8)
    detected_edges[edge_mask] = (255, 255, 255)
    return detected_edges

#def run_edge_algorithm(semantic_data_path: str, return_mask: bool = False) -> np.ndarray:
#    semantic_data = np.load(semantic_data_path)  # shape: (N, H, W)
#
#    # True if a cell differs from any 4-connected neighbour.
#    edge_mask = np.zeros(semantic_data.shape, dtype=bool)
#
#    # Compare vertically adjacent cells: (N, H - 1, W, C)
#    vertical_change = semantic_data[:, 1:, :] != semantic_data[:, :-1, :]
#    edge_mask[:, 1:, :] |= vertical_change
#    edge_mask[:, :-1, :] |= vertical_change
#
#    # Compare horizontally adjacent cells: (N, H, W - 1, C)
#    horizontal_change = semantic_data[:, :, 1:] != semantic_data[:, :, :-1]
#    edge_mask[:, :, 1:] |= horizontal_change
#    edge_mask[:, :, :-1] |= horizontal_change
#
#    truth_label = (255,255,255)
#    false_label = (0  ,0  ,0  )
#    
#    structure_2d = generate_binary_structure(2,1) # 4 connectivity
#    # Apply erosion to each frame independently
#    edge_mask_1px = np.zeros_like(edge_mask, dtype=bool)
#    for i in range(edge_mask.shape[0]):
#        inner = binary_erosion(edge_mask[i], structure=structure_2d, iterations=1)
#        edge_mask_1px[i] = edge_mask[i] & (~inner) # type: ignore
#
#    if return_mask:
#        return edge_mask
#
#    # overlay black and white over the mask
#    detected_edges = np.zeros_like(np.repeat(edge_mask_1px[..., None], 3, axis=-1), dtype=np.uint8)
#    
#    detected_edges[edge_mask_1px] = truth_label
#    detected_edges[~edge_mask_1px] = false_label
#    
#    return detected_edges

truth_label = (255,255,255)
false_label = (0  ,0  ,0  )
tolerance = 255

gt_semantic_edge_mask = run_edge_algorithm(raw_semantic, return_mask=True)
# run canny edge detection on the raw RGB input and convert it to a mask
canny_edge_rgb = run_canny(target) 
canny_edge_mask = np.all(np.abs(canny_edge_rgb - (255,255,255)) < tolerance, axis=-1)

# apply a mask over the canny edge
semantic_mask = get_frame_buffer(semantic_mask)
semantic_mask_bool = np.all(np.abs(semantic_mask - (255,255,255)) < tolerance, axis=-1)

# we only want the detected edges in the ROIs
roi_edge_mask = canny_edge_mask & semantic_mask_bool

# compute OR over both the semantic GT mask and the canny roi mask
detected_edges = roi_edge_mask | gt_semantic_edge_mask

# convert final mask to edge video
final_output = np.zeros_like(np.repeat(detected_edges[..., None], 3, axis=-1), dtype=np.uint8)
final_output[detected_edges] = truth_label
final_output[~detected_edges] = false_label

save_mp4(final_output, output_dir)


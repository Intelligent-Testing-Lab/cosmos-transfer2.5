OUTPUT_DIR="/data/cosmos/cosmos-transfer2.5/outputs/old_runs/config_001/52e1ed8a-ead0-41c9-8ddc-45f37c6e24c8"
EVAL_ROOT="/data/cosmos/cosmos-transfer2.5/evaluation"
SSIM_PRED="${OUTPUT_DIR}/sensor_recordings/front_rgb/front_rgb.mp4"
SSIM_TRUE="${OUTPUT_DIR}/guided_generation.mp4"
RESULTS_FILE="${OUTPUT_DIR}/results.json"

INPUT_COSMOS_RGB="${SSIM_TRUE}"
INPUT_RAW_RGB="${SSIM_PRED}"
OUTPUT_COSMOS_SEG="${OUTPUT_DIR}/yolo_seg_cosmos.mp4"
OUTPUT_RAW_SEG="${OUTPUT_DIR}/yolo_seg.npy"
OUTPUT_IOU_RAW="${OUTPUT_DIR}/semantic_iou_raw.npy"

CUDA_VISIBLE_DEVICES="3" RESULTS_FILE="${RESULTS_FILE}" INPUT_COSMOS_RGB=${INPUT_COSMOS_RGB} INPUT_RAW_RGB=${INPUT_RAW_RGB} OUTPUT_COSMOS_SEG=${OUTPUT_COSMOS_SEG} OUTPUT_RAW_SEG=${OUTPUT_RAW_SEG} OUTPUT_IOU_RAW=${OUTPUT_IOU_RAW} python3 "${EVAL_ROOT}/iou.py"
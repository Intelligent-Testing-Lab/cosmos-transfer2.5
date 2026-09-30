#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./process_all_outputs.sh /path/to/outputs

OUTPUTS_DIR="${1:?Usage: $0 /path/to/outputs}"
OUTPUTS_DIR="$(cd "$OUTPUTS_DIR" && pwd)"

# Locate UUID directories within every outputs/config_00n directory.
mapfile -t uuid_dirs < <(
    find "$OUTPUTS_DIR" \
        -mindepth 2 \
        -maxdepth 2 \
        -type d \
        -path "$OUTPUTS_DIR/config_0[0-9][0-9]/*" \
        -regextype posix-extended \
        -regex '.*/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' \
        -print \
        | sort
)

source "/data/cosmos/cosmos-transfer2.5/evaluation/.venv/bin/activate"
echo "Found ${#uuid_dirs[@]} UUID directories."

for uuid_dir in "${uuid_dirs[@]}"; do

    EVAL_ROOT="/data/cosmos/cosmos-transfer2.5/evaluation"
    SSIM_PRED="${uuid_dir}/sensor_recordings/front_rgb/front_rgb.mp4"
    SSIM_TRUE="${uuid_dir}/guided_generation.mp4"
    RESULTS_FILE="${uuid_dir}/results.json"

    # check that SSIM_TRUE exists (generation was complete)
    # if not skip the current for loop iteration
    if [[ ! -f "${SSIM_TRUE}" ]]; then 
        echo "Skipping $uuid_dir; no generation artifacts found."
        continue
    fi
    echo "Processing: $uuid_dir"

    CUDA_VISIBLE_DEVICE="3" LPIP_PRED="${SSIM_PRED}" LPIP_TRUE="${SSIM_TRUE}" RESULTS_FILE="${RESULTS_FILE}" python3 "${EVAL_ROOT}/lpip.py"
    CUDA_VISIBLE_DEVICE="3" CLIPQA_PRED="${SSIM_PRED}" RESULTS_FILE="${RESULTS_FILE}" python3 "${EVAL_ROOT}/clipqa.py"
    SSIM_PRED="${SSIM_PRED}" SSIM_TRUE="${SSIM_TRUE}" RESULTS_FILE="${RESULTS_FILE}" python3 "${EVAL_ROOT}/ssim.py"

    INPUT_COSMOS_RGB="${SSIM_TRUE}"
    INPUT_RAW_RGB="${SSIM_PRED}"
    OUTPUT_COSMOS_SEG="${uuid_dir}/yolo_seg_cosmos.mp4"
    OUTPUT_RAW_SEG="${uuid_dir}/yolo_seg.npy"
    OUTPUT_IOU_RAW="${uuid_dir}/semantic_iou_raw.npy"

    CUDA_VISIBLE_DEVICES="3" RESULTS_FILE="${RESULTS_FILE}" INPUT_COSMOS_RGB=${INPUT_COSMOS_RGB} INPUT_RAW_RGB=${INPUT_RAW_RGB} OUTPUT_COSMOS_SEG=${OUTPUT_COSMOS_SEG} OUTPUT_RAW_SEG=${OUTPUT_RAW_SEG} OUTPUT_IOU_RAW=${OUTPUT_IOU_RAW} python3 "${EVAL_ROOT}/iou.py"

    OUTPUT_FLOW_VIDEO="${uuid_dir}/yolo_segmentation_eval_temporal.mp4"
    OUTPUT_FLOW_RAW="${uuid_dir}/temporal_iou_raw.npy"

    RESULTS_FILE="${RESULTS_FILE}" INPUT_RGB=${SSIM_PRED} OUTPUT_FLOW_VIDEO=${OUTPUT_FLOW_VIDEO} OUTPUT_FLOW_RAW=${OUTPUT_FLOW_RAW} python3 "${EVAL_ROOT}/iou_temporal.py"
done

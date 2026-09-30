#!/bin/bash

CARLA_ENV="/data/cosmos/cosmos-transfer2.5/prompt_processor/.venv/bin/activate"
COSMOS_ENV="/data/cosmos/cosmos-transfer2.5/.venv/bin/activate"
EVAL_ENV="/data/cosmos/cosmos-transfer2.5/evaluation/.venv/bin/activate"

SCENARIO_LENGTH=20 # seconds
FIXED_DELTA=0.05 # seconds
VEHICLES=50
TOWN="Town10HD"

EXPERIMENTS=10
INDEX=0 
SEED_MULTIPLIER=324

while [ $INDEX -le $EXPERIMENTS ]
do
    SEED=$((INDEX * SEED_MULTIPLIER))
    OUTPUT_UUID=$(python3 "scripts/generate_uuid.py")
    OUTPUT_DIR="/data/cosmos/cosmos-transfer2.5/outputs/scratch/test_3/${OUTPUT_UUID}_${SEED}"
    

    source "${CARLA_ENV}"

    python3 basic_recorder.py \
        --town "${TOWN}" \
        --seconds "${SCENARIO_LENGTH}" \
        --delta "${FIXED_DELTA}" \
        --traffic-vehicles "${VEHICLES}" \
        --traffic-speed-difference 10 \
        --output-dir "${OUTPUT_DIR}/sensor_recordings" \
        --seed "${SEED}"

    # inputs and sensor data
    RGB_INPUT="${OUTPUT_DIR}/sensor_recordings/front_rgb/front_rgb.mp4"
    SEMANTIC_INPUT="${OUTPUT_DIR}/sensor_recordings/front_semantic_cityscapes/front_semantic_cityscapes.mp4"
    DEPTH_INPUT="${OUTPUT_DIR}/sensor_recordings/front_depth_log/front_depth_log.mp4"
    RAW_SEMANTIC_DATA="${OUTPUT_DIR}/sensor_recordings/front_semantic_id/raw_frames.npy"
    
    # outputs
    SEG_MASK_OUTPUT="${OUTPUT_DIR}/seg_mask.mp4"
    EDGE_MASK_OUTPUT="${OUTPUT_DIR}/edge_mask.mp4"
    CANNY_DETECTION_OUTPUT="${OUTPUT_DIR}/edge.mp4"
    
    source "${COSMOS_ENV}"

    # run prompt generator
    RGB_INPUT="${RGB_INPUT}" OUTPUT_DIR="${OUTPUT_DIR}" python run_prompt_processor.py
    # run seg mask generator
    RAW_SEG_TARGET="${RAW_SEMANTIC_DATA}" SEG_MASK_OUTPUT="${SEG_MASK_OUTPUT}" EDGE_MASK_OUTPUT="${EDGE_MASK_OUTPUT}" python3 preprocess_seg.py
    
    # run canny edge  
    RAW_SEG_TARGET="${RAW_SEMANTIC_DATA}" EDGE_MASK_INPUT="${EDGE_MASK_OUTPUT}" CANNY_DETECTION_TARGET="${RGB_INPUT}" CANNY_DETECTION_OUTPUT="${CANNY_DETECTION_OUTPUT}" python3 edge_detection.py
    
    # read from each sweep dir
    # save generation config
    cat >> "${OUTPUT_DIR}/generation_config.json" <<EOF
        {
    "name": "guided_generation",
    "prompt_path": "${OUTPUT_DIR}/prompt.txt",
    "video_path": "${RGB_INPUT}",
    "guidance": 5,
    "edge": {
      "control_path": "${CANNY_DETECTION_OUTPUT}",
      "control_weight": 0.3
    },
    "seg": {
      "control_path": "${SEMANTIC_INPUT}",
      "mask_path": "${SEG_MASK_OUTPUT}",
      "control_weight": 0.1
    },
    "depth": {
      "control_path": "${DEPTH_INPUT}",
      "control_weight": 0.5
    }
    } 
EOF

    # setup multi-gpu inference
    MAX_GPUS=4 INPUT_COSMOS_PATH="${OUTPUT_DIR}/generation_config.json" OUTPUT_COSMOS_PATH="${OUTPUT_DIR}/" python3 run_inference.py

    EVAL_ROOT="/data/cosmos/cosmos-transfer2.5/evaluation"
    SSIM_PRED="${OUTPUT_DIR}/sensor_recordings/front_rgb/front_rgb.mp4"
    SSIM_TRUE="${OUTPUT_DIR}/guided_generation.mp4"
    RESULTS_FILE="${OUTPUT_DIR}/results.json"

    # check that SSIM_TRUE exists (generation was complete)
    # if not skip the current for loop iteration
    if [[ ! -f "${SSIM_TRUE}" ]]; then 
        echo "Skipping $OUTPUT_DIR; no generation artifacts found."
        continue
    fi
    echo "Evaluating"
    source "${EVAL_ENV}"

    CUDA_VISIBLE_DEVICE="3" LPIP_PRED="${SSIM_PRED}" LPIP_TRUE="${SSIM_TRUE}" RESULTS_FILE="${RESULTS_FILE}" python3 "${EVAL_ROOT}/lpip.py"
    CUDA_VISIBLE_DEVICE="3" CLIPQA_PRED="${SSIM_PRED}" RESULTS_FILE="${RESULTS_FILE}" python3 "${EVAL_ROOT}/clipqa.py"
    SSIM_PRED="${SSIM_PRED}" SSIM_TRUE="${SSIM_TRUE}" RESULTS_FILE="${RESULTS_FILE}" python3 "${EVAL_ROOT}/ssim.py"

    INPUT_COSMOS_RGB="${SSIM_TRUE}"
    INPUT_RAW_RGB="${SSIM_PRED}"
    OUTPUT_COSMOS_SEG="${OUTPUT_DIR}/yolo_seg_cosmos.mp4"
    OUTPUT_RAW_SEG="${OUTPUT_DIR}/yolo_seg.npy"
    OUTPUT_IOU_RAW="${OUTPUT_DIR}/semantic_iou_raw.npy"

    CUDA_VISIBLE_DEVICES="3" RESULTS_FILE="${RESULTS_FILE}" INPUT_COSMOS_RGB=${INPUT_COSMOS_RGB} INPUT_RAW_RGB=${INPUT_RAW_RGB} OUTPUT_COSMOS_SEG=${OUTPUT_COSMOS_SEG} OUTPUT_RAW_SEG=${OUTPUT_RAW_SEG} OUTPUT_IOU_RAW=${OUTPUT_IOU_RAW} python3 "${EVAL_ROOT}/iou.py"

    OUTPUT_FLOW_VIDEO="${OUTPUT_DIR}/yolo_segmentation_eval_temporal.mp4"
    OUTPUT_FLOW_RAW="${OUTPUT_DIR}/temporal_iou_raw.npy"

    RESULTS_FILE="${RESULTS_FILE}" INPUT_RGB=${SSIM_PRED} OUTPUT_FLOW_VIDEO=${OUTPUT_FLOW_VIDEO} OUTPUT_FLOW_RAW=${OUTPUT_FLOW_RAW} python3 "${EVAL_ROOT}/iou_temporal.py"

    ((INDEX++)) # bash arithmetic needs (()) 
done

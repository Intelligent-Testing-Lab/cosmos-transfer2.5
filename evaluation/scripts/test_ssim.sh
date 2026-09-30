#!/bin/bash

PWD="/data/cosmos/cosmos-transfer2.5"
TEST_DATA="${PWD}/evaluation/data/test"

SSIM_PRED="${TEST_DATA}/guided_generation.mp4"
SSIM_TRUE="${TEST_DATA}/sensor_recordings/front_rgb/front_rgb.mp4"
RESULTS_FILE="${TEST_DATA}/results.json"

source "${PWD}/.venv/bin/activate"

SSIM_PRED="${SSIM_PRED}" SSIM_TRUE="${SSIM_TRUE}" RESULTS_FILE="${RESULTS_FILE}" python3 "${PWD}/evaluation/ssim.py"


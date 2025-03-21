#!/bin/bash
DATA_DIR="/$Home/robotics/dex-retargeting/manitask/data/h5"
ENV_ID="LiftPegUpright-v1"
ROBOT_ID="xarm7_allegro_right"


# 设置输入和输出路径
INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/origin"
OUTPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/merged"
OUTPUT_NAME="traj"
PATTERN="traj.h5"

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 运行合并脚本
python merge_trajectory.py \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR/$OUTPUT_NAME.h5" \
  -p "$PATTERN"

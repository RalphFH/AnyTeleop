#!/bin/bash
DATA_DIR="$HOME/robotics/dex-retargeting/manitask/data/h5"
ZARR_DIR="$HOME/robotics/dex-retargeting/manitask/data/zarr"
ROBOT_DIR="$HOME/robotics/dex-retargeting/thirdparty/ManiSkill/mani_skill/assets/robots"

ENV_ID="LiftPegUpright-v1"
ARM_ID="xarm7"
ROBOT_ID="xarm7_allegro_right"


INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/merged"
INPUT_NAME="traj"

OUTPUT_DIR="$ZARR_DIR/$ENV_ID/$ROBOT_ID"
OUTPUT_NAME="converted_data_wzx"


python maniskill_dataset_to_zarr.py \
  --pc-file "$INPUT_DIR/$INPUT_NAME.pointcloud.pd_joint_pos.physx_cpu" \
  --rgbd-file "$INPUT_DIR/$INPUT_NAME.rgb+depth+segmentation.pd_joint_pos.physx_cpu" \
  --robot-path "$ROBOT_DIR/$ARM_ID/" \
  --robot-name "$ROBOT_ID" \
  --episode-num 10 \
  --zarr-save-path "$OUTPUT_DIR/$OUTPUT_NAME.zarr" \
  --visualize
#!/bin/bash
DATA_DIR="data/h5"
ZARR_DIR="data/zarr"
ROBOT_DIR="../thirdparty/ManiSkill/mani_skill/assets/robots"

ENV_ID="PlaceSphere-v1" # 1
ARM_ID="ur5e" # 2
ROBOT_ID="ur5e_allegro_right" # 3


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
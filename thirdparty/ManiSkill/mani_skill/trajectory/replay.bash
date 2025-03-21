#!/bin/bash
DATA_DIR="$HOME/robotics/dex-retargeting/manitask/data/h5"
ENV_ID="LiftPegUpright-v1"
ROBOT_ID="xarm7_allegro_right"


# 设置输入和输出路径
INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/merged"
INPUT_NAME="traj"


# rgb+depth+segmentation | pointcloud

python replay_trajectory.py \
    --traj-path "$INPUT_DIR/$INPUT_NAME.h5" \
    --save_traj \
    --save-video \
    --obs-mode "rgb+depth+segmentation" \
    --use_env_states

# python replay_trajectory.py \
#     --traj-path "$INPUT_DIR/$INPUT_NAME.rgb+depth+segmentation.pd_joint_pos.physx_cpu.h5" \
#     --save_traj \
#     --save-video \
#     --obs-mode "pointcloud" \
#     --use_env_states
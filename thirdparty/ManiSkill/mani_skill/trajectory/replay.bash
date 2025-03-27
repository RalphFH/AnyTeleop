# !!! deprecated
DATA_DIR="../../../../manitask/data/h5"
ENV_ID="PlaceSphere-v1"
ROBOT_ID="ur5e_allegro_right"

EPISODE_RANGE="0-9"

INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/merged"
INPUT_NAME="trajectory"


# rgb+depth+segmentation | pointcloud
python replay_trajectory.py \
    --traj-path "$INPUT_DIR/$EPISODE_RANGE/$INPUT_NAME.h5" \
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
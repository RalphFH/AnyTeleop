# !!! deprecated
DATA_DIR="../../../../manitask/data/h5"
ENV_ID="OpenFaucet-v1"
ROBOT_ID="panda"

INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/"
INPUT_NAME="trajectory_1"


# rgb+depth+segmentation | pointcloud
python replay_trajectory.py \
    --traj-path "$INPUT_DIR/$INPUT_NAME.h5" \
    --save_traj \
    --no-save-video \
    --obs-mode "pointcloud" \
    --use_env_states


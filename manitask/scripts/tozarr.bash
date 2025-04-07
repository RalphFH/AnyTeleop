DATA_DIR="../data/h5"
ZARR_DIR="../data/zarr"
ROBOT_PC_DIR="../data/assets/robots/maniskill"


# modify based on the task
ENV_ID="PlaceSphere-v1"
ARM_ID="xarm7"
ROBOT_ID="xarm7_leap_right"

# modify based on the range of episodes to convert
EPISODE_RANGE="0-29"

INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/merged"
hdf5_NAME="trajectory"

OUTPUT_DIR="$ZARR_DIR/$ENV_ID/$ROBOT_ID"
OUTPUT_NAME="converted_data_wzx"

# --replay | --no-replay : whether to replay the trajectory
# --use-env-state "use-first-env-state |  "use-env-states" : whether to use the first env state or all env states
# --save-replay-video | --no-save-replay-video : whether to save the replay video
# --tozarr | --no-tozarr : whether to convert the replayed data to zarr
# --visualize | --no-visualize : whether to visualize the point cloud after conversion to zarr
python ../maniskill_dataset_to_zarr.py \
  --hdf5_dir "$INPUT_DIR/$EPISODE_RANGE/" \
  --hdf5_name "$hdf5_NAME" \
  --robot-pc-dir-path "$ROBOT_PC_DIR/$ARM_ID/$ROBOT_ID/body_pc_downsampled" \
  --episode-num 30 \
  --zarr-save-path "$OUTPUT_DIR/$EPISODE_RANGE/$OUTPUT_NAME.zarr" \
  --no-replay \
  --use-env-state "use-env-states" \
  --no-save-replay-video \
  --tozarr \
  --visualize
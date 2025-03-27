DATA_DIR="data/h5"
ZARR_DIR="data/zarr"
ROBOT_DIR="../thirdparty/ManiSkill/mani_skill/assets/robots"


# modify based on the task
ENV_ID="PlaceSphere-v1"
ARM_ID="ur5e"
ROBOT_ID="ur5e_allegro_right"

# modify based on the range of episodes to convert
EPISODE_RANGE="0-9"

INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/merged/"
hdf5_NAME="trajectory"

OUTPUT_DIR="$ZARR_DIR/$ENV_ID/$ROBOT_ID"
OUTPUT_NAME="converted_data_wzx"

# --replay | --no-replay : whether to replay the trajectory
# --save-replay-video | --no-save-replay-video : whether to save the replay video
# --tozarr | --no-tozarr : whether to convert the replayed data to zarr
# --visualize | --no-visualize : whether to visualize the point cloud after conversion to zarr
python maniskill_dataset_to_zarr.py \
  --hdf5_dir "$INPUT_DIR/$EPISODE_RANGE/" \
  --hdf5_name "$hdf5_NAME" \
  --robot-path "$ROBOT_DIR/$ARM_ID/" \
  --robot-name "$ROBOT_ID" \
  --episode-num 10 \
  --zarr-save-path "$OUTPUT_DIR/$EPISODE_RANGE/$OUTPUT_NAME.zarr" \
  --replay \
  --no-save-replay-video \
  --tozarr \
  --visualize
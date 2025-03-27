DATA_DIR="data/h5"

# modify based on the task
ENV_ID="PlaceSphere-v1"
ROBOT_ID="ur5e_allegro_right"

INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/origin"
OUTPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/merged"
OUTPUT_NAME="trajectory"
PATTERN="trajectory.h5"

# modify based on the range of episodes to convert
EPISODE_RANGE="0-9"

mkdir -p "$OUTPUT_DIR"


python -m mani_skill.trajectory.merge_trajectory \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR/$EPISODE_RANGE/$OUTPUT_NAME.h5" \
  -p "$PATTERN" \
  -r "$EPISODE_RANGE"

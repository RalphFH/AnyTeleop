DATA_DIR="data/h5"

# modify based on the task
ENV_ID="LiftPegUpright-v1"
ROBOT_ID="xarm7_leap_right"

INPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/origin"
OUTPUT_DIR="$DATA_DIR/$ENV_ID/$ROBOT_ID/merged"
OUTPUT_NAME="trajectory"
PATTERN="trajectory.h5"

# modify based on the range of episodes to convert
EPISODE_RANGE="0-29"

mkdir -p "$OUTPUT_DIR"


python -m mani_skill.trajectory.merge_trajectory \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR/$EPISODE_RANGE/$OUTPUT_NAME.h5" \
  -p "$PATTERN" \
  -r "$EPISODE_RANGE"

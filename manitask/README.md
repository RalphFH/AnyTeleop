# 🤖 ManiSkill Teleoperation Task Guide

This module focuses on retargeting human hand motion to control a robot arm + hand system within the **ManiSkill** simulation environment for completing various manipulation tasks.

---

## ⚙️ Setup & Installation

### 1. Calibrate Camera
Ensure your camera's focal length is accurately set for pose estimation.
- Modify `scaled_focal_length` in [`wilor_hand_pose3d_estimation_pipeline.py`](../thirdparty/WiLoR-mini/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py) to match your camera's `fx`.
- **Note:** If you are unsure of your camera's focal length, use the tools in `manitask/camera_clib` to calibrate it.

### 2. Install ManiSkill
ManiSkill is the simulation environment used for the tasks.
```shell
cd thirdparty/ManiSkill
pip install -e .
```
> **Requirement:** ManiSkill requires [Vulkan](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to be installed on your system.

### 3. Install Data Collection Tools
Install additional dependencies required for processing and storing teleoperation data.
```shell
pip install open3d urdf_parser_py zarr fpsample
```

---

## 🚀 Quick Start

Run the teleoperation example to control an **xArm7** with an **Allegro Hand** (Right):

```shell
python teleoperate.py --arm xarm7 --hand allegro --hand-type right
```

---

## 📊 Data Collection Pipeline

The data collection process involves teleoperation, merging episodes, point cloud processing, and converting to Zarr format.

### Step 1: Teleoperate & Collect Data
Run the teleoperation script. You can modify the `env_id` in `teleoperate.py` to switch environments (default example: `LiftPegUpright-v1`).

```shell
python teleoperate.py --arm xarm7 --hand allegro --hand-type right
```
- **Storage:** Data is saved to `manitask/data/h5/{TaskName}/{RobotID}/origin/episode_{idx}.h5`.
- **Auto-Save:** Successful episodes are automatically saved. If `max_step` is exceeded, the episode is discarded.

### Step 2: Merge Episodes
Combine multiple recorded episodes into a single dataset.
- Run: `bash manitask/scripts/merge.bash`
- **Configure:** Edit the script to specify episode ranges (e.g., `0-9`).
- **Output:** `manitask/data/h5/{TaskName}/{RobotID}/merged/`

### Step 3: Point Cloud Processing
Sample and downsample point clouds from the robot model.
- **Configure:** Update `robot_uid` in the scripts as needed.
```shell
python robot_pc_collection.py
python robot_pc_downsample.py
```

### Step 4: Replay & Convert to Zarr
Verify data by replaying and converting it to the Zarr format for training.
- Run: `bash manitask/scripts/tozarr.bash`
- **Output (Replay):** `manitask/data/h5/{TaskName}/{RobotID}/merged/`
- **Output (Zarr):** `manitask/data/zarr/{TaskName}/{RobotID}/`
- **Features:** The bash script allows toggling replay, video saving, zarr conversion, and visualization independently.

> 📝 **Note:** Ensure you update `env_id` and `robot_id` in the bash scripts (`merge.bash`, `tozarr.bash`) to match your current task and robot configuration.

---

## 🛠️ Advanced: Adding a New Robot (Arm + Hand)

Follow this guide to integrate a new robot configuration (e.g., `xarm7_shadow_right`).

### 1. Add URDF Asset
- Place `xarm7_shadow_right.urdf` in `thirdparty/ManiSkill/mani_skill/assets/robots/xarm7`.
- **Naming Convention:** `arm_hand_handtype` (e.g., `xarm7_shadow_right`).

### 2. Create Agent Class
Create a new agent file `thirdparty/ManiSkill/mani_skill/agents/robots/xarm7/xarm7_shadow.py`.
- **Template:** Copy an existing agent file to start.
- **Class Name:** Update to `XArm7Shadow`.
- **UID:** Set to `xarm7_shadow_right`.
- **Friction Config:** Update `urdf_config` links (e.g., replace `thtip` with your hand's fingertip link names).
- **Joint Names:** Set `self.arm_joint_names` and `self.hand_joint_names`. Use `/manitask/utils/URDF_check.py` to inspect the URDF.
- **Keyframes (Rest Pose):** Set the `rest` qpos. This is the `env.reset` initial pose.
    - **Crucial:** Dimensions must match total active joints.
    - **Debug:** Use `python test_qpos.py -r "robot_uid" -c "pd_joint_pos" --keyframe-actions` to visualize and adjust.
    - **Tip:** Align the robot palm with your hand's initial orientation to help the retargeting "warm start". Ensure the arm is not below the hand to avoid table collisions.
- **EE Link:** Set `self.ee_link_name` (required for some tasks like `PushCube-v1`).
- **Register:** Import the new class in the folder's `__init__.py` and `agents/robots/__init__.py`.

### 3. Register Robot in Environment
Edit the task file (e.g., `thirdparty/ManiSkill/mani_skill/envs/tasks/tabletop/PushCube-v1.py`):
- Import your new agent class.
- Add it to `SUPPORTED_ROBOTS`.
- Add the Class and UID to the `agent` attribute.
- (Optional) Add a hand camera config in `_default_human_render_camera_configs` for better visualization.

### 4. Create Retargeting Config
Create a config file `dex_retargeting/configs/manitask/xarm7/leap_right.yml`.
- **Ref:** Copy from `dex_retargeting/configs/manitask/teleop`.
- **Edit:** Update URDF path and set `add_dummy_joint: False`.

### 5. Update Constants
Edit `dex_retargeting/constants_mani.py`:
- Add entries to `ArmName`, `HandName`, `HAND_NAME_MAP`, and `ARM_NAME_MAP`.
- Define `LINK_BASE` (robot root) and `LINK_WRIST` (hand palm root).

### 6. Warm Start (Optional)
To improve initial retargeting stability, you can provide a warm start pose (usually the agent's rest keyframe).

---

## 💡 Tips & Tricks

*   **Camera Visualization:** Modify `_default_human_render_camera_configs` in the task file (e.g., `pick_cube.py`) to change visualization angles.
*   **Scene Initialization:** Adjust initial positions of robots and objects in `_initialize_episode` within the task file.
*   **Retargeting Solvers:** Explore different solver types (Position, Vector, Dexpilot) in the [WiLoR-mini README](../wilor_mini/README.md).
*   **Retargeting Configs:** All solver parameters are located in `dex_retargeting/configs/manitask/`.

![Coordinate System Explanation](docs/coordinate.jpg)

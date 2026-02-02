# 🖐️ Single Hand Teleoperation Guide

This module focuses on retargeting human hand motion to control a floating robot hand in space.

---

## ⚙️ Configuration

### 1. Camera Calibration
Ensure your camera's focal length is matched in the code.
- **File:** `thirdparty/WiLoR-mini/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py`
- **Action:** Modify `scaled_focal_length` to match your camera's `fx`.

### 2. Hand Scaling
- **Parameter:** `scaling_factor` in the optimizer.
- **Purpose:** Matches the size of the human hand to the robot hand.
- **Note:** Adjust this carefully if you are using a custom robot model to ensuring accurate retargeting.

---

## 🚀 Running the Example

Launch the teleoperation script with your desired robot and retargeting method.

### Available Options
- **`--robot-name`**: `panda`, `shadow`, `allegro`
- **`--retargeting-type`**: `position`, `vector`, `dexpilot`
- **`--hand-type`**: `right`, `left`

### Example Commands

**1. Position Control (Shadow Hand, Right)**
Allows the robot hand to follow the human hand's position freely in 3D space.
```shell
python3 teleoperate.py --robot-name shadow --retargeting-type position --hand-type right
```

**2. DexPilot Control (Shadow Hand, Left)**
Uses relative vector-based retargeting for precise finger articulation.
```shell
python3 teleoperate.py --robot-name shadow --retargeting-type dexpilot --hand-type left
```

---

## 💡 Key Concepts

### Retargeting Optimizers
This project leverages three types of optimizers provided by `dex-retargeting`:

1.  **Position**:
    -   **Logic:** Solves for `qpos` based on the reference 3D position of each link.
    -   **Best For:** Spatial tracking. The robot hand moves freely in space, following the human hand's absolute position.

2.  **Vector**:
    -   **Logic:** Solves for `qpos` based on reference vectors (e.g., Vector from Wrist to Knuckles).
    -   **Best For:** Robust pose matching that ignores absolute position.

3.  **DexPilot**:
    -   **Logic:** A complex solver using fingertip positions and vectors (Fingertip Link - Palm Center).
    -   **Best For:** High-fidelity teleoperation tasks requiring precise relative finger movements.

> **Note:** Since `vector` and `dexpilot` rely on relative positions, the translation degrees of freedom of the dummy joint (wrist movement) are effectively ignored in these modes. **Use `position` mode if you want the robot hand to translate in space.**

### Hand Detection & 3D Estimation
While standard examples often use MediaPipe (local hand coordinates), this project integrates **WiLoR-mini** for superior performance:

-   **Global Context:** WiLoR outputs the hand coordinate system origin relative to the **Camera Coordinate System**.
-   **Full 3D Pose:** By combining the origin with local keypoints, we obtain the absolute 3D coordinates of hand keypoints in the camera frame.
-   **Transformation:** These coordinates are then transformed into the simulation (Sapien) world frame for controlling the robot.

<div align="center">

# AnyTeleoperate

**A Robust, Single-Camera Teleoperation Framework for Dexterous Hands**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Overview

This project builds upon and enhances the dexterous hand teleoperation capabilities of [AnyTeleop](https://arxiv.org/pdf/2307.04577). By integrating advanced pose estimation and retargeting optimizations, it offers a accessible and stable solution for robotic hand control.

### ✨ Key Features

- 🚀 **Robust to Occlusion**: Handles partial hand occlusions effectively during operation.
- 🦾 **6-DoF Wrist Pose**: Supports accurate calculation and mapping of wrist pose.
- 💻 **Minimal Hardware**: Requires only a single RGB camera and a standard laptop.

---

## 🛠️ Installation

### 1. Create Environment
Create a clean conda environment with Python 3.9:
```shell
conda create -n retarget python=3.9
conda activate retarget
```

### 2. Install Core Optimizer
Install `dex-retargeting` which provides the optimization solvers for retargeting hand keypoints to robot joint positions:
```shell
pip install dex_retargeting loguru
```

### 3. Install Project & Dependencies
Clone this repository and install the required dependencies:
```shell
git clone --recursive https://github.com/Ralph-cong/AnyTeleop.git
cd AnyTeleop
pip install -e ".[example]"
pip install tyro pyyaml sapien==3.0.0b0 pyrealsense2 numpy-quaternion
```

### 4. Install Local Submodules
This project uses modified versions of **WiLoR-mini** (for hand detection/3D estimation) and **Ultralytics**.

**Install WiLoR-mini:**
```shell
# Clone into thirdparty directory
git clone https://github.com/warmshao/WiLoR-mini.git thirdparty/WiLoR-mini
# Install in editable mode
pip install -e thirdparty/WiLoR-mini
```


### 4. Troubleshooting
If you encounter library linking errors during execution, you may need to export the library path:
```shell
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/retarget/lib/python3.9/site-packages/cmeel.prefix/lib:$LD_LIBRARY_PATH
```

---

## 🎮 Usage Examples

### 1. Single Robot Hand Teleoperation
Retarget human hand motion to a floating robot hand.
👉 **[View Tutorial](wilor_mini/README.md)**

### 2. ManiSkill Task Teleoperation (Recommended)
Retarget human hand motion to control a robot arm + hand system to complete manipulation tasks in ManiSkill.
👉 **[View Tutorial](manitask/README.md)**

---

## 📜 Acknowledgements

This project is developed based on:
*   [dex-retargeting](https://github.com/yzqin/dex-hand-teleop)
*   [WiLoR-mini](https://github.com/warmshao/WiLoR-mini)

<div align="center">
  <h1 align="center"> AnyTeleoperate </h1>
</div>



## Installation
```shell
conda create -n retarget python=3.9
```
The [dex-retargeting project](https://github.com/dexsuite/dex-retargeting) provides three types of optimizer which help to compute the qpos given the 3D information of the keypoints of our hand. The command of installing it is as follows.
```shell
pip install dex_retargeting loguru
```

Then clone this repository and install some dependecies about dex-retargeting.

```shell
git clone https://github.com/Ralph-cong/AnyTeleop.git
cd dex-retargeting
pip install -e ".[example]"
```
For the hand detector and 3D estimation part, I implement with [WiLor-mini](https://github.com/warmshao/WiLoR-mini), so you should install some package about it.

```shell
pip install git+https://github.com/warmshao/WiLoR-mini
```

## Examples

### Teleoperating For the single free robot hand
[Tutorial on retargeting from human hand to a single free robot hand](wilor_mini/README.md)

tips: Free means there is a free 6D dummy joint of the hand but without arm.

### Teleoperating in the maniskill env

coding...

## Retargeting examples from [dex-retargeting project](https://github.com/yzqin/dex-hand-teleop) [Optional reading section]

### Retargeting from human hand video

This type of retargeting can be used for applications like teleoperation,
e.g. [AnyTeleop](https://yzqin.github.io/anyteleop/).

[Tutorial on retargeting from human hand video](example/vector_retargeting/README.md)


### Retarget from hand object pose dataset

![teaser](example/position_retargeting/hand_object.webp)

This type of retargeting can be used post-process human data for robot imitation,
e.g. [DexMV](https://yzqin.github.io/dexmv/).

[Tutorial on retargeting from hand-object pose dataset](example/position_retargeting/README.md)

## Joint Orders for Retargeting

URDF parsers, such as ROS, physical simulators, real robot driver, and this repository, may parse URDF files with
different joint orders. To use `dex-retargeting` results with other libraries, handle joint ordering explicitly **using
joint names**, which are unique within a URDF file.

Example: Using `dex-retargeting` with the SAPIEN simulator

```python
from dex_retargeting.seq_retarget import SeqRetargeting

retargeting: SeqRetargeting
sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
retargeting_joint_names = retargeting.joint_names
retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

# Use the index map to handle joint order differences
sapien_robot.set_qpos(retarget_qpos[retargeting_to_sapien])
```

This example retrieves joint names from the SAPIEN robot and `SeqRetargeting` object, creates a mapping
array (`retargeting_to_sapien`) to map joint indices, and sets the SAPIEN robot's joint positions using the retargeted
joint positions.





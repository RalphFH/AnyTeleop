## Setup
1. 在安装的`wilor_mini`包中，将`site-packages/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py` 中的scaled_focal_length修改自身摄像头的fx

2. 安装ManiSkill
    ```shell
    cd thirdparty/ManiSkill
    pip install -e .
    ```
    maniskill还需要安装[Vulkan](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)



## Commands for running the example 
```shell
python3 panda_pick_cube.py --robot-name panda --retargeting-type position --hand-type left
```


## Tips
1. 相机设置在`thirdparty/ManiSkill/mani_skill/envs/tasks/tabletop/pick_cube.py`中的`_default_human_render_camera_configs()`可修改各种参数
2. 每个episode重置的机器人、物体和目标点的位置量在`thirdparty/ManiSkill/mani_skill/envs/tasks/tabletop/pick_cube.py`中的`_initialize_episode()`可修改，其中机器人需进入`self.table_scene.initialize(env_idx)`中
3. hand_detector中的`points[:,2] = points[:,2] + 0.08 points[:,0] = points[:,0] + 0.2` 是我们映射过去的第一帧对应的机器手所在位置，不过注意这里的0.08和0.2在optimizer中还会乘以一个config中的scaling factor，这个scaling factor是为了匹配人手大小和机器手大小的。这个初始帧对应的初始位置求解出来的qpos只要和前面说的_initialize_episode()所设定的Sapien中设置的qpos不能完全对上，就会有跳变存在。
4. config文件在`dex_retargeting/configs/manitask/panda_gripper_left.yml`下，主要是配置一些retargeting求解的参数
5. 如果对原项目`dex-retargeting`提供的三种求解器感兴趣，可前往wilor_mini/README.md中查看

![坐标说明](docs/coordinate.jpg)


## Setup
1. 在安装的`wilor_mini`包中，将`site-packages/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py` 中的scaled_focal_length修改自身摄像头的fx,如果不知道摄像头的焦距，可以使用manitask/camera_clib中的文件标定出摄像头的焦距。

2. 安装ManiSkill
    ```shell
    cd thirdparty/ManiSkill
    pip install -e .
    ```
    maniskill还需要安装[Vulkan](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

3. 收集数据
    ```shell
    pip install open3d urdf_parser_py zarr fpsample
    ```
   

## Commands for running the example 
```shell

python3 teleoperate.py --arm xarm7 --hand allegro --hand-type right

```

## pipeline
### 添加新的“arm+hand” set
以`xarm7_shadow_right`为例：
1. 添加urdf
    - 在`thirdparty/ManiSkill/mani_skill/assets/robots/xarm7`中，添加`xarm7_shadow_right.urdf`
    - urdf文件命名遵循`arm_hand_handtype`的规范
2. 添加`agent`类
    - 在`thirdparty/ManiSkill/mani_skill/agents/robots/xarm7`中，创建`xarm7_shadow.py`(可先直接复制一个其它的机器人作为template)
    - 修改类名，如`XArm7Shadow`, 规范命名uid(遵循`arm_hand_handtype`的规范), 如`xarm7_shadow_right`
    - `urdf_config=dict(...` 里的link是在修改材料的摩擦力，如`thtip`就是leap的指尖link, 需修改为对应机器手的link名称
    - 设置`self.arm_joint_names`，`self.hand_joint_names`（这个可以通过'/manitask/utils/URDF_check.py'加载对应的urdf文件查看）
    - 设置keyframes中`rest`帧的qpos，这里注意qpos维度要和前面的查看的activate joint(arm+hand)数量一致,这个是env.reset后的初始位姿，很重要,可以在完成后续步骤后用'/manitask/utils/test_qpos.py'，调试修改 | `python test_qpos.py -r "robot_uid" -c "pd_joint_pos" --keyframe-actions`测试调整
        - 在retarget时，视频第一帧的人手掌根关键点会和机械手的手掌根`LINK_WRIST`重叠；
        - retarget解算时用warm_start加速初始迭代，这个keyframe将会作为初始解`last_pos`，因此最好让机械手掌朝向和你习惯的第一帧人手朝向对齐一些
        - 由于retarget在解算qpos时不会考虑除机器人自身外的障碍(如桌子)，初始的这个值最好不要让arm部分在hand下面，否则容易造成hand未能碰到桌面时arm就碰到了，然后hand下不去
    - 设置`self.ee_link_name`，这个主要看task需不需要用到了，像`PushCube-v1`这个环境就需要用到这个来计算奖励
    - 在`thirdparty/ManiSkill/mani_skill/assets/robots/xarm7`的`__init__.py`中import一下这个类，同时如果是新的arm的话在`robots/__init__.py`里也import一下，这样才能注册这个agent
3. 在对应的任务环境中添加该机器人，这里以`PushCube-v1`为例
    - 在`thirdparty/ManiSkill/mani_skill/envs/tasks/tabletop/PushCube-v1.py`中，import刚刚创建的agent类，分别在`SUPPORTED_ROBOTS`和`agent`属性中添加上类和uid
    - 如果有新的hand，则在`_default_human_render_camera_configs`中添加对应的手部相机方便操作
4. 添加对应的config文件
    - 在`dex_retargeting/configs/manitask/xarm7`下创建`leap_right.yml`(可以参考dex_retargeting/configs/manitask/teleop中相同机器手的)，注意命名为`arm/hand_handtype`,然后修改urdf路径，add_dummy_joint置False
5. 修改`dex_retargeting/constants_mani.py`，主要用于寻找各个config文件什么的
    - 在ArmName，HandName，HAND_NAME_MAP，ARM_NAME_MAP添加对应的arm和hand
    - LINK_BASE为机器人的root_link，如果有新的arm需要添加, LINK_WRIST为机器手的掌根，如有新的hand需要添加
6. 可选：为了让机器人最开始求解qpos的时候可以顺利求解，我们可以给个warm_start，这里我使用的是agent的keyframe，也就是env.reset后机器人的qpos

### 数据采集
以“lift_peg_upright”为例, 不同环境在`teleoperate.py`里修改`env_id`即可
1. teleoperate采数据
```shell
python3 teleoperate.py --arm xarm7 --hand allegro --hand-type right
```
- 一次采集一个episode,数据存放在 `manitask/data/h5/LiftPegUpright-v1/xarm7_allegro_right/origin`内，以“episode_{idx}”区分各episode(程序自动命名)
- success会自动保存，如果超过env的max_step会退出并且不会保存东西

2. 多个episode融合
- 执行`manitask`中的`merge.bash`(bash里可指定融合的episode序号数 0-9， 10-19之类)
- 输出在 `manitaskdata/h5/LiftPegUpright-v1/xarm7_allegro_right/merged/0-9`中

3. replay h5 file & convert to zarr (可指定episode序号数)
- 执行`manitask`中的`tozarr.bash`
- replay 输出在 `manitask/data/h5/LiftPegUpright-v1/xarm7_allegro_right/merged/0-9`中，
- zarr 输出在`manitask/data/zarr/LiftPegUpright-v1/xarm7_allegro_right/0-9`中
- bash 中可以控制是否replay，replay时是否存储replay视频，是否将replay后的h5转为zarr,转为zarr后是否可视化显示点云等，从而实现 `replay h5 file`和`convert to zarr`的良好解耦。


注：不同环境收集到的数据的命名都是自动的，不同env,robot要记得修改两个bash里env_id 和 robot_id




## Tips
1. 相机设置在`thirdparty/ManiSkill/mani_skill/envs/tasks/tabletop/pick_cube.py`中的`_default_human_render_camera_configs()`可修改各种参数,这个相机只是可视化使用
2. 每个episode重置的机器人、物体和目标点的位置量在`thirdparty/ManiSkill/mani_skill/envs/tasks/tabletop/pick_cube.py`中的`_initialize_episode()`可修改，其中机器人需进入`self.table_scene.initialize(env_idx)`中
4. config文件在`dex_retargeting/configs/manitask/`下，主要是配置一些retargeting求解的参数
5. 如果对原项目`dex-retargeting`提供的三种求解器感兴趣，可前往wilor_mini/README.md中查看

![坐标说明](docs/coordinate.jpg)


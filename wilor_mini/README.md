## Setup
- 在安装的`wilor_mini`包中，将`site-packages/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py` 中的scaled_focal_length修改自身摄像头的fx
- 注意scaling_factor在optimizer中的缩放作用,这个缩放因子用于从人手的大小匹配到机器手的大小，如果使用自己的机器人模型需注意


## Commands for running the example
- robot-name: panda shadow allegro
- retargeting-type: position vector dexpilot 
```shell
python3 teleoperate.py --robot-name shadow --retargeting-type position --hand-type right
```
```shell
python3 teleoperate.py --robot-name shadow --retargeting-type dexpilot --hand-type left
```

## Tips

### Three types of optimizer provided by dex-retargeting project
1. position: 给定参考的各个link的pos，去求解输出对应的qpos
2. vector: 给定参考的vector (比如在shadow hand里是各个指尖+指腹的pos - 腕部的pos),然后求解对应的qpos
3. dexpolit: 给定参考的vector(这个比较复杂，是各个指尖的link+掌心指尖的pos两两相减)，然后求解对应的qpos

由于vector和dexpolit是根据相对位置求解的，所以dummy joint的平移自由度法没用上的，用position可以让机器人的手在空间跟随人手自由运动。

### Hand detector
example/vector_retargeting 中提供的示例使用media_pipe进行手部关键点识别，但是是基于手部局部坐标系的，因此在此项目里使用了WiLor-mini进行手部识别和关键点位置估计。

WiLor可以输出手部坐标系原点在相机坐标系下的坐标，因此利用其与输出的关键点在手部坐标系下的坐标相加即可获得手部关键点在相机坐标系下的坐标。再变换到Sapien的坐标系下即可。
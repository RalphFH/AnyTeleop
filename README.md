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
pip install -e ".[example]"
```
For the hand detector and 3D estimation part, I implement with [WiLor-mini](https://github.com/warmshao/WiLoR-mini), so you should install some package about it.

```shell
pip install git+https://github.com/warmshao/WiLoR-mini
```

And during my experiment, I met an error message, but I just remember its solution:
```shell
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/retarget/lib/python3.9/site-packages/cmeel.prefix/lib:$LD_LIBRARY_PATH
```

## Examples

### Teleoperating For a single fly robot hand
[Tutorial on retargeting from human hand to a single fly robot hand](wilor_mini/README.md)


### Teleoperating task in the maniskill env (recommended)
[Tutorial on retargeting from human hand to the maniskill robot to complete some tasks](manitask/README.md)






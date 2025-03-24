import gymnasium as gym
import mani_skill.envs
import matplotlib.pyplot as plt

import cv2
import torch
import numpy as np
import time


"""
['pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos',
 'pd_ee_delta_pose', 'pd_ee_pose', 'pd_joint_target_delta_pos',
 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 'pd_joint_vel',
 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel']
"""



env = gym.make(
    "LiftPegUpright-v1",#here are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    robot_uids="xarm6_shadow_right", #panda_wristcam
    obs_mode="rgb+depth+segmentation", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
    # parallel_in_single_scene=True,
    render_mode="human", # rgb_array | human 
)


# """
# env <class 'mani_skill.utils.registration.TimeLimitWrapper'>
# env.env <class 'gymnasium.wrappers.order_enforcing.OrderEnforcing'>
# env.env.env / env.unwrapped <class 'mani_skill.envs.tasks.tabletop.pick_cube.PickCubeEnv'>
# """

agent = env.unwrapped.agent # <class 'mani_skill.agents.robots.panda.panda.Panda'>
robot = agent.robot # <class 'mani_skill.utils.structs.articulation.Articulation'>
qpos = agent.keyframes["rest"].qpos

# for link in robot.get_links():
#     print(link.get_name())

env.reset()
# obs, reward, terminated, truncated, info=env.step(qpos)
# print("obs",(obs).keys()) # torch.Tensor or dict | ['agent', 'extra', 'sensor_param', 'sensor_data']
# print("reward",reward) # torch.Tensor | torch.Size([1])
# print("terminated",terminated) # torch.Tensor | torch.Size([1]) | tensor([False])
# print("truncated",truncated) # torch.Tensor | torch.Size([1]) | tensor([False])
# print("info",info['success']) # dict | ['elapsed_steps', 'success']



while True:

    env.render()
    # obs, reward, terminated, truncated, info=env.step(qpos)
    # print("truncated",truncated.item())
    time.sleep(0.1)

# try:
#     while True:
#         img = env.render().squeeze(0)  # 获取环境渲染的图像
        
#         # 如果图像是 Torch Tensor，转换为 NumPy
#         if isinstance(img, torch.Tensor):
#             img = img.cpu().numpy()

#         # OpenAI Gym 的 render() 通常返回 RGB 图像 (H, W, 3)，但 cv2 需要 BGR
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  

#         # 显示图像
#         cv2.imshow("Environment", img)

#         # `waitKey(1)` 控制帧率，按 `q` 退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# except KeyboardInterrupt:
#     pass
# finally:
#     env.close()
#     cv2.destroyAllWindows()  # 关闭窗口


   

# print("qpos",type(qpos))
# panda_hand = robot.links_map["panda_hand"] # <class 'mani_skill.utils.structs.link.Link'>
# panda_hand_pose=robot.links_map["panda_hand"].pose.raw_pose.detach().squeeze(0).numpy()[:3]
# root_pose=robot.links_map["panda_link0"].pose.raw_pose.detach().squeeze(0).numpy()[:3]
# print(panda_hand_pose,root_pose)

# controller = agent.controller # <class 'mani_skill.agents.controllers.base_controller.CombinedController'>
# print(controller.controllers.values())

# robot = env.unwrapped.agent.robot
# root_pose=robot.links_map["panda_link0"].pose.raw_pose.detach().squeeze(0).numpy()[:3]
# print("root_pose",root_pose)



# print("Observation space", env.observation_space) # dict | ['agent', 'extra', 'sensor_param', 'sensor_data']
# print("Action space", env.action_space) # 7 DoF robot arm | array of 7 floats
# obs, _ = env.reset(seed=0)



# obs, _ = env.reset(seed=0) # reset with a seed for determinism
# done = False
# for _ in range(500):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated | truncated
#     # print(f"Obs shape: {obs.shape}, Reward shape {reward.shape}, Done shape {done.shape}")
#     env.render()  # a display is required to render
# env.close()




"""
camera
"""
# human_render_cameras = env.unwrapped.scene.human_render_cameras
# camera = human_render_cameras["render_camera_left"]
# camera.capture()
# rgb = camera.get_obs(
#     rgb=True, depth=False, segmentation=False, position=False
# )["rgb"].squeeze(0).detach().cpu().numpy()
# plt.imshow(rgb)
# plt.show()

# for name,camera in human_render_cameras.items():
#     params = camera.get_params() # dict | ['extrinsic_cv', 'cam2world_gl', 'intrinsic_cv']
#     print("camera_uid",name)
#     extrinsic_cv = params["extrinsic_cv"]
#     intrinsic_cv = params["intrinsic_cv"]
#     cam2world_gl = params["cam2world_gl"]

#     print("extrinsic_cv",extrinsic_cv)
#     print("intrinsic_cv",intrinsic_cv)
#     print("cam2world_gl",cam2world_gl)
import gymnasium as gym
import mani_skill.envs
import matplotlib.pyplot as plt

import cv2
import torch
import numpy as np

"""
['pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos',
 'pd_ee_delta_pose', 'pd_ee_pose', 'pd_joint_target_delta_pos',
 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 'pd_joint_vel',
 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel']
"""

# import mani_skill.examples.demo_robot as demo_robot_script
# demo_robot_script.main()


def world_to_image(world_points, extrinsic_cv, intrinsic_cv):
    """
    Project 3D world points to 2D image coordinates
    
    Args:
        world_points: Nx3 array of points in world coordinates (x, y, z)
                     or single [x, y, z] point
        extrinsic_cv: 4x4 extrinsic matrix (OpenCV convention)
        intrinsic_cv: 3x3 intrinsic matrix
        
    Returns:
        Nx2 array of (u, v) image coordinates in pixels
    """    

        
    image_points = np.zeros((world_points.shape[0], 2), dtype=int)
    
    for i, point in enumerate(world_points):
        # Convert to homogeneous coordinates
        world_point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
        
        # Transform from world to camera coordinates
        camera_point = extrinsic_cv @ world_point_homogeneous
        
        # Skip points behind the camera (negative z)
        if camera_point[2] <= 0:
            image_points[i] = [-1, -1]  # Invalid point marker
            continue
        
        # Normalize by dividing by z (perspective division)
        x_cam = camera_point[0] / camera_point[2]
        y_cam = camera_point[1] / camera_point[2]
        
        # Project to image plane using intrinsic parameters
        u = intrinsic_cv[0, 0] * x_cam + intrinsic_cv[0, 2]
        v = intrinsic_cv[1, 1] * y_cam + intrinsic_cv[1, 2]
        
        image_points[i] = [int(u), int(v)]
    
    return image_points  # Return array of points


def draw_projected_point(rgb_image, world_points, extrinsic_cv, intrinsic_cv, 
                         marker_size=10, color=(255, 0, 0)):
    
    if isinstance(world_points, torch.Tensor):
        world_points = world_points.detach().cpu().numpy()

    if len(world_points.shape) == 1:
        world_points = world_points.reshape(1, 3)
    
    # Project the 3D point to 2D image coordinates
    images_points = world_to_image(world_points, extrinsic_cv, intrinsic_cv)
    
    # Convert tensor to numpy if needed
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.detach().cpu().numpy()
    
    # Create a copy of the image to avoid modifying the original
    img_with_point = rgb_image.copy()
    
    # Make sure the point is within image bounds
    h, w = img_with_point.shape[:2]
    for u, v in images_points:
        if 0 <= u < w and 0 <= v < h:
            # Draw a circle at the projected point
            cv2.circle(img_with_point, (u, v), marker_size, color, -1)
            # print(f"Point projected to image coordinates: ({u}, {v})")
        else:
            print(f"Warning: Projected point ({u}, {v}) is outside image bounds ({w}x{h})")
    
    return img_with_point
    

def visualize_point_on_all_cameras(env, world_point):
    human_render_cameras = env.unwrapped.scene.human_render_cameras
    
    for name, camera in human_render_cameras.items():
        print(f"\nProjecting point onto camera: {name}")
        
        # Get camera parameters
        params = camera.get_params()
        extrinsic_cv = params["extrinsic_cv"].squeeze(0).detach().cpu().numpy()
        intrinsic_cv = params["intrinsic_cv"].squeeze(0).detach().cpu().numpy()
        
        # Capture camera image
        camera.capture()
        rgb = camera.get_obs(rgb=True, depth=False, segmentation=False, position=False)["rgb"].squeeze(0)
        
        # Project and draw the point
        img_with_point = draw_projected_point(rgb, world_point, extrinsic_cv, intrinsic_cv)
        
        # Display the result
        plt.figure(figsize=(10, 8))
        plt.title(f"Camera: {name} with projected world point (0.2, 0, 0.1)")
        plt.imshow(img_with_point)
        plt.axis('on')
        plt.grid(True, alpha=0.3)
        plt.show()

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    robot_uids="panda", 
    obs_mode="rgb", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
    # parallel_in_single_scene=True,
    render_mode="rgb_array", # rgb_array | human 
)

# """
# env <class 'mani_skill.utils.registration.TimeLimitWrapper'>
# env.env <class 'gymnasium.wrappers.order_enforcing.OrderEnforcing'>
# env.env.env / env.unwrapped <class 'mani_skill.envs.tasks.tabletop.pick_cube.PickCubeEnv'>
# """

env.reset()



try:
    while True:
        img = env.render().squeeze(0)  # 获取环境渲染的图像
        
        # 如果图像是 Torch Tensor，转换为 NumPy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        # OpenAI Gym 的 render() 通常返回 RGB 图像 (H, W, 3)，但 cv2 需要 BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  

        # 显示图像
        cv2.imshow("Environment", img)

        # `waitKey(1)` 控制帧率，按 `q` 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    env.close()
    cv2.destroyAllWindows()  # 关闭窗口

# human_render_cameras = env.unwrapped.scene.human_render_cameras

# world_point = np.array([0.2, 0.0, 0.1])


# visualize_point_on_all_cameras(env, world_point)

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
   
# agent = env.unwrapped.agent # <class 'mani_skill.agents.robots.panda.panda.Panda'>
# robot = agent.robot # <class 'mani_skill.utils.structs.articulation.Articulation'>
# panda_hand = robot.links_map["panda_hand"] # <class 'mani_skill.utils.structs.link.Link'>
# panda_hand_pose=robot.links_map["panda_hand"].pose.raw_pose.detach().squeeze(0).numpy()[:3]
# root_pose=robot.links_map["panda_link0"].pose.raw_pose.detach().squeeze(0).numpy()[:3]
# print(panda_hand_pose,root_pose)

# controller = agent.controller # <class 'mani_skill.agents.controllers.base_controller.CombinedController'>
# print(controller.controllers.values())

# robot = env.unwrapped.agent.robot
# root_pose=robot.links_map["panda_link0"].pose.raw_pose.detach().squeeze(0).numpy()[:3]
# print("root_pose",root_pose)

# print("active joint",type(robot.active_joints))


# print(type(env.env.env))
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



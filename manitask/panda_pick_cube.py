import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger

from dex_retargeting.constants_mani import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_detector import HandDetector

import gymnasium as gym
import mani_skill.envs

import torch
from filter_module import Filter

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


def draw_points_on_tiled_image(tiled_image, world_points, camera_extrinsics, camera_intrinsics, 
                              marker_size=5, colors=None):
    """
    Draw 3D world points projected onto each camera view in a tiled image
    
    Args:
        tiled_image: The composite image from tile_images()
        world_points: Nx3 array of 3D world points to project
        camera_extrinsics: Dictionary mapping camera names to 4x4 extrinsic matrices
        camera_intrinsics: Dictionary mapping camera names to 3x3 intrinsic matrices
        marker_size: Size of the marker to draw
        colors: Optional list of colors for each point
    
    Returns:
        Image with projected points drawn on it
    """
    # Make a copy of the tiled image
    img_points = tiled_image.copy()

    if isinstance(world_points, torch.Tensor):
        world_points = world_points.detach().cpu().numpy()

    if len(world_points.shape) == 1:
        world_points = world_points.reshape(1, 3)
    
    # If no colors provided, use default
    if colors is None:
        colors = [(0, 0, 255) for _ in range(len(world_points))]
    elif len(colors) != len(world_points):
        raise ValueError("Number of colors must match number of points")
    
    # Get camera names
    camera_names = list(camera_extrinsics.keys())

     # Calculate the position of each camera view in the tiled image
    camera_offsets = {}
    current_x_offset = 0
    
    for cam_name in camera_names:

        img_w = camera_intrinsics[cam_name][0, 2]*2
        img_h = camera_intrinsics[cam_name][1, 2]*2

        camera_offsets[cam_name] = (current_x_offset, 0)  # Assuming single row tiling
        current_x_offset += img_w
    
    # For each camera, project points and draw on the tiled image
    for camera_name in camera_names:
        extrinsic = camera_extrinsics[camera_name]
        intrinsic = camera_intrinsics[camera_name]
        x_offset, y_offset = camera_offsets[camera_name]
        
        # Project all points at once
        image_points = world_to_image(world_points, extrinsic, intrinsic)
        
        # Draw each valid point with its offset in the tiled image
        for i, (u, v) in enumerate(image_points):
            # Skip invalid points (behind camera or outside frame)
            if u < 0 or v < 0 or u >= img_w or v >= img_h:
                continue
                
            # Apply offset for this camera's position in the tiled image
            tiled_u = int(u + x_offset)
            tiled_v = int(v + y_offset)
            
            # Draw the point
            cv2.circle(img_points, (tiled_u, tiled_v), marker_size, colors[i], -1)
    
    return img_points
    

def start_retargeting(isStart, isEnd, queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    
    # Load retargeting optimizer
    # override = dict(add_dummy_free_joint=True)
    config = RetargetingConfig.load_from_file(config_path)
    retargeting = config.build()
    retargeting.warm_start_panda()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    retargeting_type = retargeting.optimizer.retargeting_type
    
    # Load robot
    env = gym.make(
        "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="rgb", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="rgb_array", # rgb_array | human | all
    )

    robot = env.unwrapped.agent.robot
    root_pose=robot.links_map["panda_link0"].pose.raw_pose.detach().squeeze(0).numpy()[:3]
    obs,_ = env.reset(seed=0)
    viewer = env.render()
    human_render_cameras = env.unwrapped.scene.human_render_cameras
    
    # Get camera parameters
    camera_extrinsics = {}
    camera_intrinsics = {}
    for cam_name, camera in human_render_cameras.items():
        params = camera.get_params()
        camera_extrinsics[cam_name] = params["extrinsic_cv"].squeeze(0).detach().cpu().numpy()
        # camera_extrinsics[cam_name] = params["cam2world_gl"].squeeze(0).detach().cpu().numpy()
        camera_intrinsics[cam_name] = params["intrinsic_cv"].squeeze(0).detach().cpu().numpy()


    # Different robot loader may have different orders for joints, compute retargeting_to_sapien for the mapping
    # sapien_joint_names = [joint.get_name() for joint in robot.active_joints()]
    # retargeting_joint_names = retargeting.joint_names
    # retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    filter = Filter(
            filter_type="exponential",  # "moving_average" "exponential", "median", "none"
            window_size=3, # used in moving average filter
            alpha=0.7, # used in exponential filter
            jump_threshold=0.2 
        )

    # Load hand detector
    detector = HandDetector(hand_type=hand_type)
    cv2.namedWindow("realtime_retargeting", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Environment", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("realtime_retargeting", 640, 480)
    cv2.resizeWindow("Environment", 1536, 512)

    cv2.moveWindow("realtime_retargeting", 960, 100)  # x=50, y=100
    cv2.moveWindow("Environment", 512, 700)
    
    isStart.set()
    while True:
        # frame_start_time = time.time()  # Start tracking each frame's retargeting time
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # (480, 640, 3)
        except Empty:
            logger.error(f"Fail to fetch image from camera in 5 secs. Please check your web camera device.")
            return

        # 1. Hand Detection and 3D Pose Estimation
        # detect_start_time = time.time()
        _, keypoints_pos = detector.detect(rgb)
        # detect_end_time = time.time()
        # detect_duration = detect_end_time - detect_start_time
        # logger.info(f"Time taken for detection: {detect_duration:.4f} seconds")
        
        # 2. Drawing skeleton
        detect_img = detector.draw_skeleton_on_image(bgr, style="default")
        cv2.imshow("realtime_retargeting", detect_img)
        
        # print("predict key points",keypoints_pos.shape)
        # 3. Retargeting 
        if keypoints_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = keypoints_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = keypoints_pos[task_indices, :] - keypoints_pos[origin_indices, :]
    
            is_success,filtered_points = filter.filter(ref_value)
            ref_value = filtered_points if is_success else ref_value
            qpos = retargeting.retarget(ref_value)
            # robot.set_qpos(qpos)
            keypoints_3d = ref_value*config.scaling_factor + root_pose
            
            if qpos[-2] > 0.01:
                qpos[-2] = 1
            if qpos[-2] < 0.01:
                qpos[-2] = -1
            
            action = qpos[:-1]
            # print(f"action: {action}")
            # print(f"retarget: {qpos[-1]} action: {action[-1]}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated 

        img = env.render().squeeze(0).detach().cpu().numpy()
        img_with_points = draw_points_on_tiled_image(
                                img, keypoints_3d, camera_extrinsics, camera_intrinsics, 
                                marker_size=8)
        img_with_points = cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR)
        cv2.imshow("Environment", img_with_points)

    

        if done:
            isEnd.set()
            cv2.destroyAllWindows()
            break
        
        # End of each frame 



def produce_frame(isStart, isEnd, queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    isStart.wait()
    while cap.isOpened() and not(isEnd.is_set()):
        success, image = cap.read()
        time.sleep(1 / 12.0)
        if not success:
            continue
        queue.put(image)
    
    

def main(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType, camera_path: Optional[str] = None
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent /"thirdparty"/"ManiSkill"/"mani_skill"/"assets" / "robots" 

    isStart = multiprocessing.Event()
    isEnd = multiprocessing.Event()
    queue = multiprocessing.Queue(maxsize=5)
    producer_process = multiprocessing.Process(target=produce_frame, args=(isStart, isEnd, queue, camera_path))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(isStart, isEnd, queue, str(robot_dir), str(config_path)))

    try:
        producer_process.start()
        consumer_process.start()

        producer_process.join()
        consumer_process.join()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止进程.")
    finally:
        producer_process.terminate()
        consumer_process.terminate()
        producer_process.join()
        consumer_process.join()
    
    print("done")


if __name__ == "__main__":
    tyro.cli(main)

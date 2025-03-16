import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger
import sapien
import pyrealsense2 as rs

from dex_retargeting.constants_mani import *
from dex_retargeting.retargeting_config import RetargetingConfig

import gymnasium as gym
import mani_skill.envs

import torch
from utils.filter_module import Filter
from utils.hand_detector import HandDetector
from utils.reproject import draw_points_on_tiled_image
    

def start_retargeting(isStart, isEnd, queue: multiprocessing.Queue, robot_dir: str, config_path: str, robot_uid:str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    
    arm, hand, hand_type = robot_uid.split("_")

    # Load retargeting optimizer
    config = RetargetingConfig.load_from_file(config_path)
    retargeting = config.build()


    retargeting_type = retargeting.optimizer.retargeting_type
    
    # Load robot
    env = gym.make(
        "PushCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        robot_uids=robot_uid,
        obs_mode="rgb", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="rgb_array", # rgb_array | human | all
    )

    agent = env.unwrapped.agent
    robot = agent.robot
    root_pose=robot.links_map[LINK_BASE[arm]].pose.raw_pose.detach().squeeze(0).numpy()[:3]
    wrist_pose = robot.links_map[LINK_WRIST[hand]].pose.raw_pose.detach().squeeze(0).numpy()[:3]
    obs,_ = env.reset(seed=0)
    viewer = env.render()
    human_render_cameras = env.unwrapped.scene.human_render_cameras
    
    init_qpos = agent.keyframes["rest"].qpos
    if hand == "panda":
        init_qpos = init_qpos[:-1]
    retargeting.warm_start_manitask(robot_uid=robot_uid, qpos=init_qpos)

    # Get camera parameters
    camera_extrinsics = {}
    camera_intrinsics = {}
    for cam_name, camera in human_render_cameras.items():
        params = camera.get_params()
        camera_extrinsics[cam_name] = params["extrinsic_cv"].squeeze(0).detach().cpu().numpy()
        # camera_extrinsics[cam_name] = params["cam2world_gl"].squeeze(0).detach().cpu().numpy()
        camera_intrinsics[cam_name] = params["intrinsic_cv"].squeeze(0).detach().cpu().numpy()

    filter = Filter(
            filter_type="exponential",  # "moving_average" "exponential", "median", "none"
            window_size=3, # used in moving average filter
            alpha=0.7, # used in exponential filter
            jump_threshold=0.2 
        )

    # Load hand detector
    trans_scale = 1.2/config.scaling_factor
    detector = HandDetector(hand_type=hand_type,trans_scale=trans_scale)

    cv2.namedWindow("realtime_retargeting", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Environment", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("realtime_retargeting", 640, 480)
    cv2.resizeWindow("Environment", 1536, 512)

    cv2.moveWindow("realtime_retargeting", 960, 100)  # x=50, y=100
    cv2.moveWindow("Environment", 512, 700)

    # initial_position = np.array([0.48, 0.0, 0.1]).reshape(1, 3) # initial position of the hand in the robot root frame
    initial_position = (wrist_pose - root_pose)

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.name for joint in robot.active_joints]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

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

        # 3. Retargeting 
        keypoints_3d = None
        done = False
        if keypoints_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
            continue

        else:
            keypoints_pos = keypoints_pos + initial_position/config.scaling_factor

            indices = retargeting.optimizer.target_link_human_indices
            ref_value = keypoints_pos[indices, :]
    
            is_success,filtered_points = filter.filter(ref_value)
            ref_value = filtered_points if is_success else ref_value

            keypoints_3d = ref_value*config.scaling_factor + root_pose

            qpos = retargeting.retarget(ref_value)
            # robot.set_qpos(qpos)
            # print("qpos",qpos)
            if hand == "panda":
                if qpos[-2] > 0.018:
                    qpos[-2] = 1
                else: 
                    qpos[-2] = -1
                action = qpos[:-1]
            else:
                action = qpos[retargeting_to_sapien]
            # print(f"retarget: {qpos[-1]} action: {action[-1]}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated 

        link_pose = None
        points_robot = []
        for i,target_link in enumerate(config.target_link_names):
            link_pose = robot.links_map[target_link].pose.raw_pose.detach().squeeze(0).numpy()[:3]
            points_robot.append(link_pose)
        
        print("points_robot",points_robot[0])
        all_points = np.vstack([keypoints_3d, np.array(points_robot)])
        num_points = len(points_robot)
        colors = [(0, 255, 0)] * num_points + [(255, 0, 0)] * num_points

        img = env.render().squeeze(0).detach().cpu().numpy()
        img_with_points = draw_points_on_tiled_image(
                                img, all_points, camera_extrinsics, camera_intrinsics, 
                                marker_size=5, colors=colors)
        img_with_points = cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Environment", img_with_points)

    

        if cv2.waitKey(2) & done :
            print("done!")
            isEnd.set()
            cv2.destroyAllWindows()
            break
        
        # End of each frame 



def produce_frame(isStart, isEnd, queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path == "rs":
        # Initialize RealSense pipeline
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        pipe.start(config)
    else:
        if camera_path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(camera_path)

    isStart.wait()
    try:
        while not(isEnd.is_set()):
            if camera_path == "rs":
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
            else:
                if not cap.isOpened():
                    break
                success, image = cap.read()
                if not success:
                    continue
            
            time.sleep(1 / 30.0)
            queue.put(image)
    finally:
        if camera_path == "rs":
            pipe.stop()
        else:
            cap.release()


def main(
    arm: ArmName, hand: HandName, hand_type: HandType, camera_path: Optional[str] = None
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        arm: The identifier for the robot arm. This should match one of the default supported robot arms
        hand: The identifier for the robot hand. This should match one of the default supported robot hands.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path,robot_uid = get_default_config_path(arm, hand, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent /"thirdparty"/"ManiSkill"/"mani_skill"/"assets" / "robots" 
    isStart = multiprocessing.Event()
    isEnd = multiprocessing.Event()
    queue = multiprocessing.Queue(maxsize=5)
    producer_process = multiprocessing.Process(target=produce_frame, args=(isStart, isEnd, queue, camera_path))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(isStart, isEnd, queue, str(robot_dir), str(config_path),robot_uid))

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

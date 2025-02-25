import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_detector import HandDetector

import gymnasium as gym
import mani_skill.envs



def start_retargeting(isStart, isEnd, queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    
    # Load retargeting optimizer
    # override = dict(add_dummy_free_joint=True)
    config = RetargetingConfig.load_from_file(config_path)
    retargeting = config.build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    retargeting_type = retargeting.optimizer.retargeting_type
    
    # Load robot
    env = gym.make(
        "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="rgb", # there is also "state_dict", "rgbd", ...
        control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )

    # Different robot loader may have different orders for joints, compute retargeting_to_sapien for the mapping
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    # Load hand detector
    detector = HandDetector(hand_type=hand_type)

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
        _, joint_pos, keypoint_2d = detector.detect(rgb)
        # detect_end_time = time.time()
        # detect_duration = detect_end_time - detect_start_time
        # logger.info(f"Time taken for detection: {detect_duration:.4f} seconds")
        
        # 2. Drawing skeleton
        detect_img = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", detect_img)
        
        # 3. Retargeting 
        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            qpos = retargeting.retarget(ref_value)
            

    

        if cv2.waitKey(2) & 0xFF == ord("q"):
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
        time.sleep(1 / 15.0)
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
    # robot_dir = Path(__file__).absolute().parent.parent / "assets" / "robots" / "hands"
    robot_dir = Path("/home/chenshan/anaconda3/envs/retarget/lib/python3.9/site-packages/mani_skill/assets/robots")
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

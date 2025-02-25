import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_detector import HandDetector

from dex_retargeting import yourdfpy as urdf
import tempfile



def start_retargeting(event,queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    override = dict(add_dummy_free_joint=True)
    config = RetargetingConfig.load_from_file(config_path,override=override)
    retargeting = config.build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = HandDetector(hand_type=hand_type)
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")  

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0 #金属度
    render_mat.roughness = 0.9 #粗糙度
    render_mat.specular = 0.8 #镜面反射
    # scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.5, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)



    robot_name = filepath.stem
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True

    retargeting_type = retargeting.optimizer.retargeting_type
    

    # if retargeting_type != "POSITION":
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    # print("Config:add_dummy_free_joint", config.add_dummy_free_joint)
    if config.add_dummy_free_joint == True:
        print("add dummy free joint !")
        robot_urdf = urdf.URDF.load(str(filepath), add_dummy_free_joints=True, build_scene_graph=False)
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{robot_name}"
        robot_urdf.write_xml_file(temp_path)
        robot = loader.load(temp_path)
    else:
        robot = loader.load(filepath)

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))

    # Different robot loader may have different orders for joints, compute retargeting_to_sapien for the mapping
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    # print(sapien_joint_names)
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    event.set()
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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
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
            robot.set_qpos(qpos[retargeting_to_sapien])

        
        # Rendering the scene
        for _ in range(2):
            viewer.render()
        
        # End of each frame 



def produce_frame(event,queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    event.wait()
    while cap.isOpened():
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
    robot_dir = Path(__file__).absolute().parent.parent / "assets" / "robots" / "hands"
    event = multiprocessing.Event()
    queue = multiprocessing.Queue(maxsize=5)
    producer_process = multiprocessing.Process(target=produce_frame, args=(event,queue, camera_path))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(event,queue, str(robot_dir), str(config_path)))


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

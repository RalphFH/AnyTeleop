import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import transforms3d as tf3d
from mani_skill.envs.tasks.my_laptop.open_laptop_task import OpenLaptopEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.utils.wrappers.record import RecordEpisode
import tqdm
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_of_laptop
import numpy as np


def solve(env: OpenLaptopEnv, seed=None, debug=False, vis=False, visualize_target_grasp_pose=True, print_env_info=False):
    
    # 1) reset
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped

    # Motion planner only supports pd_joint_pos and pd_joint_pos_vel control mode
    assert env_unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], \
        f"control_mode must be pd_joint_pos or pd_joint_pos_vel, got {env_unwrapped.control_mode}"


    #create motion planner
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env_unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=visualize_target_grasp_pose,
        print_env_info=print_env_info,
        joint_vel_limits=2,
        joint_acc_limits=1,
    )

    laptop_pose = env_unwrapped.laptop_articulation.get_pose()
    
    approaching = np.array([0, 0, -1])
    
    #compute grasp pose
    FINGER_LENGTH = 0.02
    
    grasp_info = compute_grasp_info_of_laptop(laptop_pose, approaching, FINGER_LENGTH)
    
    closing, center = grasp_info["closing"], grasp_info["center"]
   
    grasp_pose = env.unwrapped.agent.build_grasp_pose(grasp_info["approaching"], closing, center)
   
    
    rot_pose1 = sapien.Pose(
        p=[0, 0, 0],
        q=euler2quat(np.deg2rad(40), 0, 0)
    )
    
    tcp_pose = env_unwrapped.agent.tcp.pose  

    # --- reach1 ---
    planner.close_gripper()
    offset1 = np.array([-0.04, 0.01, -0.01])
    reach1_pose = sapien.Pose(
    p = np.array(grasp_pose.p) + offset1,
    q = grasp_pose.q  
        )
    res = planner.move_to_pose_with_screw(reach1_pose)
    if res == -1:
        return res
  
    # --- rotate ---
    tcp_pose = env_unwrapped.agent.tcp.pose
    planner.close_gripper()
    tcp_pose1 = tcp_pose * rot_pose1
    res = planner.move_to_pose_with_screw(tcp_pose1)
    if res == -1:
        return res
 
    
    # --- half open ---
    grasp_pose = grasp_pose * rot_pose1
    offset3 = np.array([0.25, 0, 0.16])
    half_open_pose = sapien.Pose(
    p = np.array(grasp_pose.p) + offset3,
    q = grasp_pose.q 
        )
    res = planner.move_to_pose_with_screw(half_open_pose,refine_steps=10)
    if res == -1:
        return res
  
  
    # --- Release & Retreat ---
    retreat_pose = half_open_pose * sapien.Pose([-0.08, 0, 0])
    res =planner.move_to_pose_with_screw(retreat_pose,refine_steps=10)
  
    env_unwrapped.has_been_successful=False
    if res == -1:
        return res

    planner.close()
    return res



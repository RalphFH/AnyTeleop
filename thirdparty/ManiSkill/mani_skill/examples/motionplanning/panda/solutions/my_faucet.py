import gymnasium as gym
import numpy as np
import sapien.core as sapien
import trimesh
from tqdm import tqdm
from transforms3d.axangles import *
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks import OpenFaucetEnv
from transforms3d.quaternions import mat2quat

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_of_faucet,
)


def solve(env:OpenFaucetEnv, seed=None, debug=False, vis=False, visualize_target_grasp_pose=True, print_env_info=False):
    
    # 1) reset
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped

    # Motion planner only supports pd_joint_pos and pd_joint_pos_vel control mode
    assert env_unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], \
        f"control_mode must be pd_joint_pos or pd_joint_pos_vel, got {env_unwrapped.control_mode}"

    # 2) create motion planner
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
    faucet_pose = env_unwrapped.faucet_articulation.get_pose()
    approaching = np.array([0, 0, -1])

    # 3) compute grasp pose
    FINGER_LENGTH = 0.02
    grasp_info = compute_grasp_info_of_faucet(faucet_pose, approaching, FINGER_LENGTH)
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.unwrapped.agent.build_grasp_pose(grasp_info["approaching"], closing, center)
    rot_pose1 = sapien.Pose(
        p=[0, 0, 0],
        q=euler2quat(0, 0, np.deg2rad(-90))
    )
    

    # --- touch ---
    planner.close_gripper()
    res = planner.move_to_pose_with_screw(grasp_pose,refine_steps=20)
    if res == -1:
        return res

    
     # --- rotate to 90 degree ---
    offset2 = np.array([0.1, -0.15, 0])
    full_open_pose = sapien.Pose(
    p = np.array(grasp_pose.p) + offset2,
    q = grasp_pose.q  
    )
    full_open_pose=full_open_pose * rot_pose1
    res = planner.move_to_pose_with_screw(full_open_pose,refine_steps=20)
    
    if res == -1:
        return res
    
    
    # --- Release & Retreat ---
    retreat_pose = full_open_pose * sapien.Pose([0, -0.02, 0])
    res =planner.move_to_pose_with_screw(retreat_pose,refine_steps=10)
    env_unwrapped.has_been_successful=False
    if res == -1:
        return res

    planner.close()
    return res


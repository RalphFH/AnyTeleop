import argparse
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

from transforms3d import euler
import numpy as np
import time

def main():
   
    env = gym.make(
        "OpenLaptop-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",  # or "human",
        reward_mode="sparse",
    )

    for seed in tqdm(range(10)):  
        res = solve(env, seed=seed, debug=False, vis=True)
        env.reset()
       # print("[INFO] Last step result:", res[-1])

    env.close()


def solve(env: OpenLaptopEnv, seed=None, debug=False, vis=False):
    
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
        visualize_target_grasp_pose=True,
        print_env_info=True,
        joint_vel_limits=1,
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
        q=euler2quat(np.deg2rad(80), 0, 0)
    )
    rot_pose2 = sapien.Pose(
        p=[0, 0, 0],
        q=euler2quat(0, np.deg2rad(90), 0)
    )
    #grasp_pose = grasp_pose * rot_pose1
    print(f"电脑的抓取位置是：{grasp_pose}")
    reverse_rot_pose = sapien.Pose(
        p=[0, 0, 0],
        q=euler2quat(0, 0, 0)
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
    print(f"电脑一开始的角度是：{env_unwrapped.laptop_articulation.get_qpos()}")
    grasp_pose = grasp_pose * rot_pose1
    offset3 = np.array([0.25, 0, 0.16])
    half_open_pose = sapien.Pose(
    p = np.array(grasp_pose.p) + offset3,
    q = grasp_pose.q  # 保持原来的旋转不变
)
   # half_open_pose = half_open_pose * reverse_rot_pose
    res = planner.move_to_pose_with_screw(half_open_pose,refine_steps=10)
    print(f"电脑half open开启的角度是：{env_unwrapped.laptop_articulation.get_qpos()}")
    if res == -1:
        return res
  
  
    # --- Release & Retreat ---
    retreat_pose = half_open_pose * sapien.Pose([-0.03, 0, 0])
    res =planner.move_to_pose_with_screw(retreat_pose,refine_steps=10)
  
    print(f"电脑最终开启的角度是：{env_unwrapped.laptop_articulation.get_qpos()}")
    env_unwrapped.has_been_successful=False
    if res == -1:
        return res

    planner.close()
    return res

if __name__ == "__main__":
    main()


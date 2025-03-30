import numpy as np
import sapien

from mani_skill.envs.tasks import PlaceSphereEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def solve(env: PlaceSphereEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )

    env = env.unwrapped

    # Get sphere OBB and compute grasp pose
    sphere_obb = get_actor_obb(env.obj)
    approaching = np.array([0, 0, -1])  # Approach from above
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        sphere_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.obj.pose.sp.p)
    offset = sapien.Pose([0, 0, 0.02])  # Slight offset for safe grasping
    grasp_pose = grasp_pose * offset

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])  # Approach slightly above the sphere
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift sphere to safe height
    # -------------------------------------------------------------------------- #
    lift_height = 0.2
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Move above the bin
    # -------------------------------------------------------------------------- #
    bin_pos = env.bin.pose.sp.p
    above_bin_pose = sapien.Pose(bin_pos + np.array([0, 0, lift_height]))
    above_bin_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(above_bin_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Lower sphere into the bin
    # -------------------------------------------------------------------------- #
    place_pose = sapien.Pose(bin_pos + np.array([0, 0, env.radius + env.block_half_size[0]]))
    place_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    res = planner.move_to_pose_with_screw(place_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Release sphere
    # -------------------------------------------------------------------------- #
    planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Retreat to a safe position
    # -------------------------------------------------------------------------- #
    retreat_pose = above_bin_pose
    res = planner.move_to_pose_with_screw(retreat_pose)
    if res == -1: return res

    planner.close()
    return res
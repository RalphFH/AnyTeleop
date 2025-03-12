import mani_skill.examples.demo_robot as demo_robot_script
demo_robot_script.main()

# import gymnasium as gym
# import mani_skill.envs
# import numpy as np
# import time

# env = gym.make(
#     "PushCube-v1",#here are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
#     num_envs=1,
#     robot_uids="xarm7_leap_right", 
#     obs_mode="state", # there is also "state_dict", "rgbd", ...
#     control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
#     # parallel_in_single_scene=True,
#     render_mode="human", # rgb_array | human 
# )


# env.reset()

# while True:
#     # env.reset()
#     env.render()
#     time.sleep(0.02)

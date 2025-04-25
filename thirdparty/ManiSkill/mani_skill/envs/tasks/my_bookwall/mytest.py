import gymnasium as gym
import time
from mani_skill.envs.tasks.my_bookwall.book_wall_env import * 
# env = gym.make_vec(
#     "OpenFaucet_mul-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
#     num_envs=5,
#     obs_mode="state", # there is also "state_dict", "rgbd", ...
#     control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
#     render_mode="human"
# )
# print("Observation space", env.observation_space)
# print("Action space", env.action_space)

# obs, _ = env.reset(seed=0) # reset with a seed for determinism
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = bool(terminated[0]) or bool(truncated[0])
#     env.render()  # a display is required to renderdone = terminated or truncated
#    #time.sleep(10)
# env.close()


env = gym.make(
    "BookWall-v1",
    num_envs=1,
    parallel_in_single_scene=True,
    render_mode="human",
    
)
env.reset()
for i in range(10):
    env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        env.step(action)
        env.render()

import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgb", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    # parallel_in_single_scene=True,
    render_mode="human"
)

print(type(env.env.env))
print("Observation space", env.observation_space) # dict | ['agent', 'extra', 'sensor_param', 'sensor_data']
print("Action space", env.action_space) # 7 DoF robot arm | array of 7 floats
obs, _ = env.reset(seed=0)



# obs, _ = env.reset(seed=0) # reset with a seed for determinism
# done = False
# for _ in range(500):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated | truncated
#     # print(f"Obs shape: {obs.shape}, Reward shape {reward.shape}, Done shape {done.shape}")
#     env.render()  # a display is required to render
# env.close()
import os
import gymnasium as gym
import time
import traceback
import numpy as np
from mani_skill.envs.tasks.my_faucet.my_faucet_mulenvs import *
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.solutions.my_faucet import solve        
#from mani_skill.examples.motionplanning.panda.solutions.my_laptop import solve   
      
#change the solve function to solve the different tasks

def main():
    data_dir = os.path.expanduser("~/ManiSkill/data/my_faucet_data")
    os.makedirs(data_dir, exist_ok=True)
    env_id = "OpenFaucet-v1"                                 #"OpenFaucet-v1" ,"OpenLaptop-v1"
    obs_mode = "rgb+depth+segmentation"
    control_mode = "pd_joint_pos"
    render_mode = "rgb_array"
    reward_mode = "sparse"
    sim_backend = "auto"
    shader = "default"
    vis=True                                                 #if the device has no screen, set vis=False.
    visualize_target_grasp_pose=True 
    print_env_info=False


    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        reward_mode=reward_mode,
        sim_backend=sim_backend,
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
        viewer_camera_configs=dict(shader_pack=shader),
    )
    
    max_episode_steps=500
    traj_name = time.strftime("%Y%m%d_%H%M%S")
    env = RecordEpisode(
        env,
        output_dir=os.path.join(data_dir, env_id, "motionplanning"),
        trajectory_name=traj_name,
        save_video=True,
        video_fps=30,
        save_on_reset=False,
        source_type="motionplanning",
        source_desc="my_laptop_run record",
        max_steps_per_video=max_episode_steps
    )

    only_count_success = True  
    num_episodes = 10
    seed = 2
    succeed_count = 0
    # first reset the environment
    env.reset(seed=seed)
    print(f"Motion Planning Running on {env_id}")

    try:
        for i in range(num_episodes):
            env.reset()
            print(f"Episode {i} running on {env_id} ...")
            result = solve(env, seed=seed, debug=False, vis=vis, 
                        visualize_target_grasp_pose=visualize_target_grasp_pose, print_env_info=print_env_info)
            # print("Motion planning solution result:")
            # print(result)
            # result is a tuple/list，result[-1] looks like {"elapsed_steps":..., "success":tensor([True])}
            success = False
            if isinstance(result, (list, tuple)) and len(result) > 0:
                final_info = result[-1]
                if isinstance(final_info, dict) and "success" in final_info:
                    #count success
                    success_tensor = final_info["success"]
                    success = bool(success_tensor.item())  
            
            if only_count_success and not success:
                print("Episode failed, not saving trajectory or video.")
                env.flush_trajectory(save=False)
                env.flush_video(save=False)
            else:
                print("Episode succeeded, saving trajectory and video.")
                env.flush_trajectory()
                env.flush_video()
                succeed_count += 1

            seed += 1
          

    except Exception as e:
        print("Motion planning solution encountered an error:")
        traceback.print_exc()
    finally:
        print(f"Motion planning run on {env_id} finished, {succeed_count}/{num_episodes} succeeded")
        env.close()
        time.sleep(0.5)

if __name__ == "__main__":
    main()

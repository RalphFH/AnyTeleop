
import os
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
import time
import transforms3d as tf3d


current_dir = os.path.dirname(__file__)
@register_env("OpenLaptop-v1", max_episode_steps=500)

class OpenLaptopEnv(BaseEnv):
   
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02

    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,reward_mode="sparse" ,sim_backend = "physx_cuda", **kwargs):
        self.has_been_successful = False
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.laptop_target_angle=-0.35
        super().__init__(*args, robot_uids=robot_uids,reward_mode=reward_mode,sim_backend=sim_backend, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=224,
                height=224,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.7, 0.7, 0.7], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
    #auto called in env.make() to load the agent
    def _load_agent(self, options: dict): # type: ignore
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))


    #auto called in env.make() to load the scene
    def _load_scene(self, options: dict):
        
        self.table_scene = TableSceneBuilder(
        env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        loader = self.scene.create_urdf_loader()
        loader.scale = 1 
        urdf_relative_path = "../../../assets/my_laptop/laptop_pack2/mobility.urdf"
        urdf_path = os.path.join(current_dir, urdf_relative_path)
        articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
        # set physical parameters for the faucet
        loader.set_density(5)
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        builder = articulation_builders[0]
        
    
        base_pos = np.array([0, 0.15, 0.1])
        random_offset_xy = np.random.uniform(-0.05, 0.05, size=2)
        new_pos = np.array([base_pos[0] + random_offset_xy[0],
                            base_pos[1] + random_offset_xy[1],
                            base_pos[2]])
        builder.initial_pose = sapien.Pose(p=new_pos.tolist())
        
        self.laptop_articulation = builder.build(name="laptop_articulation")
        # set friction for the faucet
        for j in self.laptop_articulation.get_joints():
            j.set_friction(0.1)
        
    
    #auto called in env.reset() to initialize the episode
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
      
        with torch.device(self.device):
            
            
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            super()._initialize_episode(env_idx, options)
            laptopqpos = np.zeros(self.laptop_articulation.dof)

            hinge_idx = 0  
            # qpos[hinge_idx]=0 is 90° open so 10° is -80°
            laptopqpos[hinge_idx] = -80 * np.pi / 180.0

            self.laptop_articulation.set_qpos(laptopqpos)

             #set according to a lot of experiments.
            base_pos = np.array([-0.075, 0.25, 0.1])
            random_offset_x = np.random.uniform(-0.025, 0.025, size=2)
            random_offset_y = np.random.uniform(-0.35, 0.35, size=2) 
            new_pos = np.array([base_pos[0] + random_offset_x[0],
                                base_pos[1] + random_offset_y[1],
                                base_pos[2]])
            self.laptop_articulation.set_pose(sapien.Pose(p=new_pos.tolist()))

            
            
            # panda initial pose
            if self.robot_init_qpos_noise > 0:
                noise = np.random.uniform(
                    -self.robot_init_qpos_noise, self.robot_init_qpos_noise, self.agent.robot.dof
                )
                new_qpos = np.array([0.09, -0.85, -0.04, -2, -0.07, 1.2, -0.7, 0 ,0]) + noise     #set accroding to a lot of experiments.
                self.agent.robot.set_qpos(new_qpos)
            
            
    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position)
        current_angle = self.laptop_articulation.get_qpos()[0]  
        is_opened = current_angle > self.laptop_target_angle

        return {"success": is_opened}
    #auto called in env.step() to get the observation
    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        return obs
    
    
    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward=0
        if self.evaluate()["success"]:
            if self.has_been_successful==True:
                reward = 0
            else:
                self.has_been_successful = True
                reward = 1
        return reward
    

    #we do not need to use the following functions in this task because it is not RL task.
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        return 0
    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return 0
from typing import Any, Dict, Union
import os
import numpy as np
import sapien
import torch
import torch.random
import trimesh
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from transforms3d.euler import euler2quat
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import  Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.utils.building import articulations
from mani_skill.utils.structs import SimConfig, GPUMemoryConfig
import multiprocessing as mp
from mani_skill.utils.structs.pose import Pose as BatchPose

current_dir = os.path.dirname(__file__)

@register_env("OpenLaptop_mul-v1", max_episode_steps=500)

class OpenLaptopMulEnv(BaseEnv):
    """
    **Task Description:**
    open the faucet by rotating the handle to a target angle.

    **Randomizations:**
    the pos of the faucet, the initial qpos of the robot

    **Success Conditions:**
    the faucet handle is rotated to the target angle
    """

    SUPPORTED_ROBOTS = ["panda"]

    # Specify some supported robot types
    agent: Panda
    
    # set some commonly used values
    goal_range = 2   # degrees
   
   
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,num_envs= 10,parallel_in_single_scene=True ,reward_mode="sparse" ,sim_backend = "physx_cuda",  **kwargs):      #num_envs= 5,parallel_in_single_scene=True,
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.num_envs=num_envs
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.laptop_target_angle=-0.17
        super().__init__(*args, robot_uids=robot_uids,reward_mode=reward_mode,num_envs=num_envs,parallel_in_single_scene=parallel_in_single_scene,sim_backend=sim_backend, **kwargs)    #
        self.has_been_successful = False

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
        # spacing between scenes
        spacing=3.0,  
        gpu_memory_config=GPUMemoryConfig(
            found_lost_pairs_capacity=2**26,
            max_rigid_patch_count=2**20
        ),
       
    )

    @property
    def _default_sensor_configs(self):
        # registers one 224x224 camera looking at the robot, cube, and target
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
        
         pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
         return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))


    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(
        env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        loader = self.scene.create_urdf_loader()
        loader.scale = 1.5 
        urdf_relative_path = "../../../assets/my_laptop/laptop_pack2/mobility.urdf"
        urdf_path = os.path.join(current_dir, urdf_relative_path)
        articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
        # set physical parameters for the faucet
        loader.set_density(5)
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        builder = articulation_builders[0]
        
        base_pos = np.array([0.05, 0.35, 0])
        random_offset_x = np.random.uniform(-0.05, 0.02, size=(self.num_envs,))
        random_offset_y = np.random.uniform(-0.1, 0.15, size=(self.num_envs,))  
        new_pos = np.stack([base_pos[0] + random_offset_x,
                            base_pos[1] + random_offset_y,
                            base_pos[2]*np.ones(self.num_envs)],axis=1)
        p_tensor = torch.tensor(new_pos, dtype=torch.float32, device=self.device)
        batched_pose = BatchPose.create_from_pq(p=p_tensor, q=None, device=self.device)
        builder.initial_pose = batched_pose
      
        
        self.laptop_articulation = builder.build(name="laptop_articulation")
        # set friction for the faucet
        for j in self.laptop_articulation.get_joints():
            j.set_friction(1)

    
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """
        every time a new episode is started, this method is called to reset the environment to its initial state
        """
        with torch.device(self.device):
            
            super()._initialize_episode(env_idx, options)
            # a little randomization for the initial position
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            #set the open angle for the laptop, which is 10 degrees open 
            hinge_idx = 0
            dof_array = self.laptop_articulation.dof  
            dof_per_env = int(dof_array[0].item()) 
            qpos_np = np.zeros((b, dof_per_env), dtype=np.float32)
            qpos_np[:, hinge_idx] = -60.0 * np.pi / 180.0 
            qpos_tensor = torch.from_numpy(qpos_np).to(self.device)
            self.laptop_articulation.set_qpos(qpos_tensor)
            
            #base pose and random offset for the laptop are determined according to experiment, do not change
            base_pos = np.array([0, 0.35, 0.1])
            random_offset_x = np.random.uniform(-0.05, 0.02, size=(self.num_envs,))
            random_offset_y = np.random.uniform(-0.1, 0.15, size=(self.num_envs,))  
            new_pos = np.stack([base_pos[0] + random_offset_x,
                                base_pos[1] + random_offset_y,
                                base_pos[2]*np.ones(self.num_envs)],axis=1)
            p_tensor = torch.tensor(new_pos, dtype=torch.float32, device=self.device)
            batched_pose1 = BatchPose.create_from_pq(p=p_tensor, q=None, device=self.device)
            self.laptop_articulation.set_pose(batched_pose1)
           
              
            if self.robot_uids == "panda":        
                noise_np = np.random.uniform(
                low=-self.robot_init_qpos_noise,
                high=self.robot_init_qpos_noise,
                size=(b, dof_per_env)
                )
                #qpos is the position of every joint of the robot,also determined according to experiments, do not change
                base_qpos = np.array([0.09, -0.85, -0.04, -2, -0.07, 1.2, -0.7, 0 ,0])
                base_qpos_tiled = np.tile(base_qpos, (b, 1))  
                new_qpos_np = base_qpos_tiled + noise_np 
                new_qpos_tensor = torch.from_numpy(new_qpos_np).float().to(self.device)
                self.agent.robot.set_qpos(new_qpos_tensor)
    
                
    def evaluate(self):
            
        current_angle = self.laptop_articulation.get_qpos()  # shape: (num_envs, dof)
        hinge_angles = current_angle[:, 0]  # shape: (num_envs,)
        is_opened = hinge_angles > self.laptop_target_angle  # shape: (num_envs,)
        return {"success": is_opened}

        

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
     
     
    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
            # success: shape (num_envs,)
        success = self.evaluate()["success"]  
        if not isinstance(self.has_been_successful, torch.Tensor):
            self.has_been_successful = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        newly_success = success & (~self.has_been_successful)
        
        reward = newly_success.float()
        
        self.has_been_successful = self.has_been_successful | success
        
        return reward
                        
                
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
        
    
        
        


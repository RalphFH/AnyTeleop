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
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.agents.robots import XArm7Allegro, XArm7Shadow, XArm7Leap
from mani_skill.agents.robots import XArm6Allegro, XArm6Shadow 
from mani_skill.agents.robots import UR5eShadow, UR5eAllegro, UR5eLeap
from mani_skill.agents.robots import IIwa7Allegro
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig, SceneConfig
from mani_skill.utils.building import articulations
from mani_skill.utils.structs import SimConfig, GPUMemoryConfig
import multiprocessing as mp
from mani_skill.utils.structs.pose import Pose as BatchPose
from mani_skill.utils.quater import product

current_dir = os.path.dirname(__file__)

@register_env("OpenLaptop_mul-v1", max_episode_steps=500)      #this 500 is used for eval, in single env, 300 is used for collection

class OpenLaptopMulEnv(BaseEnv):
    """
    **Task Description:**
    open the laptop by rotating the handle to a target angle.

    **Randomizations:**
    the pos of the laptop, the initial qpos of the robot

    **Success Conditions:**
    the laptop screen is rotated to the target angle
    """

    SUPPORTED_ROBOTS = ["panda", "fetch",
                        "xarm7_allegro_right", "xarm7_shadow_right", "xarm7_leap_right",
                        "xarm6_allegro_right", "xarm6_shadow_right",
                        "ur5e_shadow_right", "ur5e_allegro_right", "ur5e_leap_right",
                        "iiwa7_allegro_right"] 

    # Specify some supported robot types
    agent: Union[Panda, Fetch,
                 XArm7Allegro, XArm7Shadow, XArm7Leap,
                 XArm6Allegro, XArm6Shadow,
                 UR5eShadow, UR5eAllegro, UR5eLeap,
                 IIwa7Allegro]        
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,num_envs= 10,parallel_in_single_scene=True ,reward_mode="sparse" ,sim_backend = "physx_cuda",  **kwargs):      #num_envs= 5,parallel_in_single_scene=True,
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.num_envs=num_envs
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.laptop_target_angle=-0.17
        super().__init__(*args, robot_uids=robot_uids,reward_mode=reward_mode,num_envs=num_envs,parallel_in_single_scene=parallel_in_single_scene,sim_backend=sim_backend, **kwargs)    #
        

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
       scene_config=SceneConfig(enable_ccd = True, solver_position_iterations= 20,             
            solver_velocity_iterations= 1  )
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
        
        top_down = sapien_utils.look_at([-0.18, 0.0, 0.4], [-0.06, 0.0, 0])
        left_side = sapien_utils.look_at([-0.05, 0.27, 0.15], [-0.05, 0.1, 0.15]) 

        cam_config = []
        cam_config.append(CameraConfig("top_down", top_down, 512, 512, 80*np.pi/180, 0.01, 100))



        if "xarm7" in self.robot_uids:
            q2 = [np.cos(15*np.pi/180), 0, np.sin(15*np.pi/180),0]

            cam_config.append(CameraConfig(
                                uid="arm_cam",
                                pose=sapien.Pose(p=[-0.13, 0 , 0.2], q=q2),
                                width=512,
                                height=512,
                                fov=70*np.pi/180,
                                near=0.01,
                                far=100,
                                entity_uid="link7",
                            )
            )     

        if "panda_wrist" in self.robot_uids:
            cam_config.append(CameraConfig(
                                uid="hand_cam",
                                pose=sapien.Pose(p=[0, 0 , 0], q=[1, 0, 0, 0]),
                                width=512,
                                height=512,
                                fov=80*np.pi/180,
                                near=0.01,
                                far=100,
                                entity_uid="camera_link",
                            ))                           
        elif "allegro" in self.robot_uids:
            q1 = [np.cos(35*np.pi/180), 0 , 0 , -np.sin(35*np.pi/180)]
            q2 = [np.cos(30*np.pi/180), 0 , -np.sin(30*np.pi/180),0]
            q3 = [np.cos(10*np.pi/180), np.sin(10*np.pi/180),0,0]
            q = product(q3,product(q2,q1)) 
            cam_config.append( CameraConfig(
                                uid="hand_cam",
                                pose=sapien.Pose(p=[-0.018, 0.2 , -0.02], q=q1),
                                width=512,
                                height=512,
                                fov=70*np.pi/180,
                                near=0.01,
                                far=100,
                                entity_uid="base_link_hand",
                            ))
        elif "shadow" in self.robot_uids:
            q1 = [0.7044, 0.06166, 0.06166, -0.7044]
            q2 = [np.cos(-30*np.pi/180), np.sin(-30*np.pi/180), 0, 0]
            q = product(q2,q1)
            cam_config.append(CameraConfig(
                                uid="arm_cam",
                                pose=sapien.Pose(p=[0, 0.23 , 0.18], q=q),
                                width=512,
                                height=512,
                                fov=1.57,
                                near=0.01,
                                far=100,
                                entity_uid="forearm",
                            ))
            q3 = [np.cos(-80*np.pi/180), 0 , 0 , np.sin(-80*np.pi/180)]
            q4 = [np.cos(30*np.pi/180), np.sin(30*np.pi/180), 0, 0]
            q = product(q4,q3)
            cam_config.append(CameraConfig(  
                                uid="hand_cam", 
                                pose=sapien.Pose(p=[0.18, 0.05 , 0.1], q=q),
                                width=512,
                                height=512,
                                fov=1.57,
                                near=0.01,
                                far=100,
                                entity_uid="palm",
                            ))  
        elif "leap" in self.robot_uids:
            q1 = [np.cos(45*np.pi/180), 0 , np.sin(45*np.pi/180), 0]
            q2 = [np.cos(-45*np.pi/180), np.sin(-45*np.pi/180), 0 , 0]
            q3 = [np.cos(-40*np.pi/180), 0 , np.sin(-40*np.pi/180), 0]
            q = product(q3,product(q2,q1))
            cam_config.append(CameraConfig(
                                uid="hand_cam",
                                pose=sapien.Pose(p=[-0.01, 0.22 , -0.04], q=q),
                                width=512,
                                height=512,
                                fov= 70*np.pi/180,
                                near=0.01,
                                far=100,
                                entity_uid="palm_lower",
                            ))

        cam_config.append( CameraConfig("scene_left_camera", left_side, 512, 512, 80*np.pi/180, 0.01, 100))

        return cam_config

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
        # set physical parameters for the laptop
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
        # set friction for the laptop
        for j in self.laptop_articulation.get_joints():
            j.set_friction(1)

    
    #auto called in env.reset() to initialize the episode
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
       
        with torch.device(self.device):
            
            super()._initialize_episode(env_idx, options)
            # a little randomization for the initial position
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
      
            hinge_idx = 0
            dof_array = self.laptop_articulation.dof  
            dof_per_env = int(dof_array[0].item()) 
            qpos_np = np.zeros((b, dof_per_env), dtype=np.float32)
            qpos_np[:, hinge_idx] = -60.0 * np.pi / 180.0 
            qpos_tensor = torch.from_numpy(qpos_np).to(self.device)
            self.laptop_articulation.set_qpos(qpos_tensor)
            
            #base pose and random offset for the laptop are determined according to experiment, do not change
            base_pos = np.array([0, 0.35, 0.1])
            random_offset_x = np.random.uniform(-0.05, 0.02, size=(b,))
            random_offset_y = np.random.uniform(-0.1, 0.15, size=(b,))  
            new_pos = np.stack([base_pos[0] + random_offset_x,
                                base_pos[1] + random_offset_y,
                                base_pos[2]*np.ones(b)],axis=1)
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
             
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
        
    
        
        


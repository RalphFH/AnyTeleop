from typing import Any, Dict, Union
import os
import numpy as np
import sapien
import torch
import torch.random
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from transforms3d.euler import euler2quat
from mani_skill.utils.registration import register_env

from scipy.spatial.transform import Rotation as R
from mani_skill.agents.robots import  Panda
from mani_skill.agents.robots import XArm7Allegro, XArm7Shadow, XArm7Leap
from mani_skill.agents.robots import UR5eShadow, UR5eAllegro, UR5eLeap
from mani_skill.agents.robots import XArm6Allegro, XArm6Shadow
from mani_skill.agents.robots import IIwa7Allegro

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose as BatchPose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig, SceneConfig
from mani_skill.utils.building import articulations
from mani_skill.utils.quater import product
from mani_skill.utils.structs import SimConfig, GPUMemoryConfig


current_dir = os.path.dirname(__file__)


@register_env("BookWall-v1", max_episode_steps=300)
class BookWallEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda",
                        "xarm7_allegro_right", "xarm7_shadow_right", "xarm7_leap_right",
                        "xarm6_allegro_right","xarm6_shadow_right",
                        "ur5e_shadow_right", "ur5e_allegro_right", "ur5e_leap_right",
                        "iiwa7_allegro_right"]             
    
    # Specify some supported robot types
    agent: Union[Panda,
                 XArm7Allegro, XArm7Shadow, XArm7Leap,
                 XArm6Allegro, XArm6Shadow,
                 UR5eShadow, UR5eAllegro, UR5eLeap,
                 IIwa7Allegro]        
    
    
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,num_envs= 10,reward_mode="sparse" ,sim_backend = "physx_cuda",  **kwargs):   
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.goal_thresh=0.025 / 3  
        self.lift_height = 0.15
        super().__init__(*args, robot_uids=robot_uids,reward_mode=reward_mode,num_envs=num_envs,sim_backend=sim_backend, **kwargs)    #

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
        
        top_down = sapien_utils.look_at([-0.18, 0.0, 0.7], [-0.06, 0.0, 0])
        left_side = sapien_utils.look_at([0.4, 0.5, 0.5], [0.0, 0.0, 0.35]) 

        cam_config = []
        cam_config.append(CameraConfig("top_down", top_down, 512, 512, 70*np.pi/180, 0.01, 100))



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
                         
        if "allegro" in self.robot_uids:
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
            q4 = [np.cos(35*np.pi/180), np.sin(35*np.pi/180), 0, 0]
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
        
    
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        builder = self.scene.create_actor_builder()
        size = [0.02, 0.8, 0.04]
        builder.add_box_collision(half_size=size)
        builder.add_box_visual(
            half_size=size,
            material=sapien.render.RenderMaterial(
                # RGBA values, this is a red cube
                base_color=np.array([89, 65, 42, 128]) / 255,
            ),)
        builder.initial_pose = sapien.Pose(p=[0.15, 0.0, 0.02], q=[1, 0, 0, 0])            #############  wall long and not random  ###
        self.wall = builder.build_static(
            name="wall",
            )
        self.lift_height = 0.15
        
        builder = self.scene.create_actor_builder()
        size = [0.1, 0.2, 0.015]
        builder.add_box_collision(half_size=size)
        builder.add_box_visual(
            half_size=size,
            material=sapien.render.RenderMaterial(
                # RGBA values, this is a red cube
                base_color=np.array([173, 216, 230, 255]) / 255,
            ),)
        
        base_pos = np.array([-0.05, 0.0, 0.1])
        random_offset_x = np.random.uniform(-0.1, 0.05, size=(self.num_envs,))
        random_offset_y = np.random.uniform(-0.2, 0.2, size=(self.num_envs,))
        new_pos = np.stack([base_pos[0] + random_offset_x,
                            base_pos[1] + random_offset_y,
                            base_pos[2]*np.ones(self.num_envs)],axis=1)
        p_tensor = torch.tensor(new_pos, dtype=torch.float32, device=self.device)
        batched_pose = BatchPose.create_from_pq(p=p_tensor, q=None, device=self.device)
        builder.initial_pose = batched_pose
        
               #############book random ####
        self.book = builder.build(
            name="book",
            )
      
      
      
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            super()._initialize_episode(env_idx, options)
            
            # set book on the table
            base_pos = np.array([-0.05, 0, 0])
            random_offset_x = np.random.uniform(-0.1, 0.05, size=(b,))
            random_offset_y = np.random.uniform(-0.2, 0.2, size=(b,))
            new_pos = np.stack([base_pos[0] + random_offset_x,
                                base_pos[1] + random_offset_y,
                                base_pos[2]*np.ones(b)],axis=1)
            p_tensor = torch.tensor(new_pos, dtype=torch.float32, device=self.device)
            batched_pose1 = BatchPose.create_from_pq(p=p_tensor, q=None, device=self.device)
            self.book.set_pose(batched_pose1)

                
            if self.robot_uids == "panda":        
                #qpos is the position of every joint of the robot,also determined according to experiments, do not change
                base_qpos = np.array([0.09, -0.85, -0.04, -2, -0.07, 1.2, -0.7, 0 ,0])
                noise_np = np.random.uniform(
                low=-self.robot_init_qpos_noise,
                high=self.robot_init_qpos_noise,
                size=(b, len(base_qpos))
               )
                base_qpos_tiled = np.tile(base_qpos, (b, 1))  
                new_qpos_np = base_qpos_tiled + noise_np 
                new_qpos_tensor = torch.from_numpy(new_qpos_np).float().to(self.device)
                self.agent.robot.set_qpos(new_qpos_tensor)
                
            if self.robot_uids == "ur5e_allegro_right":        
                #qpos is the position of every joint of the robot,also determined according to experiments, do not change
                base_qpos = np.array([   -0.24955  ,   -1.7031   ,   1.7518   , -0.50697   ,  0.85624  ,   0.29718  ,   0.15545  ,   0.12096 ,
                                         0.18827   ,  0.32904   ,  0.66978   ,   0.5374  ,   0.74709  , -0.052536, -3.6468e-05   , 0.063846  
                                         -0.065989   ,   0.9719 ,  -0.014903 , -0.0053921 ,  -0.033205  ,   0.28124,0])
                noise_np = np.random.uniform(
                low=-self.robot_init_qpos_noise,
                high=self.robot_init_qpos_noise,
                size=(b, len(base_qpos))
               )
                base_qpos_tiled = np.tile(base_qpos, (b, 1))  
                new_qpos_np = base_qpos_tiled + noise_np 
                new_qpos_tensor = torch.from_numpy(new_qpos_np).float().to(self.device)
                self.agent.robot.set_qpos(new_qpos_tensor)
            
            if self.robot_uids == "xarm6_shadow_right":        
                #qpos is the position of every joint of the robot,also determined according to experiments, do not change
                base_qpos = np.array([   -0.79639 ,   -0.56757  ,  -0.82376   ,   1.3494     ,  1.562    ,  1.2145  ,  0.081409   , -0.20006 ,
                                         -0.24934 ,    -0.1069 ,  -0.012685   ,  0.38067   ,  0.47479 ,  -0.057781 ,   0.050859  ,   0.13706 ,  
                                         -0.31211, -0.00099533 ,    0.20491 ,   0.071981 ,  0.0030264  ,  -0.15471 , -0.080778 ,   0.051354   ,
                                         0.007908 ,  0.0014614 ,  0.0043305  ,   0.10043  ,  0.002212  ,  -0.04348])
                noise_np = np.random.uniform(
                low=-self.robot_init_qpos_noise,
                high=self.robot_init_qpos_noise,
                size=(b, len(base_qpos))
               )
                base_qpos_tiled = np.tile(base_qpos, (b, 1))  
                new_qpos_np = base_qpos_tiled + noise_np 
                new_qpos_tensor = torch.from_numpy(new_qpos_np).float().to(self.device)
                self.agent.robot.set_qpos(new_qpos_tensor)
            
            
            if self.robot_uids == "xarm7_leap_right":        
                #qpos is the position of every joint of the robot,also determined according to experiments, do not change
                base_qpos = np.array([    0.17007  ,  -0.21156  ,  -0.22471   ,   1.0581   ,    1.756    ,  1.2238    , 
                                         3.9243 ,  -0.023651  ,   0.12155 ,    0.31412  ,  0.057673   , -0.20355  ,  -0.14136 ,
                                         -0.34307  , -0.063874    , 0.23151 ,   0.019688  ,  -0.19046  ,   0.77333  , -0.010055  ,
                                         0.014319 ,   -0.19212  ,   0.28517])
                noise_np = np.random.uniform(
                low=-self.robot_init_qpos_noise,
                high=self.robot_init_qpos_noise,
                size=(b, len(base_qpos))
               )
                base_qpos_tiled = np.tile(base_qpos, (b, 1))  
                new_qpos_np = base_qpos_tiled + noise_np 
                new_qpos_tensor = torch.from_numpy(new_qpos_np).float().to(self.device)
                self.agent.robot.set_qpos(new_qpos_tensor)

        
    def evaluate(self):
        is_obj_lifted = (self.lift_height - self.book.pose.p[:, 2]) <= self.goal_thresh
     
        return {"success": is_obj_lifted}


    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
                     
                
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
        
    
        
        

    
    
 
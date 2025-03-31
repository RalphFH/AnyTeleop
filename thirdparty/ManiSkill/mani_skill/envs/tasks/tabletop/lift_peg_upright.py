from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.agents.robots import XArm7Allegro, XArm7Shadow, XArm7Leap
from mani_skill.agents.robots import UR5eShadow, UR5eAllegro, UR5eLeap
from mani_skill.agents.robots import XArm6Shadow, XArm6Allegro
from mani_skill.agents.robots import IIwa7Allegro


from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

from mani_skill.utils.quater import product


@register_env("LiftPegUpright-v1", max_episode_steps=250)
class LiftPegUprightEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to move a peg laying on the table to any upright position on the table

    **Randomizations:**
    - the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table

    **Success Conditions:**
    - the absolute value of the peg's y euler angle is within 0.08 of $\pi$/2 and the z position of the peg is within 0.005 of its half-length (0.12).
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/LiftPegUpright-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch",
                        "xarm7_allegro_right", "xarm7_shadow_right", "xarm7_leap_right",
                        "xarm6_allegro_right", "xarm6_shadow_right",
                        "ur5e_shadow_right", "ur5e_allegro_right", "ur5e_leap_right",
                        "iiwa7_allegro_right"]
    agent: Union[Panda, Fetch,
                 XArm7Allegro, XArm7Shadow, XArm7Leap,
                 XArm6Allegro, XArm6Shadow,
                 UR5eShadow, UR5eAllegro, UR5eLeap,
                 IIwa7Allegro]


    peg_half_width = 0.025
    peg_half_length = 0.12

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.has_been_successful = None

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.65], target=[-0.15, 0, 0.3])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self): 
        top_down = look_at([-0.12, 0.0, 0.36], [0.15, 0.0, 0])
        right_side = look_at([-0.2, -0.4, 0.18], [-0.2, 0.3, 0.18]) 

        cam_config = []
        cam_config.append(CameraConfig("top_down", top_down, 512, 512, 80*np.pi/180, 0.01, 100))



        if "xarm7" in self.robot_uids and (not "shadow" in self.robot_uids):
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
                                fov=1.57,
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
                                pose=sapien.Pose(p=[-0.02, 0.2 , -0.02], q=q1),
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
                                pose=sapien.Pose(p=[-0.01, 0.25 , -0.04], q=q),
                                width=512,
                                height=512,
                                fov=1.57,
                                near=0.01,
                                far=100,
                                entity_uid="palm_lower",
                            ))
            



        cam_config.append( CameraConfig("scene_right_camera", right_side, 512, 512, 80*np.pi/180, 0.01, 100))

        return cam_config

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # the peg that we want to manipulate
        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([176, 14, 14, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.05
            xyz[..., 2] = self.peg_half_width
            q = euler2quat(np.pi / 2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)

    def evaluate(self):
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
        is_peg_upright = (
            torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.08
        )  # 0.08 radians of difference permitted
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
        is_loosen = not self.agent.is_grasping(self.peg)
        return {
            "success": is_peg_upright & close_to_table & is_loosen,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.peg.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # rotation reward as cosine similarity between peg direction vectors
        # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
        # dot product with its goal orientation: (0,0,1) or (0,0,-1)
        qmats = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        vec = torch.tensor([1.0, 0, 0], device=self.device)
        goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
        rot_vec = (qmats @ vec).view(-1, 3)
        # abs since (0,0,-1) is also valid, values in [0,1]
        rot_rew = (rot_vec @ goal_vec).view(-1).abs()
        reward = rot_rew

        # position reward using common maniskill distance reward pattern
        # giving reward in [0,1] for moving center of mass toward half length above table
        z_dist = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length)
        reward += 1 - torch.tanh(5 * z_dist)

        # small reward to motivate initial reaching
        # initially, we want to reach and grip peg
        to_grip_vec = self.peg.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
        # reaching reward granted if gripping block
        reaching_rew[self.agent.is_grasping(self.peg)] = 1

        # if self.agent.is_grasping(self.peg):
        #     # weight reaching reward more
        #     print("grasping")
        # else:
        #     print("loosen")
        
        # weight reaching reward less
        reaching_rew = reaching_rew / 5
        reward += reaching_rew

        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        success = info["success"]

        if self.has_been_successful is None:
            self.has_been_successful = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        newly_success = success & (~self.has_been_successful)
        reward = newly_success.float()
        self.has_been_successful = self.has_been_successful | success

        return reward
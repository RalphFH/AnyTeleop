from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.robots.xmate3.xmate3 import Xmate3Robotiq
from mani_skill.agents.robots import XArm7Allegro, XArm7Shadow, XArm7Leap, XArm6Allegro
from mani_skill.agents.robots import UR5eShadow, UR5eAllegro, UR5eLeap
from mani_skill.agents.robots import IIwa7Allegro

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from mani_skill.utils.quater import product

WARNED_ONCE = False


@register_env("PickSingleYCB-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PickSingleYCBEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a random object sampled from the [YCB dataset](https://www.ycbbenchmarks.com/) and move it to a random goal position

    **Randomizations:**
    - the object's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the object's z-axis rotation is randomized
    - the object geometry is randomized by randomly sampling any YCB object. (during reconfiguration)

    **Success Conditions:**
    - the object position is within goal_thresh (default 0.025) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)

    **Goal Specification:**
    - 3D goal position (also visualized in human renders)

    **Additional Notes**
    - On GPU simulation, in order to collect data from every possible object in the YCB database we recommend using at least 128 parallel environments or more, otherwise you will need to reconfigure in order to sample new objects.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickSingleYCB-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch",
                        "xarm7_allegro_right", "xarm7_shadow_right", "xarm7_leap_right",
                        "xarm6_allegro_right",
                        "ur5e_shadow_right", "ur5e_allegro_right", "ur5e_leap_right",
                        "iiwa7_allegro_right"]
    agent: Union[Panda, PandaWristCam, Fetch,
                 XArm7Allegro, XArm7Shadow, XArm7Leap,
                 XArm6Allegro,
                 UR5eShadow, UR5eAllegro, UR5eLeap,
                 IIwa7Allegro]
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        self.all_model_ids = np.array(
            list(
                load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
            )
        )
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):

        top_down = sapien_utils.look_at([-0.15, 0.0, 0.4], [-0.05, 0.0, 0])
        left_side = sapien_utils.look_at([-0.03, 0.27, 0.15], [-0.03, 0.1, 0.15]) 

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

        if "panda" in self.robot_uids:
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
            cam_config.append(CameraConfig(
                                uid="hand_cam",
                                pose=sapien.Pose(p=[0, 0.23 , 0.18], q=[0.7044, 0.06166, 0.06166, -0.7044]),
                                width=512,
                                height=512,
                                fov=1.57,
                                near=0.01,
                                far=100,
                                entity_uid="palm",
                            ))
        elif "leap" in self.robot_uids:
            cam_config.append(CameraConfig(
                                uid="hand_cam",
                                pose=sapien.Pose(p=[0.03, 0.0 , 0.01], q=[1, 0, 0, 0]),
                                width=512,
                                height=512,
                                fov=1.57,
                                near=0.01,
                                far=100,
                                entity_uid="base_hand",
                            ))
            



        cam_config.append( CameraConfig("scene_left_camera", left_side, 512, 512, 80*np.pi/180, 0.01, 100))

        return cam_config

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # randomize the list of all possible models in the YCB dataset
        # then sub-scene i will load model model_ids[i % number_of_ycb_objects]
        model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        if (
            self.num_envs > 1
            and self.num_envs < len(self.all_model_ids)
            and self.reconfiguration_freq <= 0
            and not WARNED_ONCE
        ):
            WARNED_ONCE = True
            print(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be >= 1."""
            )

        self._objs: List[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.obj)

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # this value is used to set object pose so the bottom is at z=0
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Initialize robot arm to a higher position above the table than the default typically used for other table top tasks
            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            else:
                qpos = self.agent.keyframes["rest"].qpos
                qpos = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
                if "panda" in self.robot_uids:              
                    qpos[...,-2:] = 0.04
                self.agent.reset(qpos)
                self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def evaluate(self):
        obj_to_goal_pos = self.goal_site.pose.p - self.obj.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.2)
        return dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=self.agent.is_grasping(self.obj),
            success=torch.logical_and(is_obj_placed, is_robot_static),
        )

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        reward += info["is_obj_placed"] * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * is_grasped

        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6

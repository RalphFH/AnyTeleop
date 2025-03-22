from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils

from mani_skill.utils.structs.actor import Actor
import torch



@register_agent()
class UR5eAllegro(BaseAgent):
    uid = "ur5e_allegro_right"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/ur5e/ur5e_allegro_right.urdf"
    urdf_config = dict(
        _materials=dict(
            front_finger=dict(
                static_friction=4.0, dynamic_friction=4.0, restitution=0.0
            )
        ),
        link={
                **{
                    f"link_{i}": {
                        "material": "front_finger",
                        "patch_radius": 0.08,
                        "min_patch_radius": 0.05,
                    }
                    for i in range(16)  # 生成 link_0 到 link_15
                },
                **{
                    f"link_{i}_tip": {
                        "material": "front_finger",
                        "patch_radius": 0.08,
                        "min_patch_radius": 0.05,
                    }
                    for i in [3, 7, 11, 15]  # 额外添加 link_3_tip, link_7_tip, link_15_tip
                }
            },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -85*np.pi/180,
                    146*np.pi/180,
                    -97*np.pi/180,
                    90*np.pi/180,
                    0.0,
                    0.0,
                    0.0, 
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0, 
                    0.0,
                    0.0,
                    0.0,
                    0.0,                 
                ]                
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 50

        self.hand_joint_names = [
          'joint_0', 
          'joint_4', 
          'joint_8', 
          'joint_12',
          'joint_1', 
          'joint_5', 
          'joint_9', 
          'joint_13', 
          'joint_2', 
          'joint_6', 
          'joint_10', 
          'joint_14', 
          'joint_3', 
          'joint_7', 
          'joint_11', 
          'joint_15' 
        ]
        self.hand_stiffness = 2e3
        self.hand_damping = 1e2
        self.hand_friction = 3.0
        self.hand_force_limit = 20

        self.ee_link_name = "palm"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True


        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
        hand_target_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.1,
            0.1,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            use_delta=True,
        )
        hand_target_delta_pos.use_target = True

        hand_target_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=None,
            upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            friction=self.hand_friction,
            normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_pos, gripper=hand_target_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_target_pos)
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "link_15"
        )

        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "link_3"
        )

        self.finger1_link_tip = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "link_15_tip"
        )

        self.finger2_link_tip = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "link_3_tip"
        )

        self.palm = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "palm"
        )

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()

    def is_grasping(self, object: Actor, min_force=0.5, min_tip_distance=0.15, max_contact_force=5.0):
        """Check if the robot is grasping an object based on tip distance and contact forces.

        Args:
            object (Actor): The object to check if the robot is grasping.
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object. Defaults to 0.5 N.
            max_tip_distance (float, optional): Maximum distance between finger tips for a valid grasp. Defaults to 0.05 meters.
            max_contact_force (float, optional): Maximum total force between the links and the palm to consider a valid grasp. Defaults to 5.0 N.
        """
        
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        
        # distance between finger tips
        l_finger_pos = self.finger1_link_tip.pose.p
        r_finger_pos = self.finger2_link_tip.pose.p
        tip_distance = torch.linalg.norm(l_finger_pos - r_finger_pos)

        # total contact forces between the links and the object
        link_forces = 0
        for link in [self.finger1_link, self.finger2_link, self.finger1_link_tip, self.finger2_link_tip]:
            link_contact_forces = self.scene.get_pairwise_contact_forces(link, object)
            link_forces += torch.sum(torch.linalg.norm(link_contact_forces, axis=1))


        return torch.logical_and((tip_distance <= min_tip_distance), (link_forces >= max_contact_force))

    
    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :7]
        return torch.max(torch.abs(qvel), 1)[0] <= (threshold)
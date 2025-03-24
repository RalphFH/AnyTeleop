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

from mani_skill.utils.structs.actor import Actor
import torch


@register_agent()
class XArm6Shadow(BaseAgent):
    uid = "xarm6_shadow_right"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm6/xarm6_shadow_right.urdf"
    _links = [
        'thtip', 'fftip', 'mftip', 'rftip', 'lftip',
        'thbase', 'thhub',
        'thproximal', 'ffproximal', 'mfproximal', 'rfproximal', 'lfproximal',
        'thmiddle', 'ffmiddle', 'mfmiddle', 'rfmiddle', 'lfmiddle',
        'thdistal', 'ffdistal', 'mfdistal', 'rfdistal', 
        'fftip', 'mftip', 'rftip', 'lfdistal','lftip'        
    ]    
    urdf_config = dict(
        _materials=dict(
            front_finger=dict(
                static_friction=4.0, dynamic_friction=4.0, restitution=0.0
            ),
            palm=dict(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0
            ),
        ),
        link={**{
            link_name: dict(
                material="front_finger", patch_radius=0.03, min_patch_radius=0.02
            ) for link_name in _links},
            "palm": dict(  # 单独为手掌配置
                material="palm", 
                patch_radius=0.10, 
                min_patch_radius=0.08
            )
        }
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -10*np.pi/180,
                    0.0,
                    np.pi,
                    90*np.pi/180,
                    0.0,

                    0.0,
                    -1.3, 
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
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 50

        self.hand_joint_names = [
            'WRJ2', 
            'WRJ1', 
            'FFJ4', 
            'MFJ4', 
            'RFJ4', 
            'LFJ5', 
            'THJ5', 
            'FFJ3', 
            'MFJ3', 
            'RFJ3', 
            'LFJ4', 
            'THJ4', 
            'FFJ2', 
            'MFJ2', 
            'RFJ2', 
            'LFJ3', 
            'THJ3', 
            'FFJ1', 
            'MFJ1', 
            'RFJ1', 
            'LFJ2', 
            'THJ2', 
            'LFJ1', 
            'THJ1'
        ]
        self.hand_stiffness = 2e3
        self.hand_damping = 1e2
        self.hand_friction = 3
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
            self.robot.get_links(), "thdistal"
        )

        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "ffdistal"
        )

        self.finger1_link_tip = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "thtip"
        )

        self.finger2_link_tip = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "fftip"
        )

        self.palm = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "palm"
        )

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :6]
        return torch.max(torch.abs(qvel), 1)[0] <= (threshold)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=20, min_tip_distance=0.15, max_contact_force=5.0):
        """Check if the robot is grasping an object based on tip distance and contact forces.

        Args:
            object (Actor): The object to check if the robot is grasping.
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object. Defaults to 0.5 N.
            min_tip_distance (float, optional): Manimum distance between finger tips for a valid grasp. Defaults to 0.15 meters.
            max_contact_force (float, optional): Maximum total force between the links and the palm to consider a valid grasp. Defaults to 5.0 N.
        """
        
        # distance between finger tips
        l_finger_pos = self.finger1_link_tip.pose.p
        r_finger_pos = self.finger2_link_tip.pose.p
        tip_distance = torch.linalg.norm(l_finger_pos - r_finger_pos)

        # total contact forces between the links and the object
        link_forces = 0
        for link in [self.finger1_link, self.finger2_link, self.finger1_link_tip, self.finger2_link_tip]:
            link_contact_forces = self.scene.get_pairwise_contact_forces(link, object)
            link_forces += torch.sum(torch.linalg.norm(link_contact_forces, axis=1))


        return torch.logical_and((tip_distance <= min_tip_distance), (link_forces >= max_contact_force)).unsqueeze(0)

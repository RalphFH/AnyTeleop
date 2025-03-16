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


@register_agent()
class XArm7Shadow(BaseAgent):
    uid = "xarm7_shadow_right"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm7/xarm7_shadow_right.urdf"
    urdf_config = dict(
        _materials=dict(
            front_finger=dict(
                static_friction=5.0, dynamic_friction=5.0, restitution=0.0
            )
        ),
        link=dict(
            thtip=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            fftip=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            mftip=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            rftip=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            lftip=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            )
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -0.6,
                    0.0,
                    0.1,
                    3.14,
                    0.5,
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
            "joint7",
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
        self.hand_stiffness = 1e3
        self.hand_damping = 1e2
        self.hand_friction = 1
        self.hand_force_limit = 50

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
        # hand_front_link_names = [
        #     "thumb_L2",
        #     "index_L2",
        #     "middle_L2",
        #     "ring_L2",
        #     "pinky_L2",
        # ]
        # self.hand_front_links = sapien_utils.get_objs_by_names(
        #     self.robot.get_links(), hand_front_link_names
        # )

        # finger_tip_link_names = [
        #     "thumb_tip",
        #     "index_tip",
        #     "middle_tip",
        #     "ring_tip",
        #     "pinky_tip",
        # ]
        # self.finger_tip_links = sapien_utils.get_objs_by_names(
        #     self.robot.get_links(), finger_tip_link_names
        # )

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()


    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """

        return False

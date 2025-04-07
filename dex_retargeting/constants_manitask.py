import enum
from pathlib import Path
from typing import Optional

import numpy as np



class ArmName(enum.Enum):
    xarm7 = enum.auto()
    xarm6 = enum.auto()
    ur5e = enum.auto()
    iiwa7 = enum.auto()

class HandName(enum.Enum):
    allegro = enum.auto() # 4 fingers
    shadow = enum.auto() # 5 fingers
    leap = enum.auto() # 4 fingers


class RetargetingType(enum.Enum):
    vector = enum.auto()  # For teleoperation, no finger closing prior
    position = enum.auto()  # For offline data processing, especially hand-object interaction data
    dexpilot = enum.auto()  # For teleoperation, with finger closing prior


class HandType(enum.Enum):
    right = enum.auto()
    left = enum.auto()


ARM_NAME_MAP = {
    ArmName.xarm7: "xarm7",
    ArmName.xarm6: "xarm6",
    ArmName.ur5e: "ur5e",
    ArmName.iiwa7: "iiwa7",
}

HAND_NAME_MAP = {
    HandName.allegro: "allegro",
    HandName.shadow: "shadow",
    HandName.leap: "leap",
}

LINK_BASE = {
    "xarm7": "link_base",
    "xarm6": "world",
    "ur5e": "base_link",
    "iiwa7": "link_0_arm",
}

LINK_WRIST = {
    "allegro":"palm",
    "shadow":"palm",
    "leap":"base_hand",
}

# ROBOT_NAMES = list(HAND_NAME_MAP.keys())


def get_default_config_path(
    arm:ArmName, hand: HandName, hand_type: HandType
) -> Optional[Path]:
    config_path = Path(__file__).parent / "configs"
    config_path = config_path / "manitask"

    arm_name_str = ARM_NAME_MAP[arm]
    hand_name_str = HAND_NAME_MAP[hand]
    hand_type_str = hand_type.name
    
    config_name = f"{hand_name_str}_{hand_type_str}.yml"
    robot_uid = f"{arm_name_str}_{hand_name_str}_{hand_type_str}"

    return config_path / config_name, robot_uid



import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import URDFLoader
loader = URDFLoader()
loader.set_scene(ManiSkillScene())
robot = loader.load("../../thirdparty/ManiSkill/mani_skill/assets/robots/iiwa7/iiwa7_allegro_right.urdf")
print(robot.active_joints_map.keys())
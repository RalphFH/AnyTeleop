from typing import Any, Dict, Union
import open3d as o3d
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat
import gymnasium as gym
import viser 
from pathlib import Path

from mani_skill.utils.structs import Link
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.utils import sapien_utils


@register_env("LiftPegUprightEnvForPC-v1", max_episode_steps=250)
class LiftPegUprightEnvForPC(LiftPegUprightEnv):
    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # Common parameters
        target = [-0.15, 0, 0.3]  # Target point remains the same
        radius = 1.2  # Distance from target to camera (calculated from original eye position)
        height = 0.35  # Height of cameras (z coordinate)
        
        # Calculate camera positions at 72-degree intervals (360/5 = 72)
        cameras = []
        for i in range(8):
            angle = i * 2 * np.pi / 8  # Divide circle into 5 equal parts
            eye_x = target[0] + radius * np.cos(angle)
            eye_y = target[1] + radius * np.sin(angle)
            eye = [eye_x, eye_y, height]
            pose = look_at(eye=eye, target=target)
            cameras.append(CameraConfig(f"group1_base_camera_{i}", pose, 512, 512, np.pi / 2, 0.01, 100))

        radius = 0.5
        height = 0.7
        for i in range(5):
            angle = i * 2 * np.pi / 5  # Divide circle into 5 equal parts
            eye_x = target[0] + radius * np.cos(angle)
            eye_y = target[1] + radius * np.sin(angle)
            eye = [eye_x, eye_y, height]
            pose = look_at(eye=eye, target=target)
            cameras.append(CameraConfig(f"group2_base_camera_{i}", pose, 512, 512, np.pi / 2, 0.01, 100))

        radius = 0.3
        height = 1.5
        for i in range(3):
            angle = i * 2 * np.pi / 3  # Divide circle into 5 equal parts
            eye_x = target[0] + radius * np.cos(angle)
            eye_y = target[1] + radius * np.sin(angle)
            eye = [eye_x, eye_y, height]
            pose = look_at(eye=eye, target=target)
            cameras.append(CameraConfig(f"group3_base_camera_{i}", pose, 512, 512, np.pi / 2, 0.01, 100))


        radius = 0.8
        height = 0.1
        for i in range(5):
            angle = i * 2 * np.pi / 5  # Divide circle into 5 equal parts
            eye_x = target[0] + radius * np.cos(angle)
            eye_y = target[1] + radius * np.sin(angle)
            eye = [eye_x, eye_y, height]
            pose = look_at(eye=eye, target=target)
            cameras.append(CameraConfig(f"group4_base_camera_{i}", pose, 512, 512, np.pi / 2, 0.01, 100))


        radius = 0.05
        height = 2.0
        for i in range(3):
            angle = i * 2 * np.pi / 3  # Divide circle into 5 equal parts
            eye_x = target[0] + radius * np.cos(angle)
            eye_y = target[1] + radius * np.sin(angle)
            eye = [eye_x, eye_y, height]
            pose = look_at(eye=eye, target=target)
            cameras.append(CameraConfig(f"group5_base_camera_{i}", pose, 512, 512, np.pi / 2, 0.01, 100))

        return cameras

    @property
    def _default_human_render_camera_configs(self): 
        right_side = look_at([-0.2, -0.4, 0.18], [-0.2, 0.3, 0.18]) 
        cam_config = []
        cam_config.append( CameraConfig("scene_right_camera", right_side, 512, 512, 80*np.pi/180, 0.01, 100))
        return cam_config

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
            initial_pose=sapien.Pose(p=[100, 0, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.05
            xyz[..., 2] = self.peg_half_width
            xyz[..., 0] = 100
            q = euler2quat(np.pi / 2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)

    @property 
    def robot_link_names(self):
        return sorted([link.name for link in self.agent.robot.get_links()])
    
    @property
    def robot_link_name_to_link(self):
        return {link_name:sapien_utils.get_obj_by_name(
                self.agent.robot.get_links(), link_name
            ) for link_name in self.unwrapped.robot_link_names}
    
    @property 
    def robot_link_name_to_seg_id(self):
        robot_link_name_to_seg_id = {}
        for obj_id, obj in sorted(self.unwrapped.segmentation_id_map.items()):
            if isinstance(obj, Link):
                robot_link_name_to_seg_id[obj.name] = obj_id
        return robot_link_name_to_seg_id

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        for link_name, link in self.robot_link_name_to_link.items():
            obs[link_name] = link.pose.raw_pose
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.peg.pose.raw_pose,
            )
        return obs
    

if __name__ == "__main__":
    robot_uid = "ur5e_allegro_right"
    if robot_uid != "panda" and robot_uid != "panda_wristcam":
        arm, hand, _ = robot_uid.split("_")
        save_dir = f"data/assets/robots/maniskill/{arm}/{robot_uid}/body_pc"
    else:
        save_dir = "data/assets/robots/maniskill/panda/body_pc"
    n_config = 10
    env = gym.make(
        "LiftPegUprightEnvForPC-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        robot_uids=robot_uid,
        obs_mode="pointcloud", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        render_mode="rgb_array", # rgb_array | human | all
    )
    server = viser.ViserServer()

    env.reset()

    body_name_to_pc_canonical = {}
    
    for i in range(n_config):
        print(i)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        current_pc = obs["pointcloud"]["xyzw"][0]  # (N, 4)
        current_pc_seg = obs["pointcloud"]["segmentation"][0]  # (N, 1)
        current_pc_colors = obs["pointcloud"]["rgb"][0]  # (N, 3)
        current_valid_pc_mask = current_pc[:, 3] == 1 
        current_pc = current_pc[current_valid_pc_mask] # (n, 4)
        current_pc = current_pc[:, :3].cpu().numpy()  # (n, 3)
        current_valid_pc_seg = current_pc_seg[current_valid_pc_mask].cpu().numpy()  # (n, 1)
        current_valid_pc_colors = current_pc_colors[current_valid_pc_mask].cpu().numpy()  # (n, 3)

        # Process each robot link
        for link_name in env.unwrapped.robot_link_names:
            # Get link pose from observations
            link_pose = obs['extra'][link_name].squeeze()  # [x, y, z, qw, qx, qy, qz]
            
            # Get points belonging to this link using segmentation
            link_mask = current_valid_pc_seg.squeeze() == env.unwrapped.robot_link_name_to_seg_id[link_name]
            link_points = current_pc[link_mask]
            link_colors = current_valid_pc_colors[link_mask]
            
            if len(link_points) == 0:
                continue
                
            # Transform points to link's canonical frame
            link_pose_mat = sapien.Pose(p=link_pose[:3], q=link_pose[3:]).to_transformation_matrix()
            canonical_points = (np.linalg.inv(link_pose_mat) @ np.concatenate(
                [link_points, np.ones((len(link_points), 1))], axis=1).T).T[:, :3]
            
            # Create and downsample point cloud
            link_pcd = o3d.geometry.PointCloud()
            link_pcd.points = o3d.utility.Vector3dVector(canonical_points)
            link_pcd.colors = o3d.utility.Vector3dVector(link_colors / 255.0)
            link_pcd = link_pcd.voxel_down_sample(voxel_size=0.005)
            
            if link_name not in body_name_to_pc_canonical:
                body_name_to_pc_canonical[link_name] = link_pcd
            else:
                # Merge with existing point cloud
                body_name_to_pc_canonical[link_name].points.extend(link_pcd.points)
                body_name_to_pc_canonical[link_name].colors.extend(link_pcd.colors)
                # Remove duplicates
                body_name_to_pc_canonical[link_name] = body_name_to_pc_canonical[link_name].voxel_down_sample(0.005)

    ############################################################
    # Save the point clouds
    ############################################################
    save_dir_path = Path(save_dir).resolve()
    save_dir_path.mkdir(parents=True, exist_ok=True)
    for body_name in body_name_to_pc_canonical.keys():
        o3d.io.write_point_cloud(
            str(save_dir_path / f"{body_name}.ply"),
            body_name_to_pc_canonical[body_name],
        )

    server.scene.add_point_cloud(
        "current_pc",
        current_pc,
        current_valid_pc_colors,
        point_size=0.003,
        point_shape="circle",
    )
    for link_name, pcd in body_name_to_pc_canonical.items():
        server.scene.add_point_cloud(
            link_name,
            np.asarray(pcd.points),
            (np.asarray(pcd.colors) * 255).astype(np.uint8),
            point_size=0.003,
            point_shape="circle",
        )
    input("Press Enter to exit...")

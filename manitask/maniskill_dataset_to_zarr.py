import h5py
import json
import time
import tyro
import viser
import sapien
import subprocess
import numpy as np
import gymnasium as gym
from pathlib import Path
from dataclasses import dataclass

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Link
from pov.utils.camera.pc import fpsample_pc
from pov.datasets.pov_zarr_dataset import PovZarrDataset


@dataclass
class Args:
    hdf5_dir: str
    hdf5_name: str
    robot_pc_dir_path: str
    episode_num: int = 1
    n_sample_pc: int = 512
    zarr_save_path: str = "./converted.zarr"
    replay: bool = True
    use_env_state: str = "use_env_states"
    save_replay_video: bool = True
    tozarr: bool = True
    visualize: bool = True

def parse_args():
    return tyro.cli(Args)

def visualize_all_frames_with_viser(server,pointclouds: np.ndarray, colors: np.ndarray, current_robot_points: np.ndarray, target_robot_points: np.ndarray, interval: float = 0.01):
    """
    Visualize all point cloud frames using viser in sequence (like animation).

    Args:
        pointclouds: (n_steps, n_points, 3)
        colors: (n_steps, n_points, 3)
        current_robot_points: (n_steps, n_points, 3)
        target_robot_points: (n_steps, n_points, 3)
        interval: time delay between frames in seconds
    """

    # normalize color to [0, 1]
    colors = colors.astype(np.float32) / 255.0

    print("Viser running at http://localhost:8080 — streaming point cloud...")

    for i in range(pointclouds.shape[0]):
        pc = pointclouds[i]
        col = colors[i]
        cr_pc = current_robot_points[i]
        tr_pc = target_robot_points[i]

        server.scene.add_point_cloud(
            name="pointcloud",
            points=pc,
            colors=col,
            point_size=0.01,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            name="current_robot_points",
            points=cr_pc,
            colors=(255, 0, 0),
            point_size=0.005,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            name="target_robot_points",
            points=tr_pc,
            colors=(0, 0, 255),
            point_size=0.005,
            point_shape="circle",
        )
        time.sleep(interval)

    print("Finished playing all frames.")


def main(args: Args):
    process_num = args.episode_num
    bounding_box = [[-0.4, -5, 0.01], [5, 5, 5]]
    # urdf and hdf5 data
    robot_pc_dir_path = args.robot_pc_dir_path
    hdf5_dir = args.hdf5_dir
    state_file = hdf5_dir + args.hdf5_name

    if args.replay:
        if args.save_replay_video:
            print("Converting original trajectory to rgbd+seg...")
            subprocess.run([
                "python", "-m", "mani_skill.trajectory.replay_trajectory",
                "--traj-path", state_file+".h5",
                "--"+args.use_env_state,
                "--save-traj", "--target-control-mode", "pd_joint_pos",
                "--obs-mode", "rgb+depth+segmentation", "--num-envs", "1", "--count", str(process_num)
            ], check=True)


            # prepare pointcloud
            print("Converting original trajectory to pointcloud...")
            subprocess.run([
                "python", "-m", "mani_skill.trajectory.replay_trajectory",
                "--traj-path", state_file+".h5",
                "--"+args.use_env_state, 
                "--save-traj", "--target-control-mode", "pd_joint_pos",
                "--obs-mode", "pointcloud", "--num-envs", "1", "--count", str(process_num)
            ], check=True)
        else:
            print("Converting original trajectory to rgbd+seg...")
            subprocess.run([
                "python", "-m", "mani_skill.trajectory.replay_trajectory",
                "--traj-path", state_file+".h5",
                "--"+args.use_env_state, 
                "--save-traj", "--no-save-video",
                "--target-control-mode", "pd_joint_pos",
                "--obs-mode", "rgb+depth+segmentation", "--num-envs", "1", "--count", str(process_num)
            ], check=True)


            # prepare pointcloud
            print("Converting original trajectory to pointcloud...")
            subprocess.run([
                "python", "-m", "mani_skill.trajectory.replay_trajectory",
                "--traj-path", state_file+".h5",
                "--"+args.use_env_state,
                "--save-traj", "--no-save-video",
                "--target-control-mode", "pd_joint_pos",
                "--obs-mode", "pointcloud", "--num-envs", "1", "--count", str(process_num)
            ], check=True)            

    if args.tozarr:
        rgbds_file = state_file + ".rgb+depth+segmentation.pd_joint_pos.physx_cpu"
        pc_file = state_file + ".pointcloud.pd_joint_pos.physx_cpu"
        # convert to zarr
        converter = ManiskillToZarrConverter(
            robot_pc_dir_path=robot_pc_dir_path,
            bounding_box=bounding_box,
            n_sample_pc=args.n_sample_pc
        )
        
        data = converter.convert(rgbds_file, pc_file, process_num=process_num)
        converter.save_to_zarr(data, args.zarr_save_path) 
        if args.visualize:
            server = viser.ViserServer()
            visualize_all_frames_with_viser(
                server,
                data['data/obs/point_clouds'].transpose(0, 2, 1),
                data['data/obs/point_colors'].transpose(0, 2, 1),
                data['data/obs/current_robot_points'].transpose(0, 2, 1),
                data['data/actions/target_robot_points'].transpose(0, 2, 1)
            )

        
class ManiskillToZarrConverter:
    def __init__(self, robot_pc_dir_path, bounding_box=None, n_sample_pc=512):
        self.robot_pc_dir_path = robot_pc_dir_path
        self.body_name_list = np.load(Path(robot_pc_dir_path).absolute() / "body_name_list.npy", allow_pickle=True)
        self.body_name_to_pc_canonical_downsampled = np.load(Path(robot_pc_dir_path).absolute() / "body_name_to_pc_canonical_downsampled.npy", allow_pickle=True)
        self.concatenated_canonical_points = np.load(Path(robot_pc_dir_path).absolute() / "concatenated_canonical_points.npy")
        self.concatenated_point_body_indices = np.load(Path(robot_pc_dir_path).absolute() / "concatenated_point_body_indices.npy")
        self.bounding_box = bounding_box or [[-0.4, -5, 0.01], [5, 5, 5]]
        self.n_sample_pc = n_sample_pc

    def _qpos_to_transform(self, qpos, env):
        """
        qpos: (n_time_step, dof)
        env: mani_skill env
        ---
        transforms: (n_time_step, n_links, 4, 4) | links are in the order of self.body_name_list
        """
        robot = env.unwrapped.agent.robot
        n_time_step = qpos.shape[0]
        n_links = len(self.body_name_list)
        robot_link_name_to_link = {link_name:sapien_utils.get_obj_by_name(env.agent.robot.get_links(), link_name) for link_name in env.unwrapped.robot_link_names}
        transforms = np.zeros((n_time_step, n_links, 4, 4))
        for i in range(n_time_step):
            robot.set_qpos(qpos[i])
            for j, link_name in enumerate(self.body_name_list):
                link_pose = robot_link_name_to_link[link_name].pose.raw_pose.detach().cpu().numpy()[0]
                transforms[i, j] = sapien.Pose(p=link_pose[:3], q=link_pose[3:]).to_transformation_matrix()
        return transforms

    def _transforms_to_robot_pc(self, current_transforms, tgt_transforms):
        """
        Args:
        - current_transforms: (n_time_step, n_links, 4, 4)
        - tgt_transforms: (n_time_step, n_links, 4, 4)
        
        Returns:
        - current_robot_points: (n_time_step, n_all_pc, 3)
        - tgt_robot_points: (n_time_step, n_all_pc, 3)
        """
        canonical_points, point_body_indices = (
            self.concatenated_canonical_points,
            self.concatenated_point_body_indices,
        )
        current_rotations = current_transforms[:, :, :3, :3]  # Shape: (T, n_body, 3, 3)
        current_translations = current_transforms[:, :, :3, 3]  # Shape: (T, n_body, 3)
        tgt_rotations = tgt_transforms[:, :, :3, :3]  # Shape: (T, n_body, 3, 3)
        tgt_translations = tgt_transforms[:, :, :3, 3]  # Shape: (T, n_body, 3)
        current_point_rotations = current_rotations[
            :, point_body_indices
        ]  # Shape: (T, n_all_pc, 3, 3)
        current_point_translations = current_translations[
            :, point_body_indices
        ]  # Shape: (T, n_all_pc, 3)
        tgt_point_rotations = tgt_rotations[
            :, point_body_indices
        ]  # Shape: (T, n_all_pc, 3, 3)
        tgt_point_translations = tgt_translations[
            :, point_body_indices
        ]  # Shape: (T, n_all_pc, 3)
        canonical_points_expanded = canonical_points[
            np.newaxis, :, :, np.newaxis
        ]  # Shape: (1, n_all_pc, 3, 1)
        current_robot_points = (
            np.matmul(current_point_rotations, canonical_points_expanded).squeeze(-1)
            + current_point_translations
        )  # Shape: (T, n_all_pc, 3)
        tgt_robot_points = (
            np.matmul(tgt_point_rotations, canonical_points_expanded).squeeze(-1)
            + tgt_point_translations
        )  # Shape: (T, n_all_pc, 3)
        return current_robot_points, tgt_robot_points
        

    def convert(self, rgbd_file, pc_file, process_num=2) -> dict:
        """
        Input:
        rgbd_file: maniskill official format .h5 and .json trajectory data in pd_joint_pos mode
        pc_file: maniskill official format .h5 and .json trajectory data in pd_joint_pos mode
        process_num: number of processes to use

        Output:
        mani_data: zarr dataset
        """

        mani_data = {}

        # read env metadata
        with open(rgbd_file+'.json', 'r') as f:
            traj_metadata = json.load(f)

        # env init
        env_info = traj_metadata['env_info']
        env_id = env_info['env_id']
        env_kwargs = env_info['env_kwargs']
        env = gym.make(env_id, **env_kwargs)
        robot = env.unwrapped.agent.robot
        # get segment id to link name mapping for robot mask
        link_seg_id_to_link_name = {}
        for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
            if isinstance(obj, Link):
                link_seg_id_to_link_name[obj_id] = obj.name

        # get link names and joint names 
        joint_name_list = [joint.get_name() for joint in robot.get_active_joints()]
        mani_data["meta/link_name_list"] = np.array(self.body_name_list, dtype=str)
        mani_data["meta/joint_name_list"] = np.array(joint_name_list, dtype=str)

        n_trajectories = process_num
        # read rgbd trajectory data
        with h5py.File(rgbd_file+'.h5', 'r') as f:
 
            # Create lists to hold the stacked data
            images = []
            depths = []
            robot_masks = []
            agent_qposes = []
            agent_qvels = []
            agent_tcp_poses = []
            agent_tcp_quats = []
            current_transforms = []
            target_transforms = []
            target_qposes = []
            actions = []
            trajectory_ends = []
            target_robot_pc = []
            current_robot_pc = []

            # Loop over all trajectories
            for traj_idx in range(n_trajectories):

                traj_key = f"traj_{traj_idx}"
                print(f"Processing {traj_key} rgbd")

                obs = f[traj_key]['obs']

                # process actions and episode ends
                traj_actions = f[traj_key]['actions'][:]
                padded_actions = np.concatenate([np.array(traj_actions), traj_actions[-1:]], axis=0)
                
                if traj_idx == 0:
                    trajectory_ends.append(len(traj_actions)+1)
                else:
                    trajectory_ends.append(trajectory_ends[-1]+len(traj_actions)+1)
                
                # record agent data
                traj_agent_qpos = obs['agent']['qpos'][:]
                traj_agent_qvel = obs['agent']['qvel'][:]
                traj_tcp_pos = obs['extra']['tcp_pose'][:, :3] # only need position
                traj_tcp_quat = obs['extra']['tcp_pose'][:, 3:] # only need rotation
                
                # process target qpos and target transforms                
                target_qpos = np.concatenate([np.array(traj_actions), traj_actions[-1:]], axis=0)
                if traj_agent_qpos.shape[-1] != target_qpos.shape[-1]:
                    # there is a mimic joint in panda
                    assert traj_agent_qpos.shape[-1] == target_qpos.shape[-1] + 1, f"traj_agent_qpos.shape[-1]: {traj_agent_qpos.shape[-1]}, target_qpos.shape[-1]: {target_qpos.shape[-1]}"
                    # 1. Compute the finger positions for all rows at once.
                    finger_positions = (target_qpos[:, -1] + 1) * 0.04 / 2  # shape: (n_time_step,)
                    # 2. Remove the last column (gripper command) from the actions.
                    actions_no_gripper = target_qpos[:, :-1]  # shape: (n_time_step, dof-2)
                    # 3. Create two columns for the finger positions.
                    # Each row gets [finger_pos, finger_pos].
                    finger_columns = np.repeat(finger_positions[:, np.newaxis], 2, axis=1)  # shape: (n_time_step, 2)
                    # 4. Concatenate the non-gripper part of the actions with the two finger position columns.
                    target_qpos = np.concatenate([actions_no_gripper, finger_columns], axis=1)  # shape: (n_time_step, dof)

                # get current and target transforms
                traj_current_transforms = self._qpos_to_transform(traj_agent_qpos, env)
                traj_target_transforms = self._qpos_to_transform(target_qpos, env)
                
                # start robot pc processing
                traj_current_robot_pc, traj_target_robot_pc = self._transforms_to_robot_pc(traj_current_transforms, traj_target_transforms)

                # append traj data
                agent_qposes.append(traj_agent_qpos)
                agent_qvels.append(traj_agent_qvel)
                agent_tcp_poses.append(traj_tcp_pos)
                agent_tcp_quats.append(traj_tcp_quat)
                current_transforms.append(traj_current_transforms)
                target_transforms.append(traj_target_transforms)
                target_qposes.append(target_qpos) 
                actions.append(padded_actions)  
                target_robot_pc.append(traj_target_robot_pc)
                current_robot_pc.append(traj_current_robot_pc)

                # Stack images for all cameras in this trajectory
                traj_images = []
                traj_depths = []
                traj_robot_masks = []

                camera_names = list(obs['sensor_data'].keys())
                mani_data["meta/camera_name_list"] = np.array(camera_names, dtype=str)

                # record camera related data
                for camera_name in camera_names:
                    # record camera metadata
                    if traj_idx == 0:
                        camera_data = obs['sensor_data'][camera_name]
                        camera_param = obs['sensor_param'][camera_name]
                        
                        # Get camera intrinsics and extrinsics
                        K = camera_param['intrinsic_cv'][0]  # Shape: (3,3)
                        X = camera_param['extrinsic_cv'][0]  # Shape: (3,4)
                        X = np.vstack([X, np.array([0, 0, 0, 1])]) # Shape: (4,4)
                        h = camera_data['rgb'].shape[1]
                        w = camera_data['rgb'].shape[2]
                        
                        # Store in camera_meta dict
                        mani_data[f"meta/camera_meta/{camera_name}/K"] = K.astype(np.float32)
                        mani_data[f"meta/camera_meta/{camera_name}/X"] = X.astype(np.float32) 
                        mani_data[f"meta/camera_meta/{camera_name}/h"] = np.array(h, dtype=np.int64)
                        mani_data[f"meta/camera_meta/{camera_name}/w"] = np.array(w, dtype=np.int64)
                        mani_data[f"meta/camera_meta/{camera_name}/parent_frame_name"] = np.array("world" if camera_name == "base_camera" else "camera_link", dtype=str)
                    
                    # camera wise data
                    image_camera = obs['sensor_data'][camera_name]['rgb'][:]
                    depth_camera = obs['sensor_data'][camera_name]['depth'][:]
                    traj_images.append(image_camera)
                    traj_depths.append(depth_camera)

                    # robot mask of this camera
                    segmentation_camera = obs['sensor_data'][camera_name]['segmentation'][:]
                    robot_mask = np.isin(segmentation_camera[:,:,:,0], list(link_seg_id_to_link_name.keys())).astype(np.uint8)
                    robot_mask = robot_mask[:,:,:,np.newaxis]
                    traj_robot_masks.append(robot_mask)

                # Stack data for all cameras in this trajectory and add to the overall list
                images.append(np.stack(traj_images, axis=1))  # Shape: (n_steps, n_camera, 3, h, w)
                depths.append(np.stack(traj_depths, axis=1))  # Shape: (n_steps, n_camera, 1, h, w)
                robot_masks.append(np.stack(traj_robot_masks, axis=1))  # Shape: (n_steps, n_camera, 1, h, w)

            # final reform
            images = np.concatenate(images, axis=0)
            depths = np.concatenate(depths, axis=0)
            robot_masks = np.concatenate(robot_masks, axis=0)
            agent_qposes = np.concatenate(agent_qposes, axis=0)
            agent_qvels = np.concatenate(agent_qvels, axis=0)
            agent_tcp_poses = np.concatenate(agent_tcp_poses, axis=0)
            agent_tcp_quats = np.concatenate(agent_tcp_quats, axis=0)
            current_transforms = np.concatenate(current_transforms, axis=0)
            target_transforms = np.concatenate(target_transforms, axis=0)
            target_qposes = np.concatenate(target_qposes, axis=0)
            actions = np.concatenate(actions, axis=0)
            trajectory_ends = np.array(trajectory_ends, dtype=np.int64)
            target_robot_pc = np.concatenate(target_robot_pc, axis=0)
            current_robot_pc = np.concatenate(current_robot_pc, axis=0)
        
            # save data
            mani_data["data/obs/images"] = images.astype(np.uint8).transpose(0, 1, 4, 2, 3)
            mani_data["data/obs/depths"] = depths.astype(np.float32).transpose(0, 1, 4, 2, 3)
            mani_data["data/obs/robot_masks"] = robot_masks.astype(np.uint8).transpose(0, 1, 4, 2, 3)
            mani_data["data/obs/agent_all_qpos"] = agent_qposes.astype(np.float32)
            mani_data["data/obs/agent_all_qvel"] = agent_qvels.astype(np.float32)
            mani_data["data/obs/agent_tcp_pos"] = agent_tcp_poses.astype(np.float32)
            mani_data["data/obs/agent_tcp_quat"] = agent_tcp_quats.astype(np.float32)
            mani_data["data/obs/current_transforms"] = current_transforms.astype(np.float32)
            mani_data["data/actions/target_transforms"] = target_transforms.astype(np.float32)
            mani_data["data/actions/target_all_qpos"] = target_qposes.astype(np.float32)
            mani_data["data/actions/original_actions"] = actions.astype(np.float32)
            mani_data["meta/episode_ends"] = trajectory_ends.astype(np.int64)
            mani_data["data/obs/current_robot_points"] = current_robot_pc.astype(np.float32).transpose(0, 2, 1)
            mani_data["data/actions/target_robot_points"] = target_robot_pc.astype(np.float32).transpose(0, 2, 1)

        # read pc trajectory data
        with h5py.File(pc_file+'.h5', 'r') as f:

            # Create lists to hold the stacked data
            pc = []
            point_colors = []
            for traj_idx in range(n_trajectories):

                traj_key = f"traj_{traj_idx}"
                print(f"Processing {traj_key} pointcloud")

                obs_pointcloud = f[traj_key]['obs']['pointcloud']
                traj_pc = obs_pointcloud['xyzw'][:]  # (n_steps, n_points, 4)
                traj_point_colors = obs_pointcloud['rgb'][:]  # (n_steps, n_points, 3)

                # Create mask for w=1 points
                mask = traj_pc[...,3] == 1  # (n_steps, n_points)

                # Process each timestep to maintain proper shapes
                processed_pc = []
                processed_colors = []
                for step in range(len(traj_pc)):
                    # only w=1 points
                    step_pc = traj_pc[step][mask[step],:] 
                    step_colors = traj_point_colors[step][mask[step],:]
                    step_pc = step_pc[:,:3]

                    # filter points within bounding box
                    min_bound = self.bounding_box[0]
                    max_bound = self.bounding_box[1]
                    # Filter points within bounding box
                    valid_indices = np.all((step_pc > min_bound) & (step_pc < max_bound), axis=1)
                    step_pc = step_pc[valid_indices]
                    step_colors = step_colors[valid_indices]

                    step_pc, step_colors = fpsample_pc(step_pc, step_colors, self.n_sample_pc)

                    processed_pc.append(step_pc)  # Only take xyz coordinates
                    processed_colors.append(step_colors)

                # Stack back into 3D arrays
                traj_pc = np.stack(processed_pc)  # (n_steps, n_valid_points, 3)
                traj_point_colors = np.stack(processed_colors)  # (n_steps, n_valid_points, 3)

                # append traj data
                pc.append(traj_pc)
                point_colors.append(traj_point_colors)

            
            # save data
            mani_data['data/obs/point_clouds'] = np.concatenate(pc, axis=0).astype(np.float32).transpose(0, 2, 1)
            mani_data['data/obs/point_colors'] = np.concatenate(point_colors, axis=0).astype(np.uint8).transpose(0, 2, 1)

        return mani_data

    def save_to_zarr(self, data, zarr_path):
        """Save converted data to zarr format"""
        zarr_dataset = PovZarrDataset(zarr_path)
        zarr_dataset.save_data(data)
        zarr_dataset.print_structure()



if __name__ == "__main__":
    args = parse_args()
    main(args)
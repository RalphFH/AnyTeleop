import h5py
import json
import numpy as np
import gymnasium as gym
import open3d as o3d
from urdf_parser_py.urdf import URDF
import pytorch_kinematics as pk
import subprocess


from mani_skill.utils.structs import Link
from pov.datasets.pov_zarr_dataset import PovZarrDataset
from pov.utils.camera.pc import fpsample_pc

class ManiskillToZarrConverter:
    def __init__(self, robot_path, robot_name, bounding_box=None):
        self.robot_path = robot_path
        self.robot_name = robot_name
        self.bounding_box = bounding_box or [[-0.3, -0.3, 0.01], [0.2, 0.3, 0.9]]
        
        # Initialize robot-related attributes
        self.urdf_path = robot_path + robot_name + ".urdf"
        self.links_pcs_dict = self._get_robot_pointcloud()
        self.chain = pk.build_chain_from_urdf(open(self.urdf_path, mode="rb").read())

    def _get_robot_pointcloud(self) -> dict:
        """Get links pointcloud dict from urdf file and geometry meshes"""
        urdf_file = self.urdf_path
        robot = URDF.from_xml_file(urdf_file)
        
        pointclouds = {}
        for link in robot.links:
            if link.visual and hasattr(link.visual.geometry, 'filename'):
                mesh_file = self.robot_path + link.visual.geometry.filename
            else:
                continue # virtual link
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            if link.name in ["panda_hand", "panda_leftfinger", "panda_rightfinger"]:
                o3d_pc = mesh.sample_points_poisson_disk(200)
            else:
                o3d_pc = mesh.sample_points_poisson_disk(20)

            # if mesh_file.endswith(".stl"): # no color info
            #     o3d_pc.colors = o3d.utility.Vector3dVector(np.ones((len(o3d_pc.points), 3)) * [0, 0, 1])
            
            pointclouds[link.name] = o3d_pc # (n_points, 3)
        
        return pointclouds

    def _qpos_to_transform(self, qpos, base_to_world) -> tuple[dict, np.ndarray]:
        """Apply Forward Kinematics to get transforms for each link from joint positions"""
        res = self.chain.forward_kinematics(qpos)
        link_names = list(res.keys())
        transforms_list = []
        transforms_dict = {}
        for link_name in link_names:
            link_matrix = res[link_name].get_matrix().numpy()  # Convert tensor to numpy

            # transform link_matrix to world frame
            base_to_world_expanded = np.tile(base_to_world[None, :, :], (link_matrix.shape[0], 1, 1))
            link_matrix = np.matmul(base_to_world_expanded, link_matrix) 

            transforms_list.append(link_matrix)
            transforms_dict[link_name] = link_matrix

        stacked_transforms = np.stack(transforms_list, axis=1)
        return transforms_dict, stacked_transforms

    def _transform_link_points(self, points, transforms) -> np.ndarray:
        """Transform points by applying rotation and translation to each step"""
        rotations = transforms[:, :3, :3]      # (n_steps, 3, 3)
        translations = transforms[:, :3, 3]     # (n_steps, 3)

        points_expanded = points[np.newaxis, :, :, np.newaxis]  # (1, n_points, 3, 1)
        rotations_expanded = rotations[:, np.newaxis]           # (n_steps, 1, 3, 3)

        transformed_points = (
            np.matmul(rotations_expanded, points_expanded).squeeze(-1)  # rotate
            + translations[:, np.newaxis]                               # translate
        )  # (n_steps, n_points, 3)

        return transformed_points

    def convert(self, rgbd_file, pc_file, process_num=2) -> dict:
        """Convert maniskill official format .h5 and .json trajectory data to zarr dataset"""
        urdf_path = self.urdf_path

        mani_data = {}

        # read env metadata
        with open(rgbd_file+'.json', 'r') as f:
            traj_metadata = json.load(f)

        # env init
        env_info = traj_metadata['env_info']
        env_id = env_info['env_id']
        env_kwargs = env_info['env_kwargs']
        env = gym.make(env_id, **env_kwargs)

        # get segment id to link name mapping for robot mask
        link_list = {}
        for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
            if isinstance(obj, Link):
                link_list[obj_id] = obj.name

        env.close()

        link_names = [link_list[i] for i in sorted(link_list.keys())]
        # link_names = chain.get_link_names()
        mani_data["meta/link_name_list"] = np.array(link_names, dtype=str)
        mani_data["meta/joint_name_list"] = np.array(self.chain.get_joint_parameter_names(), dtype=str)

        n_trajectories = process_num
        # read rgbd trajectory data
        with h5py.File(rgbd_file+'.h5', 'r') as f:
 
            # Create lists to hold the stacked data
            images = []
            depths = []
            robot_masks = []
            agent_qpos = []
            agent_qvel = []
            agent_tcp_pos = []
            current_transforms = []
            target_transforms = []
            target_qpos = []
            actions = []
            trajectory_ends = []
            target_robot_pc = []
            current_robot_pc = []

            # Loop over all trajectories
            for traj_idx in range(n_trajectories):

                traj_key = f"traj_{traj_idx}"
                print(f"Processing {traj_key} rgbd")

                if traj_idx == 0:
                    agent_name = list(f[traj_key]['env_states']['articulations'].keys())[0]
                    env_states = f[traj_key]['env_states']['articulations'][agent_name][0,:3]
                    base_to_world = np.eye(4)
                    base_to_world[:3,3] = env_states

                obs = f[traj_key]['obs']

                # process episode ends
                traj_actions = f[traj_key]['actions'][:]
                padded_traj_actions = np.concatenate([traj_actions[0:1], np.array(traj_actions)], axis=0)
                
                if traj_idx == 0:
                    trajectory_ends.append(len(traj_actions)+1)
                else:
                    trajectory_ends.append(trajectory_ends[-1]+len(traj_actions)+1)
                
                # record agent data
                traj_agent_qpos = obs['agent']['qpos'][:]
                traj_agent_qvel = obs['agent']['qvel'][:]
                traj_tcp_pos = obs['extra']['tcp_pose'][:, :3] # only need position
                
                # process actions and target transforms
                gripper_open_pos = 0.04
                gripper_close_pos = 0.0
                # change actions [n_transition, dof-1] to [n_time_step, dof]
                traj_qpos_actions = []
                for action in traj_actions:
                    qpos_action = action.copy()
                    # if action = gripper open, remove the column and add two columns of 0.04
                    if qpos_action[-1] == 1:
                        qpos_action = np.delete(qpos_action, -1)
                        qpos_action = np.concatenate([qpos_action, np.array([gripper_open_pos, gripper_open_pos])], axis=0)
                    # if action = gripper close, remove the column and add two columns of 0.0
                    elif qpos_action[-1] == -1:
                        qpos_action = np.delete(qpos_action, -1)
                        qpos_action = np.concatenate([qpos_action, np.array([gripper_close_pos, gripper_close_pos])], axis=0)
                    traj_qpos_actions.append(qpos_action)
                
                # add the first step qpos to the beginning of the actions
                traj_qpos_actions = np.concatenate([traj_agent_qpos[0:1], np.array(traj_qpos_actions)], axis=0)
                
                # get current and target transforms
                current_transform_dict, traj_current_transforms = self._qpos_to_transform(traj_agent_qpos, base_to_world)
                target_transform_dict, traj_target_transforms = self._qpos_to_transform(traj_qpos_actions, base_to_world)
                
                # start robot pc processing
                traj_current_robot_pc = []
                traj_target_robot_pc = []
                for link in self.links_pcs_dict.keys():
                    current_link_pc = self._transform_link_points(np.asarray(self.links_pcs_dict[link].points), current_transform_dict[link])
                    target_link_pc = self._transform_link_points(np.asarray(self.links_pcs_dict[link].points), target_transform_dict[link])
                    traj_current_robot_pc.append(current_link_pc)
                    traj_target_robot_pc.append(target_link_pc)
                # reform
                traj_current_robot_pc = np.concatenate(traj_current_robot_pc, axis=1)
                traj_target_robot_pc = np.concatenate(traj_target_robot_pc, axis=1)

                # append traj data
                agent_qpos.append(traj_agent_qpos)
                agent_qvel.append(traj_agent_qvel)
                agent_tcp_pos.append(traj_tcp_pos)
                current_transforms.append(traj_current_transforms)
                target_transforms.append(traj_target_transforms)
                target_qpos.append(traj_qpos_actions) 
                actions.append(padded_traj_actions) 
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
                    robot_mask = np.isin(segmentation_camera[:,:,:,0], list(link_list.keys())).astype(np.uint8)
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
            agent_qpos = np.concatenate(agent_qpos, axis=0)
            agent_qvel = np.concatenate(agent_qvel, axis=0)
            agent_tcp_pos = np.concatenate(agent_tcp_pos, axis=0)
            current_transforms = np.concatenate(current_transforms, axis=0)
            target_transforms = np.concatenate(target_transforms, axis=0)
            target_qpos = np.concatenate(target_qpos, axis=0)
            actions = np.concatenate(actions, axis=0)
            trajectory_ends = np.array(trajectory_ends, dtype=np.int64)
            target_robot_pc = np.concatenate(target_robot_pc, axis=0)
            current_robot_pc = np.concatenate(current_robot_pc, axis=0)
        
            # save data
            mani_data["data/obs/images"] = images.astype(np.uint8).transpose(0, 1, 4, 2, 3)
            mani_data["data/obs/depths"] = depths.astype(np.float32).transpose(0, 1, 4, 2, 3)
            mani_data["data/obs/robot_masks"] = robot_masks.astype(np.uint8).transpose(0, 1, 4, 2, 3)
            mani_data["data/obs/panda_all_qpos"] = agent_qpos.astype(np.float32)
            mani_data["data/obs/panda_all_qvel"] = agent_qvel.astype(np.float32)
            mani_data["data/obs/panda_tcp_pos"] = agent_tcp_pos.astype(np.float32)
            mani_data["data/obs/current_transforms"] = current_transforms.astype(np.float32)
            mani_data["data/actions/target_transforms"] = target_transforms.astype(np.float32)
            mani_data["data/actions/target_all_qpos"] = target_qpos.astype(np.float32)
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

                    step_pc, step_colors = fpsample_pc(step_pc, step_colors, 224)

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


def main():

    process_num = 10
    bounding_box = [[-0.3, -0.3, 0.01], [0.2, 0.3, 0.9]]

    # urdf and hdf5 data
    robot_path = "robots/panda/"
    robot_name = "panda_v3"
    hdf5_dir = "../../../data/maniskill_demos/PullCubeTool-v1/motionplanning"
    state_file = hdf5_dir + "/trajectory"

    print("Converting original trajectory to rgbd+seg...")
    subprocess.run([
        "python", "-m", "mani_skill.trajectory.replay_trajectory",
        "--traj-path", state_file+".h5",
        "--use-first-env-state", "-b", "physx_cuda",
        "--save-traj", "--target-control-mode", "pd_joint_pos",
        "--obs-mode", "rgb+depth+segmentation", "--num-envs", "10", "--count", str(process_num)
    ], check=True)
    rgbds_file = hdf5_dir + "/trajectory.rgb+depth+segmentation.pd_joint_pos.physx_cuda"

    # prepare pointcloud
    print("Converting original trajectory to pointcloud...")
    subprocess.run([
        "python", "-m", "mani_skill.trajectory.replay_trajectory",
        "--traj-path", state_file+".h5",
        "--use-first-env-state", "-b", "physx_cuda",
        "--save-traj", "--target-control-mode", "pd_joint_pos",
        "--obs-mode", "pointcloud", "--num-envs", "10", "--count", str(process_num)
    ], check=True)
    pc_file = hdf5_dir + "/trajectory.pointcloud.pd_joint_pos.physx_cuda"

    # convert to zarr
    converter = ManiskillToZarrConverter(
        robot_path=robot_path,
        robot_name=robot_name,
        bounding_box=bounding_box
    )
    
    data = converter.convert(rgbds_file, pc_file, process_num=process_num)
    converter.save_to_zarr(data, "converted_data.zarr")

if __name__ == "__main__":
    main()
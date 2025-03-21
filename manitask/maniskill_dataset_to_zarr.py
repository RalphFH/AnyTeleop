import h5py
import json
import numpy as np
import gymnasium as gym
import open3d as o3d
from urdf_parser_py.urdf import URDF
import pytorch_kinematics as pk

from mani_skill.utils.structs import Link
from pov.datasets.pov_zarr_dataset import PovZarrDataset
from utils.hand_link import hand_link
import viser
import time

from dataclasses import dataclass
import tyro

@dataclass
class Args:
    pc_file: str
    rgbd_file: str
    robot_path: str
    robot_name: str
    episode_num: int = 1
    zarr_save_path: str = "./converted.zarr"
    visualize: bool = True

def parse_args():
    return tyro.cli(Args)

def visualize_all_frames_with_viser(server,pointclouds: np.ndarray, colors: np.ndarray, interval: float = 0.03):
    """
    Visualize all point cloud frames using viser in sequence (like animation).

    Args:
        pointclouds: (n_steps, n_points, 3)
        colors: (n_steps, n_points, 3)
        interval: time delay between frames in seconds
    """

    # normalize color to [0, 1]
    colors = colors.astype(np.float32) / 255.0

    print("Viser running at http://localhost:8080 — streaming point cloud...")

    for i in range(pointclouds.shape[0]):
        pc = pointclouds[i]
        col = colors[i]

        server.add_point_cloud(
            name="pointcloud",
            points=pc,
            colors=col,
            point_size=0.01,
            point_shape="circle",
        )

        time.sleep(interval)

    print("Finished playing all frames.")


def main(args: Args):
    data = convert_trajectory_to_zarr(
        args.rgbd_file,
        args.pc_file,
        args.robot_path,
        args.robot_name,
        args.episode_num
    )

    if args.visualize:
        server = viser.ViserServer()
        visualize_all_frames_with_viser(
            server,
            data['data/obs/point_clouds'].transpose(0, 2, 1),
            data['data/obs/point_colors'].transpose(0, 2, 1)
        )

    zarr_dataset = PovZarrDataset(args.zarr_save_path)
    zarr_dataset.save_data(data)
    zarr_dataset.print_structure()

def qpos_to_transform(qpos, chain, base_to_world) -> tuple[dict, np.ndarray]:
    """
    Apply Forward Kinematics to get transforms for each link from joint positions
    """
    res = chain.forward_kinematics(qpos)
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

    stacked_transforms = np.stack(transforms_list, axis=1) # (n_steps, n_links, 4, 4)
    return transforms_dict, stacked_transforms

def get_robot_pointcloud(robot_path, robot_name) -> dict:
    """
    get links pointcloud dict from urdf file and geometry meshes
    """

    urdf_file = robot_path + robot_name + ".urdf"
    arm, hand, hand_type = robot_name.split("_")
    robot = URDF.from_xml_file(urdf_file)
    
    pointclouds = {}
    for link in robot.links:
        if link.visual and hasattr(link.visual.geometry, 'filename'):
            mesh_file = robot_path + link.visual.geometry.filename
        else:
            continue # virtual link
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        if link.name in hand_link[hand]:
            o3d_pc = mesh.sample_points_poisson_disk(50)
        else:
            o3d_pc = mesh.sample_points_poisson_disk(20)

        # if mesh_file.endswith(".stl"): # no color info
        #     o3d_pc.colors = o3d.utility.Vector3dVector(np.ones((len(o3d_pc.points), 3)) * [0, 0, 1])
            
        pointclouds[link.name] = o3d_pc # (n_points, 3)
    
    return pointclouds

def transform_link_points(points, transforms) -> np.ndarray:
    """
    Transform points by applying rotation and translation to each step
    """
    rotations = transforms[:, :3, :3]      # (n_steps, 3, 3)
    translations = transforms[:, :3, 3]     # (n_steps, 3)

    points_expanded = points[np.newaxis, :, :, np.newaxis]  # (1, n_points, 3, 1)
    rotations_expanded = rotations[:, np.newaxis]           # (n_steps, 1, 3, 3)

    transformed_points = (
        np.matmul(rotations_expanded, points_expanded).squeeze(-1)  # rotate
        + translations[:, np.newaxis]                               # translate
    )  # (n_steps, n_points, 3)



    return transformed_points

def convert_trajectory_to_zarr(rgbd_trajectory, pc_trajectory, robot_path, robot_name, process_num) -> dict:
    """
    Convert maniskill official format .h5 and .json trajectory data to zarr dataset
    """
    
    urdf_path = robot_path + robot_name + ".urdf"

    mani_data = {}

    links_pcs_dict = get_robot_pointcloud(robot_path, robot_name) # for robot pc
    #robot_points_color = np.concatenate([np.asarray(pc.colors) for pc in links_pcs_dict.values()], axis=0)
    chain = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read()) # for getting transforms from qpos

    # read env metadata
    with open(rgbd_trajectory+'.json', 'r') as f:
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
    mani_data["meta/joint_name_list"] = np.array(chain.get_joint_parameter_names(), dtype=str)

    n_trajectories = process_num
    # read rgbd trajectory data
    with h5py.File(rgbd_trajectory+'.h5', 'r') as f:
 
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
        #env_states = []
        trajectory_ends = []
        target_robot_pc = []
        current_robot_pc = []


        # Loop over all trajectories
        for traj_idx in range(n_trajectories):

            traj_key = f"traj_{traj_idx}"
            print(f"Processing {traj_key} rgbd")

            if traj_idx == 0:
                agent_name = list(f[traj_key]['env_states']['articulations'].keys())[0] # "xarm7_allegro_right"
                env_states = f[traj_key]['env_states']['articulations'][agent_name][0,:3] # robot root position
                base_to_world = np.eye(4)
                base_to_world[:3,3] = env_states
                # env_state = common.flatten_state_dict(env_state)
                # env_states.append(env_state)

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
            # gripper_open_pos = 0.04
            # gripper_close_pos = 0.0
            # change actions [n_transition, dof-1] to [n_time_step, dof]
            traj_qpos_actions = traj_actions.copy()

            # for action in traj_actions:
            #     qpos_action = action.copy()
            #     # if action = gripper open, remove the column and add two columns of 0.04
            #     if qpos_action[-1] == 1:
            #         qpos_action = np.delete(qpos_action, -1)
            #         qpos_action = np.concatenate([qpos_action, np.array([gripper_open_pos, gripper_open_pos])], axis=0)
            #     # if action = gripper close, remove the column and add two columns of 0.0
            #     elif qpos_action[-1] == -1:
            #         qpos_action = np.delete(qpos_action, -1)
            #         qpos_action = np.concatenate([qpos_action, np.array([gripper_close_pos, gripper_close_pos])], axis=0)
            #     traj_qpos_actions.append(qpos_action)   
            
            """
            why add the first step qpos to the beginning of the actions? why lack of one step in the actions?
            """
            # add the first step qpos to the beginning of the actions  
            traj_qpos_actions = np.concatenate([traj_agent_qpos[0:1], np.array(traj_qpos_actions)], axis=0)
            # get current and target transforms
            current_transform_dict, traj_current_transforms = qpos_to_transform(traj_agent_qpos, chain, base_to_world)

            target_transform_dict, traj_target_transforms = qpos_to_transform(traj_qpos_actions, chain, base_to_world)
            
            # start robot pc processing
            traj_current_robot_pc = []
            traj_target_robot_pc = []
            for link in links_pcs_dict.keys():
                current_link_pc = transform_link_points(np.asarray(links_pcs_dict[link].points), current_transform_dict[link])
                target_link_pc = transform_link_points(np.asarray(links_pcs_dict[link].points), target_transform_dict[link])
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

            camera_names = list(obs['sensor_data'].keys()) # 'base_camera'
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
                segmentation_camera = obs['sensor_data'][camera_name]['segmentation'][:] # Shape: (n_steps, h, w, 1)
                robot_mask = np.isin(segmentation_camera[:,:,:,0], list(link_list.keys())).astype(np.uint8)  # Shape: (n_steps, h, w)
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
        #env_states = np.concatenate(env_states, axis=0)
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
        #mani_data["data/states"] = env_states.astype(np.float32)
        mani_data["meta/episode_ends"] = trajectory_ends.astype(np.int64)
        mani_data["data/obs/current_robot_points"] = current_robot_pc.astype(np.float32).transpose(0, 2, 1)
        mani_data["data/actions/target_robot_points"] = target_robot_pc.astype(np.float32).transpose(0, 2, 1)


        # # point cloud processing
        # camera_K = np.stack([mani_data[f"meta/camera_meta/{camera_name}/K"] for camera_name in camera_names])
        # camera_X = np.stack([mani_data[f"meta/camera_meta/{camera_name}/X"] for camera_name in camera_names])
        
        # # Fuse point clouds from multiple camera views
        # point_clouds, point_colors = pc_fusion(
        #     mani_data["data/obs/images"],
        #     mani_data["data/obs/depths"],
        #     camera_K, 
        #     camera_X
        # )

        # mani_data['data/obs/point_clouds'] = point_clouds.astype(np.float32)
        # mani_data['data/obs/point_colors'] = point_colors.astype(np.uint8)



    # read pc trajectory data
    with h5py.File(pc_trajectory+'.h5', 'r') as f:

        # Create lists to hold the stacked data
        pc = []
        point_colors = []
        for traj_idx in range(n_trajectories):

            traj_key = f"traj_{traj_idx}"
            print(f"Processing {traj_key} pointcloud")

            obs_pointcloud = f[traj_key]['obs']['pointcloud']
            traj_pc = obs_pointcloud['xyzw'][:]  # (n_steps, n_points, 4)，4 means xyzw
            traj_point_colors = obs_pointcloud['rgb'][:]  # (n_steps, n_points, 3)

            # Create mask for w=1 and z>0.01 and x>-0,35 points
            mask_wzx = np.logical_and(traj_pc[...,3] == 1,np.logical_and(traj_pc[...,2] > 0.01, traj_pc[...,0] > -0.7))
             # (n_steps, n_points)
            
            # Process each timestep to maintain proper shapes
            processed_pc = []
            processed_colors = []
            for step in range(len(traj_pc)):
                # only w=1 and z>0.01 and x>-0,4 points
                step_pc = traj_pc[step][mask_wzx[step],:] 
                step_colors = traj_point_colors[step][mask_wzx[step],:]
                step_pc = step_pc[:,:3]
                
             
                # filter points within 1m
                distances = np.linalg.norm(step_pc, axis=1)
                valid_indices = distances < 2
                step_pc = step_pc[valid_indices]
                step_colors = step_colors[valid_indices]

                pc_downsample = 512
                # downsample to 512 points
                if step_pc.shape[0] > pc_downsample:
                    o3d_pc = o3d.geometry.PointCloud()
                    o3d_pc.points = o3d.utility.Vector3dVector(step_pc)
                    o3d_pc.colors = o3d.utility.Vector3dVector(step_colors)
                    o3d_pc = o3d_pc.farthest_point_down_sample(num_samples = pc_downsample)
                    step_pc = np.asarray(o3d_pc.points)
                    step_colors = np.asarray(o3d_pc.colors)

                processed_pc.append(step_pc)  # Only take xyz coordinates
                processed_colors.append(step_colors)

            # Stack back into 3D arrays
            # print("pc shape",processed_pc[0].shape,processed_pc[1].shape)
            traj_pc = np.stack(processed_pc)  # (n_steps, n_valid_points, 3)
            traj_point_colors = np.stack(processed_colors)  # (n_steps, n_valid_points, 3)

            # append traj data
            pc.append(traj_pc)
            point_colors.append(traj_point_colors)

        
        # save data
        mani_data['data/obs/point_clouds'] = np.concatenate(pc, axis=0).astype(np.float32).transpose(0, 2, 1)
        mani_data['data/obs/point_colors'] = np.concatenate(point_colors, axis=0).astype(np.uint8).transpose(0, 2, 1)

    return mani_data


if __name__ == "__main__":
    args = parse_args()
    main(args)
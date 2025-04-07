import viser
import zarr 
import numpy as np
import open3d as o3d
import trimesh

from pygltflib import GLTF2

def visualize_pointclouds(scene_pc, current_robot_pc, target_robot_pc):
    """Visualize point clouds using viser"""
    server = viser.ViserServer(port=8140)

    n_time_steps = len(scene_pc)

    for time_step in range(n_time_steps):
        # test with one frame of current robot
        server.scene.add_point_cloud(
            name="current_robot_pc",
            points=current_robot_pc[time_step].T,
            colors=np.tile([0, 0, 255], (current_robot_pc[time_step].shape[1], 1)), # blue
            point_size=0.002,
        )

        server.scene.add_point_cloud(
            name="target_robot_pc",
            points=target_robot_pc[time_step].T,
            colors=np.tile([0, 255, 0], (target_robot_pc[time_step].shape[1], 1)), # green
            point_size=0.002,
        )   

        server.scene.add_point_cloud(
            name="scene_pc",
            points=scene_pc[time_step].T,
            colors=np.tile([255, 0, 0], (scene_pc[time_step].shape[1], 1)), # red
            point_size=0.002,
        )
    
        
    input("Press Enter to exit visualization...")

if __name__ == "__main__":
    zarr_path = "/home/chenshan/robotics/dex-retargeting/manitask/data/zarr/PlaceSphere-v1/ur5e_allegro_right/0-29/converted_data_wzx.zarr"  # Update this path as needed
    
    root = zarr.open(zarr_path, mode='r')
    # access 'data/obs/point_clouds'
    scene_pc = root['data/obs/point_clouds']  # (T, 3, N)
    current_robot_pc = root['data/obs/current_robot_points']  # (T, 3, N)
    target_robot_pc = root['data/actions/target_robot_points'] 

    visualize_pointclouds(
        scene_pc=scene_pc,
        current_robot_pc=current_robot_pc,
        target_robot_pc=target_robot_pc,
    )






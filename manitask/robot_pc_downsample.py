import open3d as o3d
import numpy as np
from pathlib import Path
from termcolor import cprint
import fpsample 
from pov.utils.dataset.common import output_path_confirmation
from utils.sapien_link import hand_link

def maniskill_pc_downsample():
    robot_uid = "xarm7_leap_right"
    if robot_uid != "panda" and robot_uid != "panda_wristcam":
        arm,hand,_ = robot_uid.split("_")
    body_pc_dir_path = Path(f"data/assets/robots/maniskill/{arm}/{robot_uid}/body_pc")
    save_dir_path = body_pc_dir_path.parent / "body_pc_downsampled"
    output_path_confirmation(save_dir_path, only_warn=False)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # Load all point clouds
    all_body_names = sorted([body_name.stem for body_name in body_pc_dir_path.glob("*.ply")])
    body_name_to_pc_canonical = {}
    
    # Read point clouds
    for body_name in all_body_names:
        pc = o3d.io.read_point_cloud(str(body_pc_dir_path / f"{body_name}.ply"))
        body_name_to_pc_canonical[body_name] = np.asarray(pc.points).astype(np.float32)

    # Downsample point clouds
    body_name_to_pc_canonical_downsampled = {}
    all_points = []
    all_body_indices = []

    for i, body_name in enumerate(all_body_names):
        points = body_name_to_pc_canonical[body_name]
        # Determine downsampling ratio based on body part
        print(f"Processing {body_name}  with {points.shape[0]} points")
        if robot_uid == "panda" or robot_uid == "panda_wristcam":
            if "hand" in body_name:
                n_body_keep_pc = points.shape[0] // 10
            elif "finger" in body_name:  # For both leftpad and rightpad
                n_body_keep_pc = points.shape[0] // 3
            else:
                n_body_keep_pc = points.shape[0] // 120
        else:
            if body_name in hand_link[hand]:
                n_body_keep_pc = points.shape[0] // 5
            else:
                n_body_keep_pc = points.shape[0] // 120

        # Perform FPS downsampling
        body_fps_samples_idx = fpsample.fps_npdu_kdtree_sampling(points, n_body_keep_pc)
        downsampled_points = points[body_fps_samples_idx]
        
        body_name_to_pc_canonical_downsampled[body_name] = downsampled_points
        
        # Collect points and indices for concatenated arrays
        all_points.append(downsampled_points)
        all_body_indices.extend([i] * len(downsampled_points))
        
        cprint(
            f"Downsampled {body_name} from {points.shape[0]} to {downsampled_points.shape[0]}",
            "green",
        )

    # Concatenate all points and indices
    concatenated_canonical_points = np.concatenate(all_points, axis=0)
    concatenated_point_body_indices = np.array(all_body_indices)

    # Save the results
    np.save(save_dir_path / "body_name_list.npy", all_body_names)
    np.save(save_dir_path / "concatenated_canonical_points.npy", concatenated_canonical_points)
    np.save(save_dir_path / "concatenated_point_body_indices.npy", concatenated_point_body_indices)
    np.save(save_dir_path / "body_name_to_pc_canonical_downsampled.npy", body_name_to_pc_canonical_downsampled)

    cprint(f"Saved downsampled point clouds to {save_dir_path}", "green")
    cprint(f"Total points after downsampling: {len(concatenated_canonical_points)}")


if __name__ == "__main__":
    maniskill_pc_downsample()

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

def world_to_image(world_points, extrinsic_cv, intrinsic_cv):
    """
    Project 3D world points to 2D image coordinates
    
    Args:
        world_points: Nx3 array of points in world coordinates (x, y, z)
                     or single [x, y, z] point
        extrinsic_cv: 4x4 extrinsic matrix (OpenCV convention)
        intrinsic_cv: 3x3 intrinsic matrix
        
    Returns:
        Nx2 array of (u, v) image coordinates in pixels
    """    

        
    image_points = np.zeros((world_points.shape[0], 2), dtype=int)
    
    for i, point in enumerate(world_points):
        # Convert to homogeneous coordinates
        world_point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
        
        # Transform from world to camera coordinates
        camera_point = extrinsic_cv @ world_point_homogeneous
        
        # Skip points behind the camera (negative z)
        if camera_point[2] <= 0:
            image_points[i] = [-1, -1]  # Invalid point marker
            continue
        
        # Normalize by dividing by z (perspective division)
        x_cam = camera_point[0] / camera_point[2]
        y_cam = camera_point[1] / camera_point[2]
        
        # Project to image plane using intrinsic parameters
        u = intrinsic_cv[0, 0] * x_cam + intrinsic_cv[0, 2]
        v = intrinsic_cv[1, 1] * y_cam + intrinsic_cv[1, 2]
        
        image_points[i] = [int(u), int(v)]
    
    return image_points  # Return array of points


def draw_projected_point(rgb_image, world_points, extrinsic_cv, intrinsic_cv, 
                         marker_size=10, color=(255, 0, 0)):
    
    if isinstance(world_points, torch.Tensor):
        world_points = world_points.detach().cpu().numpy()

    if len(world_points.shape) == 1:
        world_points = world_points.reshape(1, 3)
    
    # Project the 3D point to 2D image coordinates
    images_points = world_to_image(world_points, extrinsic_cv, intrinsic_cv)
    
    # Convert tensor to numpy if needed
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.detach().cpu().numpy()
    
    # Create a copy of the image to avoid modifying the original
    img_with_point = rgb_image.copy()
    
    # Make sure the point is within image bounds
    h, w = img_with_point.shape[:2]
    for u, v in images_points:
        if 0 <= u < w and 0 <= v < h:
            # Draw a circle at the projected point
            cv2.circle(img_with_point, (u, v), marker_size, color, -1)
            # print(f"Point projected to image coordinates: ({u}, {v})")
        else:
            print(f"Warning: Projected point ({u}, {v}) is outside image bounds ({w}x{h})")
    
    return img_with_point
    

def visualize_point_on_all_cameras(env, world_point):
    human_render_cameras = env.unwrapped.scene.human_render_cameras
    
    for name, camera in human_render_cameras.items():
        print(f"\nProjecting point onto camera: {name}")
        
        # Get camera parameters
        params = camera.get_params()
        extrinsic_cv = params["extrinsic_cv"].squeeze(0).detach().cpu().numpy()
        intrinsic_cv = params["intrinsic_cv"].squeeze(0).detach().cpu().numpy()
        
        # Capture camera image
        camera.capture()
        rgb = camera.get_obs(rgb=True, depth=False, segmentation=False, position=False)["rgb"].squeeze(0)
        
        # Project and draw the point
        img_with_point = draw_projected_point(rgb, world_point, extrinsic_cv, intrinsic_cv)
        
        # Display the result
        plt.figure(figsize=(10, 8))
        plt.title(f"Camera: {name} with projected world point (0.2, 0, 0.1)")
        plt.imshow(img_with_point)
        plt.axis('on')
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    env = gym.make(
        "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        robot_uids="panda", 
        obs_mode="rgb", # there is also "state_dict", "rgbd", ...
        control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
        # parallel_in_single_scene=True,
        render_mode="rgb_array", # rgb_array | human 
    )

    # world_point = np.array([0.2, 0.0, 0.1])
    # visualize_point_on_all_cameras(env, world_point)
import numpy as np
import cv2
import torch

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


def draw_points_on_tiled_image(tiled_image, world_points, camera_extrinsics, camera_intrinsics, 
                              marker_size=5, colors=None):
    """
    Draw 3D world points projected onto each camera view in a tiled image
    
    Args:
        tiled_image: The composite image from tile_images()
        world_points: Nx3 array of 3D world points to project
        camera_extrinsics: Dictionary mapping camera names to 4x4 extrinsic matrices
        camera_intrinsics: Dictionary mapping camera names to 3x3 intrinsic matrices
        marker_size: Size of the marker to draw
        colors: Optional list of colors for each point
    
    Returns:
        Image with projected points drawn on it
    """
    if world_points is None:
        return tiled_image
    # Make a copy of the tiled image
    img_points = tiled_image.copy()

    if isinstance(world_points, torch.Tensor):
        world_points = world_points.detach().cpu().numpy()

    if len(world_points.shape) == 1:
        world_points = world_points.reshape(1, 3)
    
    # If no colors provided, use default
    if colors is None:
        colors = [(0, 0, 255) for _ in range(len(world_points))]
    elif len(colors) != len(world_points):
        raise ValueError("Number of colors must match number of points")
    
    # Get camera names
    camera_names = list(camera_extrinsics.keys())

     # Calculate the position of each camera view in the tiled image
    camera_offsets = {}
    current_x_offset = 0
    
    for cam_name in camera_names:

        img_w = camera_intrinsics[cam_name][0, 2]*2
        img_h = camera_intrinsics[cam_name][1, 2]*2

        camera_offsets[cam_name] = (current_x_offset, 0)  # Assuming single row tiling
        current_x_offset += img_w
    
    # For each camera, project points and draw on the tiled image
    for camera_name in camera_names:
        extrinsic = camera_extrinsics[camera_name]
        intrinsic = camera_intrinsics[camera_name]
        x_offset, y_offset = camera_offsets[camera_name]
        
        # Project all points at once
        image_points = world_to_image(world_points, extrinsic, intrinsic)
        
        # Draw each valid point with its offset in the tiled image
        for i, (u, v) in enumerate(image_points):
            # Skip invalid points (behind camera or outside frame)
            if u < 0 or v < 0 or u >= img_w or v >= img_h:
                continue
                
            # Apply offset for this camera's position in the tiled image
            tiled_u = int(u + x_offset)
            tiled_v = int(v + y_offset)
            
            # Draw the point
            cv2.circle(img_points, (tiled_u, tiled_v), marker_size, colors[i], -1)
    
    return img_points
import open3d as o3d
import numpy as np
import json

def create_masked_point_cloud(color_image, depth_image, mask_image, intrinsic):
    # Convert images to numpy arrays
    color = np.asarray(color_image)
    depth = np.asarray(depth_image)
    mask = np.asarray(mask_image)

    # Ensure mask is binary
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask > 0

    # Apply mask to color and depth images
    masked_color = color * np.expand_dims(mask, axis=2)
    masked_depth = depth * mask

    # Create RGBD image from masked color and depth
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(masked_color.astype(np.uint8)),
        o3d.geometry.Image(masked_depth.astype(np.float32)),
        depth_scale=1000.0,  # Adjust based on your depth image scale
        depth_trunc=np.max(masked_depth)/1000.0,
        convert_rgb_to_intensity=False
    )

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Remove points with zero depth (background)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    non_zero_mask = points[:, 2] > 0
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[non_zero_mask])
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[non_zero_mask])

    return pcd

# Load color image
color_raw = o3d.io.read_image("./PlushSim/interaction_sequence/img/rgb_0000_000000.jpg")

# Load depth image
depth_raw = o3d.io.read_image("./PlushSim/interaction_sequence/img/depth_0000_000000.png")

# Load mask image
mask_raw = o3d.io.read_image("./PlushSim/interaction_sequence/img/seg_0000_000000.jpg")

# Load camera intrinsics
with open('./PlushSim/interaction_sequence/info/scene_meta.json') as f:
    data = json.load(f)
    cam_info = data['cam_info']['Camera']
    # Parse extrinsic matrix
    extrinsic = np.array(cam_info[0])
    
    # Parse intrinsic matrix
    intrinsic = np.array(cam_info[1])
    
    # Extract intrinsic parameters
    fx = fy = intrinsic[0, 0]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # Estimate image size (assuming square image)
    width = height = int(max(cx, cy) * 2)

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width, height, fx, fy, cx, cy)

# Create point cloud from masked images
pcd = create_masked_point_cloud(color_raw, depth_raw, mask_raw, intrinsic)

# The point cloud is by default in the camera coordinate system
# Flip it, so it's in the world coordinate system
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])

# Downsample the point cloud
voxel_size = 0.05
downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

o3d.visualization.draw_geometries([downpcd])

# Save the downsampled point cloud
o3d.io.write_point_cloud("demo.ply", pcd)
o3d.io.write_point_cloud("downsampled_demo.ply", downpcd)
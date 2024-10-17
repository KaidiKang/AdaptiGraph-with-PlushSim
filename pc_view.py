import open3d as o3d

# Read the PLY file
pcd = o3d.io.read_point_cloud("demo.ply")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
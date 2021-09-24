import open3d as o3d
import numpy as np
import os

# point = np.loadtxt('./example_image/pcd_scene_020.txt')
point = np.loadtxt('./example_image/pcd_crop_020.txt')
pcd = o3d.geometry.PointCloud()
point = point[:,:3]
pcd.points = o3d.utility.Vector3dVector(point)
# o3d.visualization.draw_geometries([pcd])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False

while True:
    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                            rotate_view)
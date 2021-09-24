import numpy as np
import open3d as o3d
import os
from multiprocessing import Process
import h5py
import json

# load data & environment generation setting from json file
f = open('data_generation_setting.json')
json_setting = json.load(f)

START_CYCLE_                = json_setting["data_generation"]["start_cycle"]# starting count of cycle to run
MAX_CYCLE_                  = json_setting["data_generation"]["end_cycle"]  # maximum count of cycle to run
MAX_DROP_                   = json_setting["data_generation"]["max_drop"]   # max number of item drop in each cycle 

DATASET_FOLDER_NAME_        = json_setting["folder_struct"]["dataset_folder_name"]
ITEM_NAME_                  = json_setting["folder_struct"]["item_name"] 
TRAIN_TEST_FOLDER_NAME_     = json_setting["folder_struct"]["train_test_folder_name"] 
SCENE_FOLDER_NAME_          = json_setting["folder_struct"]["scene_folder_name"] 
CROPPED_SCENE_FOLDER_NAME_  = json_setting["folder_struct"]["cropped_scene_folder_name"] 
H5_FOLDER_NAME_             = json_setting["folder_struct"]["h5_folder_name"] 

CURR_DIR_                   = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_PATH_        = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_) 
ITEM_FOLDER_PATH_           = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_NAME_) 
TRAIN_TEST_FOLDER_PATH_     = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_) 
SCENE_FOLDER_PATH_          = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,SCENE_FOLDER_NAME_) 
CROPPED_SCENE_FOLDER_PATH_  = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,CROPPED_SCENE_FOLDER_NAME_) 
H5_FOLDER_PATH_             = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,H5_FOLDER_NAME_) 

# create data folder 
if not os.path.exists(CROPPED_SCENE_FOLDER_PATH_):
  os.makedirs(CROPPED_SCENE_FOLDER_PATH_)

if not os.path.exists(H5_FOLDER_PATH_):
  os.makedirs(H5_FOLDER_PATH_)
  
def crop_pointcloud_from_camera_fov(data_points,cam_viewpoint=None):
    ### data_points format: x y z idx score
    point = data_points[:,:3]
    detail = data_points[:,3:].copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    # o3d.visualization.draw_geometries([pcd])
    
    if cam_viewpoint is None: # compute based on point cloud bounding box size
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        cam_viewpoint = [0, 0, diameter]
    radius = cam_viewpoint[2] * 1000
    
    _, pt_map = pcd.hidden_point_removal(cam_viewpoint, radius)
    visible_point = np.concatenate([point[pt_map,:],detail[pt_map,:]],axis=1)
    cropped_pcd = pcd.select_by_index(pt_map)
    # o3d.visualization.draw_geometries([cropped_pcd])

    return visible_point, cropped_pcd


def fpcc_save_h5(h5_filename, data,  data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename,'w')

    p_xyz = data[...,:6]
    gid = data[:,:,-2]
    center_score = data[:,:,-1]

    h5_fout.create_dataset(
            'data', data=p_xyz,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'center_score', data=center_score,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'gid', data=gid,
            compression='gzip', compression_opts=4,
            dtype=label_dtype)
    h5_fout.close()


def samples(data, sample_num_point, dim=None):
    if dim is None:
        dim = data.shape[-1]
    N = data.shape[0]
    order = np.arange(N)
    np.random.shuffle(order)
    data = data[order, :]
    batch_num = int(np.ceil(N / float(sample_num_point)))
    sample_datas = np.zeros((batch_num, sample_num_point, dim))

    for i in range(batch_num):
        beg_idx = i*sample_num_point
        end_idx = min((i+1)*sample_num_point, N)
        num = end_idx - beg_idx
        sample_datas[i,0:num,:] = data[beg_idx:end_idx, :]

        if num < sample_num_point:
            makeup_indices = np.random.choice(N, sample_num_point - num)
            sample_datas[i,num:,:] = data[makeup_indices, :]
    return sample_datas


def samples_plus_normalized(data_label, num_point=4096):
    data = data_label
    dim = data.shape[-1]
    # print('dim',dim)

    xyz_min = np.amin(data_label, axis=0)[0:3]
    data[:, 0:3] -= xyz_min

    max_x = max(data[:,0])
    max_y = max(data[:,1])
    max_z = max(data[:,2])

    data_batch = samples(data, num_point, dim=dim)
    # print('label_batch',label_batch,np.max(label_batch))
    new_data_batch = np.zeros((data_batch.shape[0], num_point, dim+3))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 3] = data_batch[b, :, 0]/max_x
        new_data_batch[b, :, 4] = data_batch[b, :, 1]/max_y
        new_data_batch[b, :, 5] = data_batch[b, :, 2]/max_z

    new_data_batch[:, :, 0:3] = data_batch[:,:,0:3]
    new_data_batch[:, :, 6:] = data_batch[:,:,3:]

    return new_data_batch



for cycle_idx in range(START_CYCLE_,MAX_CYCLE_+1):
    cycle_scene_path = os.path.join(SCENE_FOLDER_PATH_, 'cycle_%04d'%cycle_idx)
    cycle_cropped_scene_path = os.path.join(CROPPED_SCENE_FOLDER_PATH_, 'cycle_%04d'%cycle_idx)
    cycle_h5_path = os.path.join(H5_FOLDER_PATH_, 'cycle_%04d'%cycle_idx)

    if not os.path.exists(os.path.join(cycle_cropped_scene_path)):
        os.makedirs(os.path.join(cycle_cropped_scene_path))

    if not os.path.exists(os.path.join(cycle_h5_path)):
        os.makedirs(os.path.join(cycle_h5_path))

    for item_count in range(1,MAX_DROP_+1):
        filename = str('%03d'%item_count)+'.txt'
        h5_filename = os.path.join(cycle_h5_path, str('%03d'%item_count)+'.h5')
        cropped_pc_filename = os.path.join(cycle_cropped_scene_path, str('%03d'%item_count)+'.txt')

        point = np.loadtxt(os.path.join(cycle_scene_path,filename))
        pcd = o3d.geometry.PointCloud()
        visible_points, pcd = crop_pointcloud_from_camera_fov(point)
        # o3d.visualization.draw_geometries([pcd])
        data = samples_plus_normalized(visible_points, 4096)
        
        # save cropped scene point cloud data
        with open(cropped_pc_filename, 'w') as f:
            for i in range(visible_points.shape[0]):
                f.write('%.3f %.3f %.3f %d %.3f\n' % (visible_points[i][0], visible_points[i][1], visible_points[i][2], visible_points[i][3],visible_points[i][4]))
                
        # save data into H5 format 
        fpcc_save_h5(h5_filename, data)
        print('Saved h5 data : ' + str(h5_filename))


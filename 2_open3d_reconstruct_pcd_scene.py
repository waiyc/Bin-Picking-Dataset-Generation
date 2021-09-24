import glob
import open3d as o3d
import numpy as np
import os
import json

# load data & environment generation setting from json file
f = open('data_generation_setting.json')
json_setting = json.load(f)


START_CYCLE_                = json_setting["data_generation"]["start_cycle"]# starting count of cycle to run
MAX_CYCLE_                  = json_setting["data_generation"]["end_cycle"]  # maximum count of cycle to run
MAX_DROP_                   = json_setting["data_generation"]["max_drop"]   # max number of item drop in each cycle 

DATASET_FOLDER_NAME_        = json_setting["folder_struct"]["dataset_folder_name"]
MODEL_FOLDER_NAME_          = json_setting["folder_struct"]["model_folder_name"]
ITEM_NAME_                  = json_setting["folder_struct"]["item_name"] 
TRAIN_TEST_FOLDER_NAME_     = json_setting["folder_struct"]["train_test_folder_name"] 
GT_MATRIX_POSES_FOLDER_NAME_= json_setting["folder_struct"]["gt_matrix_poses_folder_name"]
SCENE_FOLDER_NAME_          = json_setting["folder_struct"]["scene_folder_name"] 

ITEM_MODEL_FILENAME_        = json_setting["model_param"]["model_filename"]  
ITEM_MAX_D_                 = json_setting["model_param"]["max_d"]            # diameter , NOTE: check this from object model?
ITEM_MAX_R_                 = ITEM_MAX_D_/2.0 # radius = d/2 
CAD_PCD_SIZE_               = 1024*10         # 1024 * N of points used for construct pcd
VOXEL_SIZE_                 = 0.001           # voxel size for point cloud data downsample 

CURR_DIR_                   = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_PATH_        = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_) 
ITEM_FOLDER_PATH_           = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_NAME_) 
TRAIN_TEST_FOLDER_PATH_     = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_) 
GT_MATRIX_FOLDER_PATH_      = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,GT_MATRIX_POSES_FOLDER_NAME_) 
SCENE_FOLDER_PATH_          = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,SCENE_FOLDER_NAME_) 
ITEM_MODEL_FILE_PATH_       = os.path.join(CURR_DIR_,MODEL_FOLDER_NAME_,ITEM_NAME_,ITEM_MODEL_FILENAME_)# ./model/IPAGearShaft/IPAGearShaft.stl'

# create /data/item/training folder 
if not os.path.exists(SCENE_FOLDER_PATH_):
  os.makedirs(SCENE_FOLDER_PATH_)

def get_center_score(model_pc, center, max_dis):
	distance = np.sqrt(np.sum((model_pc[:,:3]-center)**2,axis=1))
	distance_rate = distance/max_dis
	distance_rate = np.clip(distance_rate,0,1)
	score = np.clip(1-distance_rate**2,0,1)
	return score

# read in .stl n convert to .pcd  
mesh = o3d.io.read_triangle_mesh(ITEM_MODEL_FILE_PATH_)
pcd_cad = mesh.sample_points_uniformly(number_of_points=CAD_PCD_SIZE_)
# pcd_cad = mesh.sample_points_poisson_disk(number_of_points=CAD_PCD_SIZE_,init_factor=5)
pcd_cad.voxel_down_sample(VOXEL_SIZE_)
# o3d.visualization.draw_geometries([pcd_cad])
model_pc = np.asarray(pcd_cad.points)
ones = np.ones([model_pc.shape[0],1])
# ones = np.zeros([model_pc.shape[0],1])
model_pc = np.concatenate([model_pc,ones],axis=1)

for cycle_idx in range(START_CYCLE_,MAX_CYCLE_+1):
    cycle_gt_matrix_path = os.path.join(GT_MATRIX_FOLDER_PATH_, 'cycle_%04d'%cycle_idx)
    cycle_scene_path = os.path.join(SCENE_FOLDER_PATH_, 'cycle_%04d'%cycle_idx)

    if not os.path.exists(os.path.join(cycle_scene_path)):
        os.makedirs(os.path.join(cycle_scene_path))

    for item_count in range(1,MAX_DROP_+1):
        # read in matrix pose list
        scene = []
        scene_filename = str('%03d'%item_count)+'.txt'
        Trans = np.loadtxt(os.path.join(cycle_gt_matrix_path,scene_filename))
        # Trans = np.loadtxt('./data/IPAGearShaft/training/gt_matrix/cycle_0001/030.txt')
        Trans = Trans.reshape([-1,4,4])

        for index in range(Trans.shape[0]):
            M = Trans[index,:,:]
            points = model_pc.copy()
            part = np.matmul(M,points.transpose())
            part = part.T
            part = part[:,:3]

            center = M[0:3,-1]
            center_score = get_center_score(part[:,:3],center, max_dis=ITEM_MAX_R_)
            center_score = center_score.reshape([-1,1])

            instance_label = np.ones(part.shape[0])
            instance_label = instance_label * (index+1) # starting index is 1 
            instance_label = instance_label.reshape([-1,1])
            # print('instance_label',instance_label.shape)
            # print('center_score',center_score.shape)
            part = np.concatenate([part,instance_label,center_score],axis=-1)

            scene += [part]

        scene = np.concatenate(scene,axis=0)
        # print(scene.shape)
        scene_file = os.path.join(cycle_scene_path,scene_filename)
        with open(scene_file, 'w') as f:
            for i in range(scene.shape[0]):
                f.write('%.3f %.3f %.3f %d %.3f\n' % (scene[i][0], scene[i][1], scene[i][2], scene[i][3],scene[i][4]))
        print('Complete reconstruction and saved to : ' + scene_file)

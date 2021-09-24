import pybullet as p
import time 
import pybullet_data
import random
import os
import matplotlib.image as mp
import numpy as np
import json

# load data & environment generation setting from json file
f = open('data_generation_setting.json')
json_setting = json.load(f)

'''pybullet env parameters'''
useGUI = True
useRealTimeSimulation = 1 # 0 will freeze the simualtion?
TIMESTEP_ = 1. / 240. # Time in seconds.

if useGUI:
  p.connect(p.GUI)
else:
  p.connect(p.DIRECT)

''' virtual camera parameter in pybullet''' 
CAMERA_IMG_WIDTH_       = 512 #px
CAMERA_IMG_HEIGHT_      = 512 #px

CAMERA_FOV_             = 60 # describes how “wide” the camera’s visual field is
CAMERA_ASPECT_          = CAMERA_IMG_WIDTH_ / CAMERA_IMG_HEIGHT_ # describes the camera aspect ratio
CAMERA_NEAR_            = 0.02
CAMERA_FAR_             = 2.0                   # describe the minimum and maximum distance which the camera will render objects 
CAMERA_EYE_POSITION_    = [0, 0, 1.25]  # physical location of the camera in x, y, and z coordinates
CAMERA_TARGET_POSITION_ = [0, 0, 0]     # the point that we wish the camera to face. [0, 0, 0] is origin
CAMERA_UP_VECTOR_       = [0, 1, 0]     # describe the orientation of the camera

CAMERA_VIEW_MATRIX_ = p.computeViewMatrix(cameraEyePosition=CAMERA_EYE_POSITION_,
                                          cameraTargetPosition=CAMERA_TARGET_POSITION_,
                                          cameraUpVector=CAMERA_UP_VECTOR_)
CAMERA_PROJ_MATRIX_ = p.computeProjectionMatrixFOV(CAMERA_FOV_, CAMERA_ASPECT_, CAMERA_NEAR_, CAMERA_FAR_)


'''Box/Container parameters'''
BOX_MODEL_PATH_ = "model/tote_box/tote_box.urdf"
BOX_WIDTH_X_    = 0.6 #meters
BOX_WIDTH_Y_    = 0.4
BOX_SCALING_    = 2.0 #adjust scaling factor if box is too small 

''' Dropping parameters'''
ITEM_MODEL_PATH_  = "model/IPAGearShaft/IPAGearShaft.urdf"
DROP_X_MIN_       = -(BOX_WIDTH_X_ * BOX_SCALING_ * 0.6) / 2.0 # box width * box scaling* limit range scaling (to prevent drop at the edge of the box) / half
DROP_X_MAX_       =  (BOX_WIDTH_X_ * BOX_SCALING_ * 0.6) / 2.0
DROP_Y_MIN_       = -(BOX_WIDTH_Y_ * BOX_SCALING_ * 0.6) / 2.0
DROP_Y_MAX_       =  (BOX_WIDTH_Y_ * BOX_SCALING_ * 0.6) / 2.0
DROP_Z_MIN_       = 1.0
DROP_Z_MAX_       = 1.5

''' data collection cycle and drop setting '''
START_CYCLE_                  = json_setting["data_generation"]["start_cycle"]# starting count of cycle to run
MAX_CYCLE_                    = json_setting["data_generation"]["end_cycle"]  # maximum count of cycle to run
MAX_DROP_                     = json_setting["data_generation"]["max_drop"]   # max number of item drop in each cycle 

'''path for save img data'''
DATASET_FOLDER_NAME_          = json_setting["folder_struct"]["dataset_folder_name"]
ITEM_NAME_                    = json_setting["folder_struct"]["item_name"] 
TRAIN_TEST_FOLDER_NAME_       = json_setting["folder_struct"]["train_test_folder_name"] 
RGB_IMG_FOLDER_NAME_          = json_setting["folder_struct"]["rgb_img_folder_name"]
DEPTH_IMG_FOLDER_NAME_        = json_setting["folder_struct"]["depth_img_folder_name"]
SYN_SEG_IMG_FOLDER_NAME_      = json_setting["folder_struct"]["syn_seg_img_folder_name"]
GT_POSES_FOLDER_NAME_         = json_setting["folder_struct"]["gt_poses_folder_name"]
GT_MATRIX_POSES_FOLDER_NAME_  = json_setting["folder_struct"]["gt_matrix_poses_folder_name"]

CURR_DIR_                     = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_PATH_          = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_) 
ITEM_FOLDER_PATH_             = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_NAME_) 
TRAIN_TEST_FOLDER_PATH_       = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_) 
RGB_IMG_FOLDER_PATH_          = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,RGB_IMG_FOLDER_NAME_) 
DEPTH_IMG_FOLDER_PATH_        = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,DEPTH_IMG_FOLDER_NAME_) 
SEG_IMG_FOLDER_PATH_          = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,SYN_SEG_IMG_FOLDER_NAME_) 
GT_POSES_FOLDER_PATH_         = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,GT_POSES_FOLDER_NAME_) 
GT_MATRIX_POSES_FOLDER_PATH_  = os.path.join(CURR_DIR_,DATASET_FOLDER_NAME_,ITEM_FOLDER_PATH_,TRAIN_TEST_FOLDER_NAME_,GT_MATRIX_POSES_FOLDER_NAME_) 

# create /data/item/training folder 
if not os.path.exists(RGB_IMG_FOLDER_PATH_):
  os.makedirs(RGB_IMG_FOLDER_PATH_)
if not os.path.exists(DEPTH_IMG_FOLDER_PATH_):
  os.makedirs(DEPTH_IMG_FOLDER_PATH_)
if not os.path.exists(SEG_IMG_FOLDER_PATH_):
  os.makedirs(SEG_IMG_FOLDER_PATH_)
if not os.path.exists(GT_POSES_FOLDER_PATH_):
  os.makedirs(GT_POSES_FOLDER_PATH_)

def setup_env():
  p.resetSimulation()
  p.setAdditionalSearchPath(pybullet_data.getDataPath()) #default pybullet model library 
  p.setGravity(0, 0, -9.87)
  p.setRealTimeSimulation(useRealTimeSimulation)  
  p.resetDebugVisualizerCamera(cameraDistance=2.33,
                              cameraYaw= 0.0,
                              cameraPitch= -65.0,
                              cameraTargetPosition=[0.0, 0.0, -0.16])
  p.setPhysicsEngineParameter(numSolverIterations=30)
  p.setPhysicsEngineParameter(fixedTimeStep=TIMESTEP_)
  planeId = p.loadURDF("plane100.urdf")
  if (useRealTimeSimulation):
    p.setRealTimeSimulation(1)


for cycle_idx in range(START_CYCLE_,MAX_CYCLE_+1):
  #create sub cycle folders
  cycle_rgb_path       = os.path.join(RGB_IMG_FOLDER_PATH_,         'cycle_%04d'%cycle_idx)
  cycle_depth_path     = os.path.join(DEPTH_IMG_FOLDER_PATH_,       'cycle_%04d'%cycle_idx)
  cycle_seg_path       = os.path.join(SEG_IMG_FOLDER_PATH_,         'cycle_%04d'%cycle_idx)
  cycle_gt_path        = os.path.join(GT_POSES_FOLDER_PATH_,        'cycle_%04d'%cycle_idx)
  cycle_gt_matrix_path = os.path.join(GT_MATRIX_POSES_FOLDER_PATH_, 'cycle_%04d'%cycle_idx)

  if not os.path.exists(os.path.join(cycle_rgb_path)):
    os.makedirs(os.path.join(cycle_rgb_path))

  if not os.path.exists(os.path.join(cycle_depth_path)):
    os.makedirs(os.path.join(cycle_depth_path))

  if not os.path.exists(os.path.join(cycle_seg_path)):
    os.makedirs(os.path.join(cycle_seg_path))

  if not os.path.exists(os.path.join(cycle_gt_path)):
    os.makedirs(os.path.join(cycle_gt_path))

  if not os.path.exists(os.path.join(cycle_gt_matrix_path)):
    os.makedirs(os.path.join(cycle_gt_matrix_path))

  for item_count in range(1,MAX_DROP_+1):
    '''reset the environemnt'''
    setup_env() 
    
    '''place a box at the middle''' 
    boxStartPos = [0, 0, 0.01]
    boxStartOrientation = p.getQuaternionFromEuler([1.571, 0, 0])
    boxId = p.loadURDF(BOX_MODEL_PATH_, boxStartPos, boxStartOrientation,useFixedBase=1,globalScaling=BOX_SCALING_)
    boxPos, boxQuat = p.getBasePositionAndOrientation(boxId)
    time.sleep(0.1)
    
    '''start the dropping loop'''
    obj_id = []
    count = 0
    for count in range(1,item_count+1):
      pose = [random.uniform(DROP_X_MIN_,DROP_X_MAX_),
              random.uniform(DROP_Y_MIN_,DROP_Y_MAX_),
              random.uniform(DROP_Z_MIN_,DROP_Z_MAX_)]
      orientation = p.getQuaternionFromEuler( [random.uniform(0.01,3.0142),
                                               random.uniform(0.01,3.0142),
                                               random.uniform(0.01,3.0142)])
      obj_id.append(p.loadURDF(ITEM_MODEL_PATH_, pose,orientation))
      time.sleep(0.25)#to prevent all objects drop at the same time
      count += 1
      images = p.getCameraImage(CAMERA_IMG_WIDTH_,
                                CAMERA_IMG_HEIGHT_,
                                viewMatrix=CAMERA_VIEW_MATRIX_,
                                projectionMatrix=CAMERA_PROJ_MATRIX_,
                                lightDirection=[-0., -1., -1.],
                                lightColor=[1., 1., 1.],
                                lightDistance=2,
                                shadow=1,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
      if (useRealTimeSimulation):
        time.sleep(TIMESTEP_) 
      else:
        p.stepSimulation()

    print('End of cycle %04d_%03d'%(cycle_idx,item_count))
    print("Total item drop:" + str(len(obj_id)))
    
    '''give it some time (3sec) to let the physics settle down'''
    time_start = time.time()
    while time.time() < (time_start+3.0):
      images = p.getCameraImage(CAMERA_IMG_WIDTH_,
                                CAMERA_IMG_HEIGHT_,
                                viewMatrix=CAMERA_VIEW_MATRIX_,
                                projectionMatrix=CAMERA_PROJ_MATRIX_,
                                lightDirection=[-0., -1., -1.],
                                lightColor=[1., 1., 1.],
                                lightDistance=2,
                                shadow=1,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
      if (useRealTimeSimulation):
        time.sleep(TIMESTEP_) 
      else:
        p.stepSimulation()

    ''' end of this dropping cycle, start of data saving process '''
    ''' image convertion '''
    rgb_opengl = np.reshape(images[2], (CAMERA_IMG_HEIGHT_, CAMERA_IMG_WIDTH_, 4)) * 1. / 255.
    depth_buffer_opengl = np.reshape(images[3], [CAMERA_IMG_WIDTH_, CAMERA_IMG_HEIGHT_])
    depth_opengl = CAMERA_FAR_ * CAMERA_NEAR_ / (CAMERA_FAR_ - (CAMERA_FAR_ - CAMERA_NEAR_) * depth_buffer_opengl)
    seg_opengl = np.reshape(images[4], [CAMERA_IMG_WIDTH_, CAMERA_IMG_HEIGHT_]) * 1. / 255.
    
    ''' save rgb,depth,seg images '''
    mp.imsave(os.path.join(cycle_rgb_path ,   str('%03d_rgb.png'%item_count)), rgb_opengl)
    mp.imsave(os.path.join(cycle_depth_path , str('%03d_depth.png'%item_count)), depth_opengl)
    mp.imsave(os.path.join(cycle_seg_path ,   str('%03d_segmentation.png'%item_count)), seg_opengl)

    ''' save each cycle with different number of object poses (target object only, without bin pose) into .txt '''
    ''' format : x y z quat_x quat_y quat_z quat_w'''
    ''' format : matrix 4x4 '''

    gt_filename = str('%03d'%item_count)+'.txt'
    gt_poses_str = ''
    gt_matrix_poses_str = ''
    for idx in obj_id:
      boxPos, boxQuat = p.getBasePositionAndOrientation(idx)
      gt_poses_str = gt_poses_str + str(round(boxPos[0],5))  + ' ' + str(round(boxPos[1],5))  + ' ' + str(round(boxPos[2],5))  + ' ' + \
                                    str(round(boxQuat[0],5)) + ' ' + str(round(boxQuat[1],5)) + ' ' + str(round(boxQuat[2],5)) + ' ' + \
                                    str(round(boxQuat[3],5)) + '\n'
      
      #put into matrix form
      matrix = np.zeros((4,4))
      matrix[:3,:3] = np.array(p.getMatrixFromQuaternion(boxQuat)).reshape((3,3))
      matrix[:3,3] = boxPos
      matrix[3,:] = [0, 0, 0, 1]
      gt_matrix_poses_str += str(round(matrix[0][0],5))  + ' ' + str(round(matrix[0][1],5))  + ' ' + str(round(matrix[0][2],5))  + ' ' + str(round(matrix[0][3],5)) + ' \n' + \
                             str(round(matrix[1][0],5))  + ' ' + str(round(matrix[1][1],5))  + ' ' + str(round(matrix[1][2],5))  + ' ' + str(round(matrix[1][3],5)) + ' \n' + \
                             str(round(matrix[2][0],5))  + ' ' + str(round(matrix[2][1],5))  + ' ' + str(round(matrix[2][2],5))  + ' ' + str(round(matrix[2][3],5)) + ' \n' + \
                             str(round(matrix[3][0],5))  + ' ' + str(round(matrix[3][1],5))  + ' ' + str(round(matrix[3][2],5))  + ' ' + str(round(matrix[3][3],5)) + ' \n' 
      gt_matrix_poses_str += "\n"
                             
    # write xyzquat into txt
    f = open(os.path.join(cycle_gt_path,gt_filename),'w')
    f.write(gt_poses_str)
    f.close()
    
    # write matrix into txt
    f = open(os.path.join(cycle_gt_matrix_path,gt_filename),'w')
    f.write(gt_matrix_poses_str)
    f.close()
    



import pybullet as p
import pybullet_data as pd
import os

DATA_FOLDER = "model/IPARing"
INPUT_DATA_FILENAME = "IPARing.obj"
OUTPUT_DATA_FILENAME = "IPARing_collision.obj"
LOG_FILENAME = "IPARing_convert.log"

p.connect(p.DIRECT)
name_in = os.path.join(DATA_FOLDER,INPUT_DATA_FILENAME)
name_out = os.path.join(DATA_FOLDER,OUTPUT_DATA_FILENAME)
name_log = os.path.join(DATA_FOLDER,LOG_FILENAME)

p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=50000 )
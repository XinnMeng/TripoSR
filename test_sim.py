import numpy as np
import os
import xml.etree.ElementTree as ET
import yaml
import cv2
import gc
import copy
import random
random.seed(0)

from builtins import staticmethod
import ipdb
import pybullet as p
import pybullet_data
import numpy as np
import time
import os

from math import sqrt
import open3d as o3d


obj_urdf = "/home/xin/lib/TripoSR/runs/detect/exp/crops/laptop/mesh.urdf"
obj2_urdf = "/home/xin/lib/TripoSR/runs/detect/exp/crops/bottle/mesh.urdf"

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(5, 90, -50, [2, 2, 0])
p.setGravity(0, 0, -10)

plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, restitution=0.1)
p.changeDynamics(plane_id, -1, lateralFriction=1.0)

obj_pos = [0, 0, 1]
p.loadURDF(obj_urdf, obj_pos, useFixedBase=False)
obj_pos = [0.5, 0, 1]
p.loadURDF(obj2_urdf, obj_pos, useFixedBase=False)

for i in range(1500):
    p.stepSimulation()
    time.sleep(1/500)
import ipdb; ipdb.set_trace()
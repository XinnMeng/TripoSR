import torch
import os
import numpy as np
import rembg
import open3d as o3d
import random
import trimesh
import pybullet as p
import pybullet_data
import time

from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from preprocessing import *

device = "cuda:0"
if not torch.cuda.is_available():
    device = "cpu"

def get_tsr_mesh(data_path):
    rembg_session = rembg.new_session()
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
        ) 
    model.renderer.set_chunk_size(8192)
    model.to(device)

    name_list = os.listdir(data_path)

    for idx in range(len(name_list)):
        obj_data_path = os.path.join(data_path, name_list[idx])
        if not os.path.isdir(obj_data_path):
            continue
        print(name_list[idx])
        image = remove_background(Image.open(os.path.join(obj_data_path, "rgb_crop.png")), rembg_session)
        # image = Image.open(os.path.join(obj_data_path, "rgb_crop.png"))
        image = resize_foreground(image, 0.85)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        mesh_path = os.path.join(obj_data_path, "mesh.obj")
        urdf_path = os.path.join(obj_data_path, "mesh.urdf")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        meshes = model.extract_mesh(scene_codes, resolution=256)
        # meshes[0].show()
        meshes[0].export(mesh_path)

        # write urdf
        obj_vhacd_path = os.path.join(obj_data_path, "mesh_vhacd.obj")
        run_vhacd(vhacd_dir, mesh_path, obj_vhacd_path)
        com_pos = meshlabcompute(obj_vhacd_path, urdf_path, "mesh", 600, meshlabserver_exe, meshlab_mlx, meshlab_output_txt)


def load_scene(data_path, dataset_path=None):
    name_list = os.listdir(data_path)
    obj_init_pos_list = []
    obj_urdf_list = []
    for idx in range(len(name_list)):
        obj_data_path = os.path.join(data_path, name_list[idx])
        if not os.path.isdir(obj_data_path):
            continue
        print(name_list[idx])
        name = name_list[idx].split("_")[0]
        pc_path = os.path.join(obj_data_path, "pc.ply")
        # load pc
        pc = o3d.io.read_point_cloud(pc_path, remove_nan_points=True, remove_infinite_points=True)
        mean_pos, cov = pc.compute_mean_and_covariance()

        if dataset_path is None:
            pc_extents = pc.get_axis_aligned_bounding_box().get_extent()
            obj_model_path = os.path.join(obj_data_path, "mesh.obj")
            obj_urdf_path = os.path.join(obj_data_path, "mesh.urdf")
            obj_model = trimesh.load(obj_model_path)
            model_extents = np.array(obj_model.bounding_box.primitive.extents)
            scale = np.average(pc_extents/model_extents)
            obj_model.apply_scale(scale)
            # import ipdb; ipdb.set_trace()
        else:
            # load model
            obj_model_folder = os.path.join(dataset_path, name)
            if not os.path.exists(obj_model_folder):
                print("not in dataset: ", name)
                continue
            
            obj_model_list = os.listdir(obj_model_folder)
            # randomly select one
            obj_model_name = random.choice(obj_model_list)
            obj_urdf_path = os.path.join(obj_model_folder, obj_model_name, "origin", obj_model_name + "_xform.urdf")
            obj_model_path = os.path.join(obj_model_folder, obj_model_name, "origin", obj_model_name + "_xform.obj")
            obj_model = trimesh.load(obj_model_path)

        bbox_pos = np.array(obj_model.bounding_box.primitive.transform[:3, -1])
        extents = obj_model.bounding_box.primitive.extents
        obj_init_pos = np.copy(mean_pos)
        obj_init_pos[-1] = max(obj_init_pos[-1], extents[-1]/2 - bbox_pos[-1]) + 0.02
        obj_init_pos_list.append(obj_init_pos)
        obj_urdf_list.append(obj_urdf_path)
    
    # Load into pybullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(5, 90, -50, [2, 2, 0])
    p.setGravity(0, 0, -10)

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, restitution=0.1)
    p.changeDynamics(plane_id, -1, lateralFriction=1.0)

    for idx in range(len(obj_urdf_list)):
        p.loadURDF(obj_urdf_list[idx], obj_init_pos_list[idx], useFixedBase=False)

    # for i in range(1500):
    #     p.stepSimulation()
    #     time.sleep(1/500)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # Camera info
    camera_info_yaml = "/home/xin/Dropbox/goldilocks/azure_kinect_rgb_000839921812_1536P.yaml"
    camera_calib_file = "/home/xin/Dropbox/goldilocks/calib_ak_20240104/pose.txt"
    camera_info_url = "file:///home/xin/.ros/camera_info/azure_kinect_rgb_000839921812_1536P.yaml"

    # Save path
    data_path = "/home/xin/Dropbox/Reconstruction/real_test/test_1"
    dataset_path = "/home/xin/Dropbox/Reconstruction/dataset"
    
    # get_tsr_mesh(data_path)

    load_scene(data_path, dataset_path)
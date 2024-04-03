"""
ICRA 2020 Final Version
Preprocessing the raw obj file for imagination.

OBB transform the arbitrarily oriented object
VHACD for the OBB transformed object
Generate the URDF file for the OBB object

Author: Hongtao Wu, Deven Misra
Feb 20, 2020

Preprocessing the oriented raw off file for imagination.
Scale the object into a user defined dimension.
Randomly orient the objetc.

Author: Xin Meng
Feb 22, 2023
"""

import os
import time
import subprocess
import trimesh
import csv
import meshlabxml as mlx
import numpy as np
import re

import random
random.seed(0)

root_dir = "/home/xin/lib/TripoSR"

vhacd_dir = "/home/xin/lib/v-hacd/src/build/test"
meshlabserver_exe = "/home/xin/lib/meshlab-Meshlab-2020.03/distrib/meshlabserver"
meshlab_mlx = os.path.join(root_dir, "doc/TEMP3D_measure_geometry_scale.mlx")
meshlab_output_txt = os.path.join(root_dir, "doc/meshlab_output.txt")

def make_rigid_transformation(pos, rotm):
    """
    Rigid transformation from position and orientation.
    Args:
    - pos (3, numpy array): translation
    - rotm (3x3 numpy array): orientation in rotation matrix
    Returns:
    - homo_mat (4x4 numpy array): homogenenous transformation matrix
    """
    homo_mat = np.c_[rotm, np.reshape(pos, (3, 1))]
    homo_mat = np.r_[homo_mat, [[0, 0, 0, 1]]]

    return homo_mat



def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces are
        # all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', 'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', 'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, 'face should have %d vertices but as %d (%s)' % (face[0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces

    assert False, 'could not open %s' % file 

def read_obj(file):
    if not os.path.exists(file):
        return [], []
    
    vertices = []
    faces = []
    with open(file, 'r') as objFile:
        for idx, line in enumerate(objFile):
            list = line.split(" ")
            if len(list) < 2:
                continue
            if list[0] == "v":
                vertices.append([float(list[1]), float(list[2]), float(list[3])])
            if list[0] == "f":
                faces.append([0, int(list[1])-1, int(list[2])-1, int(list[3])-1])
    
    return vertices, faces

def write_to_obj(vertices, faces, file, scale, rotm):
    if os.path.exists(file):
        os.remove(file)
    with open(file, "w") as objFile:
        for vert in vertices:
            vert = np.matmul(rotm, vert)
            objFile.write("v ")
            objFile.write(str(vert[0]*scale))
            objFile.write(" ")
            objFile.write(str(vert[1]*scale))
            objFile.write(" ")
            objFile.write(str(vert[2]*scale))
            objFile.write("\n")
        objFile.write("s off\n")

        for face in faces:
            objFile.write("f ")
            objFile.write(str(face[1]+1))
            objFile.write(" ")
            objFile.write(str(face[2]+1))
            objFile.write(" ")
            objFile.write(str(face[3]+1))
            objFile.write("\n")
        objFile.close()


def scale_obj(obj_extents_origin, obj_scale_range):
    '''Scale the object according to the required range.
    Args:
        obj_extents_origin (3x array): extents of the object bounding box, from smallest to biggest.
        obj_scale_range ([[a1, a2], [b1, b2]]): a1-a2 is the range of the smallest obb extent after scalling, b1-b2 is the longest.
    Returns:
        scale (float)'''
    obj_extents_ratio = obj_extents_origin[-1] / obj_extents_origin[0]
    range_small = obj_scale_range[0][1] - obj_scale_range[0][0]
    range_big = obj_scale_range[1][1] - obj_scale_range[1][0]
    scale = 1
    # import ipdb; ipdb.set_trace()
    if obj_scale_range[0][0] * obj_extents_ratio >= obj_scale_range[1][1]:
        scale = obj_scale_range[0][0] / obj_extents_origin[0]

    elif obj_scale_range[0][0] * obj_extents_ratio > obj_scale_range[1][0] and obj_scale_range[0][0] * obj_extents_ratio < obj_scale_range[1][1]:
        if obj_scale_range[1][1] / obj_extents_ratio > obj_scale_range[0][0] and obj_scale_range[1][1] / obj_extents_ratio < obj_scale_range[0][1]:
            range_small = obj_scale_range[1][1] / obj_extents_ratio - obj_scale_range[0][0]
        obj_extents_small = obj_scale_range[0][0] + random.uniform(0, 1) * range_small
        scale = obj_extents_small / obj_extents_origin[0]
    
    elif obj_scale_range[1][1] / obj_extents_ratio > obj_scale_range[0][0] and obj_scale_range[1][1] / obj_extents_ratio < obj_scale_range[0][1]:
        range_small = obj_scale_range[0][1] - obj_scale_range[1][0] / obj_extents_ratio
        obj_extents_small = obj_scale_range[0][1] - random.uniform(0, 1) * range_small
        scale = obj_extents_small / obj_extents_origin[0]
    
    else:
        scale = obj_scale_range[1][0] / obj_extents_origin[1]
    
    return scale


def obb_transform(obj_path, obj_transform_path):
    """
    Apply OBB transformation on the object
    Save the OBB SE(3) transform in a csv file

    Args:
    -- obj_path: path to the object obj file
    -- obj_transform_path: path for saving the transformed obj file
    -- csv_path: path to the csv for saving the SE(3) of obb transofrm
    """
    origin_obj = trimesh.load(obj_path)
    origin_obj.apply_translation(-origin_obj.centroid)

    # Save the transformed model
    origin_obj.export(obj_transform_path, "obj")


def run_vhacd(vhacd_dir, input_file, output_file, log='log.txt', resolution=1000000, depth=20, concavity=0.0025,planeDownsampling=8, 
    convexhullDownsampling=8, alpha=0.05, beta=0.05, gamma=0.00125,pca=0, mode=0, maxNumVerticesPerCH=32, 
    minVolumePerCH=0.0001, convexhullApproximation=1, oclDeviceID=2):
    """
    The wrapper function to run the vhacd convex decomposition.

    #// --input camel.off --output camel_acd.wrl --log log.txt --resolution 1000000 --depth 20 --concavity 0.0025 --planeDownsampling 4 --convexhullDownsampling 4 --alpha 0.05 --beta 0.05 --gamma 0.00125 
    # --pca 0 --mode 0 --maxNumVerticesPerCH 256 --minVolumePerCH 0.0001 --convexhullApproximation 1 --oclDeviceID 2
    """

    vhacd_executable = os.path.join(vhacd_dir, 'testVHACD')
    if not os.path.isfile(vhacd_executable):
        print (vhacd_executable)
        raise ValueError('vhacd executable not found, have you compiled it?')

    cmd = "cd %s && %s --input %s --output %s --log %s --resolution %s --depth %s --concavity %s --planeDownsampling %s --convexhullDownsampling %s --alpha %s --beta %s --gamma %s \
        --pca %s --mode %s --maxNumVerticesPerCH %s --minVolumePerCH %s --convexhullApproximation %s --oclDeviceID %s" %(vhacd_dir, vhacd_executable, input_file, output_file, log, resolution,
        depth, concavity, planeDownsampling, convexhullDownsampling, alpha, beta, gamma, pca, mode, maxNumVerticesPerCH, minVolumePerCH, convexhullApproximation, oclDeviceID)

    print ("cmd:\n", cmd)

    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True)
    print ("started subprocess, waiting for V-HACD to finish")
    process.wait()
    elapsed = time.time() - start_time

    print ("V-HACD took %d seconds" %(elapsed))


def parse_float(string):
    """
    Parse the float within a string.
    
    Args:
    -- string: string input
    Retunrs:
    -- float_list: list of all floats in order
    """
    return np.array([float(i) for i in re.findall(r"[-+]?\d*\.\d+|\d+", string)])


def meshlabcompute(obj_path, urdf_path, obj_name, obj_density,
                   meshlabserver_exe, meshlab_mlx, meshlab_output_txt):
    """
    Compute the mass, mass center, and inertia for the object
    Install meshlab 2020.03 from source
    Args:
        obj_path: path to the obj file
        urdf_path: path to save the urdf file
        obj_density: the density of the object in kg/m^3.
        meshlabserver_exe: path to the meshlabserver executable
        meshlab_mlx: meshlab mlx file to compute the physics properties of the object
        meshlab_output_txt: output txt to save the meshlab result
    """
    if os.path.exists(meshlab_output_txt):
        os.remove(meshlab_output_txt)

    cmd = meshlabserver_exe + ' '
    cmd += '-i' + ' \"' + obj_path + '\" '
    cmd += '-s' + ' \"' + meshlab_mlx + '\" '
    cmd += '-l' + ' \"' + meshlab_output_txt + '\"'

    # print(cmd)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    # Parse the log file to get the volume, center of mass, and inertia
    with open(meshlab_output_txt, 'r') as file1:
        log = file1.readlines()

        properties = {}

        for idx, line in enumerate(reversed(log)):
            if ("Mesh Volume" in line) and (not "mass" in properties):
                volume = parse_float(line[len("Mesh Volume"):])[-1]
                properties[
                    'mass'] = obj_density * volume / 1000  # The mesh is scaled up by 10 in each dim

            if ("Center of Mass" in line) and (not "com" in properties):
                com = parse_float(line[len("Center of Mass"):])
                properties['com'] = com / 10

            if ("Inertia Tensor" in line) and (not "inertia" in properties):
                inertia_tensor = np.zeros((3, 3))
                inertia_tensor[0, :] = parse_float(log[-idx])
                inertia_tensor[1, :] = parse_float(log[-idx + 1])
                inertia_tensor[2, :] = parse_float(log[-idx + 2])
                properties['inertia'] = inertia_tensor * obj_density / 100000

            if ("mass" in properties) and ("com"
                                           in properties) and ("inertia"
                                                               in properties):
                break
    if 'inertia' not in properties:
        return []

    ixx = properties['inertia'][0][0]
    ixy = properties['inertia'][0][1]
    ixz = properties['inertia'][0][2]
    iyy = properties['inertia'][1][1]
    iyz = properties['inertia'][1][2]
    izz = properties['inertia'][2][2]

    # Write the urdf file
    urdf_file_name = urdf_path.split('/')[-1]
    with open(urdf_path, "w+") as f:
        f.write('<?xml version=\"1.0\" ?>\n')
        f.write('<robot name=\"' + urdf_file_name + '\">\n')
        f.write('  <link name=\"baseLink\">\n')
        f.write('    <contact>\n')
        f.write('      <lateral_friction value=\"1.0\"/>\n')
        f.write('      <inertia_scaling value=\"1.0\"/>\n')
        f.write('    </contact>\n')
        f.write('    <inertial>\n')
        f.write(
            '      <origin rpy=\"0 0 0\" xyz=\"%.6f %.6f %.6f\"/>\n' %
            (properties["com"][0], properties["com"][1], properties["com"][2]))
        f.write('      <mass value=\"%.6f\"/>\n' % properties["mass"], )
        f.write(
            '      <inertia ixx=\"%.6f\" ixy=\"%.6f\" ixz=\"%.6f\" iyy=\"%.6f\" iyz=\"%.6f\" izz=\"%.6f\"/>\n'
            % (ixx, ixy, ixz, iyy, iyz, izz))
        f.write('    </inertial>\n')
        f.write('    <visual>\n')
        f.write('      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n')
        f.write('      <geometry>\n')
        f.write('\t\t\t\t<mesh filename=\"' +
                obj_name + '.obj\" scale=\"1 1 1\"/>\n')
        f.write('      </geometry>\n')
        f.write('       <material name=\"white\">\n')
        f.write('        <color rgba=\"1 1 1 1\"/>\n')
        f.write('      </material>\n')
        f.write('    </visual>\n')
        f.write('    <collision>\n')
        f.write('      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n')
        f.write('      <geometry>\n')
        f.write('        <mesh filename=\"' +
                 obj_name + '.obj\" scale=\"1 1 1\"/>\n')
        f.write('      </geometry>\n')
        f.write('    </collision>\n')
        f.write('  </link>\n')
        f.write('</robot>\n')
    
    com_pos = np.array([properties["com"][0], properties["com"][1], properties["com"][2]])

    return com_pos


if __name__ == '__main__':
    start_time = time.time()
    human_height = 1.5
    input_root_dir = '/home/xin/Dropbox/bed_imagine/dataset/modelnet40_manually_aligned'
    output_root_dir = '/home/xin/Dropbox/Reconstruction/dataset'
    # input_root_dir = '/home/xin/Dropbox/LLM/laptop/'
    # output_root_dir = '/home/xin/Dropbox/LLM/test_data/'
    # obj_class_list = os.listdir(input_root_dir)
    # input_root_dir = '/home/xin/Dropbox/LLM'

    obj_class_list = ["monitor"]
    # obj_class_list = ['bed']
    # obj_class_list = ['cup', 'bookshelf', 'guitar', 'chair',
    #                   'door', 'desk', 'laptop', 'monitor', 'piano',
    #                   'sink', 'sofa', 'stairs', 'toilet', 'table',
    #                   'tent', 'tv_stand', 'vase', 'wardrobe', 'dresser',
    #                   'xbox', 'bowl']
    # obj_scale_range_list = [[[0.5, 3.0], [1.0, 3.0]]] # tv stand
    # obj_scale_range_list = [[[0.05, 0.5], [0.2, 0.7]]] #vase
    # obj_scale_range_list = [[[0.05, 0.1], [0.17, 0.2]]] #wine glass
    # obj_scale_range_list = [[[0.05, 0.15], [0.07, 0.2]]] #bed
    # obj_scale_range_list = [[[0.4, 1], [0.9, 1.7]]] #bathtub
    # obj_scale_range_list = [[[0.25, 0.4], [0.25, 0.4]]] #laptop
    # obj_scale_range_list = [[[0.05, 0.12], [0.2, 0.35]]] #bottle
    # obj_scale_range_list = [[[0.1, 0.5], [0.5, 0.7]], #monitor
    #                         [[0.02, 0.1], [0.3, 0.5]],
    #                         [[0.05, 0.12], [0.15, 0.3]],
    #                         [[0.2, 0.35], [0.25, 0.4]]]
    obj_scale_range_list = [[[0.01, 0.5], [0.3, 0.5]]] 
    # obj_scale_range_list = [
    #                         [[0.06, 0.2], [0.05, 0.3]], #cup
    #                         [[0.2, 0.5], [0.5, 2.0]], #bookshelf
    #                         [[0.1, 0.3], [0.5, 1.2]], #guitar
    #                         [[0.3, 1.0], [0.5, 1.5]], #chair
    #                         [[0.01, 0.1], [1.0, 2.5]], #door
    #                         [[0.4, 1], [0.5, 1.5]], #desk
    #                         [[0.01, 0.03], [0.25, 0.4]], #laptop
    #                         [[0.02, 0.12], [0.4, 1.0]], #monitor
    #                         [[0.5, 2], [1.0, 1.5]], #piano
    #                         [[0.2, 0.5], [0.4, 1.0]], #sink
    #                         [[0.4, 1.0], [0.5, 1.2]], #sofa
    #                         [[0.7, 1.5], [0.7, 1.5]], #stairs
    #                         [[0.4, 0.7], [0.7, 1.2]], #toilet
    #                         [[0.4, 1], [0.5, 1.5]], #table
    #                         [[1.2, 2.5], [2.0, 3.5]], #tent
    #                         [[0.5, 1.2], [1.0, 2.5]], #tv stand
    #                         [[0.2, 0.5], [0.2, 1.0]], #vase
    #                         [[0.4, 0.7], [1.2, 3.0]], #wardrob
    #                         [[0.4, 0.7], [1.2, 3.0]], #dresser
    #                         [[0.02, 0.1], [0.05, 0.2]], #xbox
    #                         [[0.05, 0.2], [0.05, 0.2]] #bowl
    # ]
    object_output_subdir = "20240126"
    wrong_data = []
    for class_idx in range(len(obj_class_list)):
        obj_class = obj_class_list[class_idx]
        print('-----------------------------')
        print(obj_class)
        class_input_root_dir = os.path.join(input_root_dir, obj_class, "train")
        class_output_root_dir = os.path.join(output_root_dir, obj_class)
        if not os.path.exists(class_output_root_dir):
            os.mkdir(class_output_root_dir)
        off_file_list = os.listdir(class_input_root_dir)
        # off_file_list = ["tv_stand_0009.off", "tv_stand_0011.off", "tv_stand_0040.off"]
        class_sucess_obj_name_list = []
        # class_input_root_dir = os.path.join(input_root_dir, obj_class)
        # obj_origin_name_list = os.listdir(class_input_root_dir)
        # for obj_idx in range(len(obj_origin_name_list)):
        #     obj_origin_name = obj_origin_name_list[obj_idx]
        #     obj_name = "wine_" + obj_origin_name
        #     obj_root_dir = os.path.join(class_input_root_dir, obj_origin_name)
            
        for obj_idx in range(len(off_file_list)):
            obj_name = off_file_list[obj_idx].split(".")[0]
            obj_root_dir = os.path.join(class_output_root_dir, obj_name)
            print(obj_root_dir)
            if not os.path.exists(obj_root_dir):
                os.mkdir(obj_root_dir)
            else:
                continue
            if obj_idx > 10:
                break
            obj_origin_dir = obj_root_dir + "/origin/"
            obj_xform_dir = os.path.join(obj_root_dir, object_output_subdir)
            print(obj_origin_dir)
            if not os.path.exists(obj_origin_dir):
                os.mkdir(obj_origin_dir)
            if not os.path.exists(obj_xform_dir):
                os.mkdir(obj_xform_dir)
            
            print(obj_name)
            input_file = class_input_root_dir + "/" + obj_name + '.off'
            output_obj_file = obj_origin_dir + obj_name + '_origin.obj'
            obj_file_scaled = obj_origin_dir + obj_name + '_scaled.obj'
            obj_urdf_scaled = obj_origin_dir + obj_name + '_xform.urdf'
            obb_transform_file = obj_origin_dir + obj_name + '_xform.obj'
            obj_vhacd = obj_origin_dir + obj_name + '_xform_vhacd.obj'
            csv_file = obj_origin_dir + obj_name + '_xform.csv'
            rotm_file = obj_origin_dir + obj_name + '_rotm.txt'
            transform_file = class_output_root_dir + obj_name + "/origin/" + obj_name + '_xform.txt'
            # Off to obj
            rotm = np.eye(3)
            vertices, faces = read_off(input_file)
            write_to_obj(vertices, faces, output_obj_file, 1, rotm)

            # output_obj_file = os.path.join(obj_origin_dir, obj_name + ".obj")
            # obj_file_scaled = os.path.join(obj_xform_dir, obj_name + "_scaled.obj")
            # rotm_file = os.path.join(obj_xform_dir, obj_name + '_random_rotm.txt')
            # vertices, faces = read_obj(output_obj_file)
            # Scale obj and random orn
            trimesh_obj = trimesh.load(output_obj_file)
            obj_scale = trimesh_obj.scale
            obj_extents = trimesh_obj.extents
            obj_extents_argsort = np.argsort(np.array(obj_extents))
            obj_extents_ordered = [obj_extents[i] for i in obj_extents_argsort]
            print('obj_extents: ', obj_extents_ordered)
            # import ipdb; ipdb.set_trace()

            # Scale the object to a common size
            scale = scale_obj(obj_extents_origin=obj_extents_ordered, obj_scale_range=obj_scale_range_list[class_idx])
            
            obj_extents_scaled = [obj_extents_ordered[i]*scale for i in range(len(obj_extents_ordered))]
            print('obj_extents_scaled: ', obj_extents_scaled)
            rotm = np.eye(3)
            write_to_obj(vertices, faces, obj_file_scaled, scale, rotm)
            obb_transform(obj_file_scaled, obb_transform_file)
            run_vhacd(vhacd_dir, obb_transform_file, obj_vhacd)
            com_pos = meshlabcompute(obj_vhacd, obj_urdf_scaled, obj_name + "_xform", 600, meshlabserver_exe, meshlab_mlx, meshlab_output_txt)

        # import ipdb; ipdb.set_trace()
    
    process_time = time.time() - start_time
    print("Total process time: ", process_time)
    import ipdb; ipdb.set_trace()
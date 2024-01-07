import os
import json
import numpy as np
import math
from pypcd import pypcd
import argparse
from tqdm import tqdm

def read_json(path_json):
    with open(path_json, 'r') as load_f:
        my_json = json.load(load_f)
    return my_json

def write_json(path_json, new_dict):
    with open(path_json, 'w') as f:
        json.dump(new_dict, f)

def get_calibs(calib_path):
    calib = read_json(calib_path)
    if 'transform' in calib.keys():
        calib = calib['transform']
    rotation = calib['rotation']
    translation = calib['translation']
    return rotation, translation

def rev_matrix(rotation, translation):
    rotation = np.matrix(rotation)
    rev_R = rotation.I
    rev_R = np.array(rev_R)
    rev_T = - np.dot(rev_R, translation)
    return rev_R, rev_T

def inverse_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R

def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation)

    return rotation, translation

def trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path,
                    veh_novatel2world_path, system_error_offset=None):
    inf_lidar2world_r, inf_lidar2world_t = get_calibs(inf_lidar2world_path)
    if system_error_offset is not None:
        inf_lidar2world_t[0][0] = inf_lidar2world_t[0][0] + system_error_offset['delta_x']
        inf_lidar2world_t[1][0] = inf_lidar2world_t[1][0] + system_error_offset['delta_y']

    veh_novatel2world_r, veh_novatel2world_t = get_calibs(veh_novatel2world_path)
    veh_world2novatel_r, veh_world2novatel_t = rev_matrix(veh_novatel2world_r, veh_novatel2world_t)
    inf_lidar2novatel_r, inf_lidar2novatel_t = mul_matrix(inf_lidar2world_r, inf_lidar2world_t,
                                                          veh_world2novatel_r, veh_world2novatel_t)

    veh_lidar2novatel_r, veh_lidar2novatel_t = get_calibs(veh_lidar2novatel_path)
    veh_novatel2lidar_r, veh_novatel2lidar_t = rev_matrix(veh_lidar2novatel_r, veh_lidar2novatel_t)
    inf_lidar2lidar_r,  inf_lidar2lidar_t = mul_matrix(inf_lidar2novatel_r, inf_lidar2novatel_t,
                                                       veh_novatel2lidar_r, veh_novatel2lidar_t)

    return inf_lidar2lidar_r, inf_lidar2lidar_t


def get_label_lidar_rotation(lidar_3d_8_points):
    """
    3D box in LiDAR coordinate system:

          4 -------- 5
         /|         /|
        7 -------- 6 .
        | |        | |
        . 0 -------- 1
        |/         |/
        3 -------- 2

        x: 3->0
        y: 1->0
        z: 0->4

        Args:
            lidar_3d_8_points: eight point list [[x,y,z],...]
        Returns:
            rotation_z: (-pi,pi) rad
    """
    x0, y0 = lidar_3d_8_points[0][0], lidar_3d_8_points[0][1]
    x3, y3 = lidar_3d_8_points[3][0], lidar_3d_8_points[3][1]
    dx, dy = x0 - x3, y0 - y3
    rotation_z = math.atan2(dy, dx)
    return rotation_z

def get_novatel2world(path_novatel2world):
    novatel2world = read_json(path_novatel2world)
    rotation = novatel2world['rotation']
    translation = novatel2world['translation']
    return rotation, translation


def get_lidar2novatel(path_lidar2novatel):
    lidar2novatel = read_json(path_lidar2novatel)
    rotation = lidar2novatel['transform']['rotation']
    translation = lidar2novatel['transform']['translation']
    return rotation, translation


def trans_point(input_point, translation, rotation):
    input_point = np.array(input_point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)
    output_point = output_point.reshape(1, 3).tolist()
    return output_point[0]

def trans_point_w2l(input_point, path_novatel2world, path_lidar2novatel):
    # world to novatel
    rotation, translation = get_novatel2world(path_novatel2world)
    new_rotation = inverse_matrix(rotation)
    new_translation = - np.dot(new_rotation, translation)
    point = trans_point(input_point, new_translation, new_rotation)

    # novatel to lidar
    rotation, translation = get_lidar2novatel(path_lidar2novatel)
    new_rotation = inverse_matrix(rotation)
    new_translation = - np.dot(new_rotation, translation)
    point = trans_point(point, new_translation, new_rotation)

    return point

def pcd2bin(pcd_file_path, bin_file_path):
    pc = pypcd.PointCloud.from_path(pcd_file_path)

    np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
    np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32) / 255

    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    points_32.tofile(bin_file_path)

def label_world2vlidar(sub_root, idx):
    path_input_label_file = os.path.join(sub_root, 'cooperative/label_world', idx + '.json')
    path_output_label_dir = os.path.join(sub_root, 'cooperative/label/lidar')
    if not os.path.exists(path_output_label_dir):
        os.makedirs(path_output_label_dir)
    path_output_label_file = os.path.join(path_output_label_dir, idx + '.json')

    input_label_data = read_json(path_input_label_file)
    lidar_3d_list = []
    path_novatel2world = os.path.join(sub_root, 'vehicle-side/calib/novatel_to_world', idx + '.json')
    path_lidar2novatel = os.path.join(sub_root, 'vehicle-side/calib/lidar_to_novatel', idx + '.json')
    for label_world in input_label_data:
        world_8_points_old = label_world["world_8_points"]
        world_8_points = []
        for point in world_8_points_old:
            point_new = trans_point_w2l(point, path_novatel2world, path_lidar2novatel)
            world_8_points.append(point_new)

        lidar_3d_data = {}
        lidar_3d_data['type'] = label_world['type']
        lidar_3d_data['occluded_state'] = label_world['occluded_state']
        lidar_3d_data["truncated_state"] = label_world['truncated_state']
        lidar_3d_data['2d_box'] = label_world['2d_box']
        lidar_3d_data["3d_dimensions"] = label_world['3d_dimensions']
        lidar_3d_data["3d_location"] = {}
        lidar_3d_data["3d_location"]["x"] = (world_8_points[0][0] + world_8_points[2][0]) / 2
        lidar_3d_data["3d_location"]["y"] = (world_8_points[0][1] + world_8_points[2][1]) / 2
        lidar_3d_data["3d_location"]["z"] = (world_8_points[0][2] + world_8_points[4][2]) / 2
        lidar_3d_data["rotation"] = get_label_lidar_rotation(world_8_points)
        lidar_3d_list.append(lidar_3d_data)
    write_json(path_output_label_file, lidar_3d_list)


parser = argparse.ArgumentParser("Preprocess the DAIR-V2X-C for FFNET.")
parser.add_argument(
    "--source-root", type=str, default="./data/dair-v2x/DAIR-V2X-Examples/cooperative-vehicle-infrastructure", 
                    help="Raw data root of DAIR-V2X-C."
)

if __name__ == "__main__":
    args = parser.parse_args()
    dair_v2x_c_root = args.source_root
    c_jsons_path = os.path.join(dair_v2x_c_root, 'cooperative/data_info.json')
    c_jsons = read_json(c_jsons_path)

    for c_json in tqdm(c_jsons):
        inf_idx = c_json['infrastructure_image_path'].split('/')[-1].replace('.jpg', '')
        inf_lidar2world_path = os.path.join(dair_v2x_c_root,
                                            'infrastructure-side/calib/virtuallidar_to_world/' + inf_idx + '.json')
        veh_idx = c_json['vehicle_image_path'].split('/')[-1].replace('.jpg', '')
        veh_lidar2novatel_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/lidar_to_novatel/' + veh_idx + '.json')
        veh_novatel2world_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/novatel_to_world/' + veh_idx + '.json')
        system_error_offset = c_json['system_error_offset']
        if system_error_offset is "":
            system_error_offset = None
        calib_lidar_i2v_r, calib_lidar_i2v_t = trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path,
                                          veh_novatel2world_path, system_error_offset)
        # print('calib_lidar_i2v: ', calib_lidar_i2v_r, calib_lidar_i2v_t)
        calib_lidar_i2v = {}
        calib_lidar_i2v['rotation'] = calib_lidar_i2v_r.tolist()
        calib_lidar_i2v['translation'] = calib_lidar_i2v_t.tolist()
        calib_lidar_i2v_save_dir = os.path.join(dair_v2x_c_root,
                                            'cooperative/calib/lidar_i2v')
        if not os.path.exists(calib_lidar_i2v_save_dir):
            os.makedirs(calib_lidar_i2v_save_dir)
        calib_lidar_i2v_save_path = os.path.join(calib_lidar_i2v_save_dir, veh_idx + '.json')
        write_json(calib_lidar_i2v_save_path, calib_lidar_i2v)

        # inf_pcd_path = os.path.join(dair_v2x_c_root,
        #                             c_json['infrastructure_pointcloud_path'])
        # inf_pcd = pypcd.PointCloud.from_path(inf_pcd_path)
        # for ii in range(len(inf_pcd.pc_data['x'])):
        #     np_x = inf_pcd.pc_data['x'][ii]
        #     np_y = inf_pcd.pc_data['y'][ii]
        #     np_z = inf_pcd.pc_data['z'][ii]
        #     inf_point = np.array([np_x, np_y, np_z])
        #     i2v_point = trans_point(inf_point, calib_lidar_i2v_r, calib_lidar_i2v_t)
        #     inf_pcd.pc_data['x'][ii] = i2v_point[0]
        #     inf_pcd.pc_data['y'][ii] = i2v_point[1]
        #     inf_pcd.pc_data['z'][ii] = i2v_point[2]
        #     inf_pcd.pc_data['intensity'][ii] = inf_pcd.pc_data['intensity'][ii] / 255
        #         i2v_pcd_save_path = os.path.join(dair_v2x_c_root, 'cooperative/velodyne_i2v/' + veh_idx + '.pcd')
        # pypcd.save_point_cloud(inf_pcd, i2v_pcd_save_path)
        # i2v_bin_save_path = os.path.join(dair_v2x_c_root, 'cooperative/velodyne_i2v/' + veh_idx + '.bin')
        # pcd2bin(i2v_pcd_save_path, i2v_bin_save_path)

        pcd_path = os.path.join(dair_v2x_c_root, 'infrastructure-side/velodyne/' + inf_idx + '.pcd')
        bin_save_path = os.path.join(dair_v2x_c_root, 'infrastructure-side/velodyne/' + inf_idx + '.bin')
        pcd2bin(pcd_path, bin_save_path)
        c_json['infrastructure_pointcloud_bin_path'] = c_json['infrastructure_pointcloud_path'].replace('.pcd', '.bin')
        c_json['infrastructure_idx'] = inf_idx

        pcd_path = os.path.join(dair_v2x_c_root, 'vehicle-side/velodyne/' + veh_idx + '.pcd')
        bin_save_path = os.path.join(dair_v2x_c_root, 'vehicle-side/velodyne/' + veh_idx + '.bin')
        pcd2bin(pcd_path, bin_save_path)
        c_json['vehicle_pointcloud_bin_path'] = c_json['vehicle_pointcloud_path'].replace('.pcd', '.bin')
        c_json['vehicle_idx'] = veh_idx

        c_json['calib_v_lidar2cam_path'] = os.path.join('vehicle-side/calib/lidar_to_camera', veh_idx + '.json')
        c_json['calib_v_cam_intrinsic_path'] = os.path.join('vehicle-side/calib/camera_intrinsic/', veh_idx + '.json')
        c_json['calib_lidar_i2v_path'] = os.path.join('cooperative/calib/lidar_i2v', veh_idx + '.json')

        label_world2vlidar(dair_v2x_c_root, veh_idx)
        c_json['cooperative_label_w2v_path'] = os.path.join('cooperative/label/lidar', veh_idx + '.json')
        c_json['cooperative_label_path'] = os.path.join('cooperative/label_world', veh_idx + '.json')

        c_json['image'] = {"image_shape": [1080, 1920]}

    c_jsons_write_path = os.path.join(dair_v2x_c_root, 'cooperative/data_info_new.json')
    write_json(c_jsons_write_path, c_jsons)

    # Complementary process:  missing infrastructure point clouds
    c_jsons_path = os.path.join(dair_v2x_c_root, 'infrastructure-side/data_info.json')
    c_jsons = read_json(c_jsons_path)
    for c_json in c_jsons:
        inf_idx = c_json['pointcloud_path'].split('/')[-1].replace('.pcd', '')
        pcd_path = os.path.join(dair_v2x_c_root, 'infrastructure-side/velodyne/' + inf_idx + '.pcd')
        bin_save_path = os.path.join(dair_v2x_c_root, 'infrastructure-side/velodyne/' + inf_idx + '.bin')
        pcd2bin(pcd_path, bin_save_path)
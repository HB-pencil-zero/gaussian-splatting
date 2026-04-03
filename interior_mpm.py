import os
import shutil
import json
from PIL import Image
import numpy as np


root_path = '/root/autodl-tmp/PhysGaussian/gaussian-splatting/data/ICCV_2025'
save_root_path = '/root/autodl-tmp/PhysGaussian/gaussian-splatting/data/process_data'
os.makedirs(save_root_path, exist_ok=True)


scene_list = os.listdir(root_path)
# scene_list = ['apple']
for scene_name in scene_list:

    scene_path = os.path.join(root_path, scene_name)

    save_scene_path = os.path.join(save_root_path, scene_name)
    os.makedirs(save_scene_path, exist_ok=True)

    save_image_root_path = os.path.join(save_scene_path, 'images')
    os.makedirs(save_image_root_path, exist_ok=True)

    save_dict = {}

    # camera intrinsics
    intrinsics_file_path = os.path.join(scene_path, 'intrin.json')
    with open(intrinsics_file_path, 'r') as f:
        intrinsics_dict = json.load(f)

    K = np.array(intrinsics_dict['K'])
    fl_x = K[0, 0]
    fl_y = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    W, H = 512, 512
    save_dict['fl_x'] = fl_x
    save_dict['fl_y'] = fl_y
    save_dict['cx'] = cx
    save_dict['cy'] = cy
    save_dict['w'] = W
    save_dict['h'] = H
    save_dict['aabb_scale'] = 1.0
    save_dict['frames'] = []
    save_dict["camera_angle_x"] = 0.6911112070083618

    file_list = os.listdir(scene_path)
    npy_file_list = [file for file in file_list if file.endswith('.npy')]
    view_id_list = [int(file.split('.')[0]) for file in npy_file_list]
    view_id_list.sort()

    for view_id in view_id_list:

        view_frame_dict = {}

        view_img_path = os.path.join(scene_path, f'{view_id:03d}.png')

        img_save_path = os.path.join(save_image_root_path, f'{view_id:06d}.png')
        img_map = Image.open(view_img_path).convert('RGB')
        img_map.save(img_save_path)


        view_frame_dict['file_path'] = img_save_path.split('.')[0]

        pose_file_path = os.path.join(scene_path, f'{view_id:03d}.npy')
        pose = np.load(pose_file_path)
        pose = np.linalg.inv(pose)
        view_frame_dict['transform_matrix'] = pose.tolist()

        save_dict['frames'].append(view_frame_dict)

    save_json_path = os.path.join(save_scene_path, 'transforms_train.json')
    with open(save_json_path, 'w') as f:
        json.dump(save_dict, f)
    save_json_path = os.path.join(save_scene_path, 'transforms_test.json')
    with open(save_json_path, 'w') as f:
        json.dump(save_dict, f)


    print(f'{scene_name} done!')


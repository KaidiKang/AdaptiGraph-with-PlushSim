import numpy as np
import glob
import time
import os
import re
import json
import cv2

from pointcloud import create_masked_point_cloud, load_camera_intrinsics
from utils import convert_ndarray, save_dic_to_h5, save_dic_to_pkl

def load_interaction(path):
    files = glob.glob(os.path.join(path, "interaction_info_*.npz"))
    required_keys = ["start_frames", "release_frames", "static_frames", "grasp_points", "target_points", "grasp_pixels"]

    data_set = {}
    for file in files:
        data = np.load(file)
        
        assert all(key in data for key in required_keys), "Missing keys in interaction data"

        data_dict = {key: data[key].tolist() for key in data}
        sequence = re.search(r'\d{4}', file).group()
        data_set[sequence] = data_dict
    
    return data_set


def load_image(image_path):
    images = {}

    # Add image to appropriate group
    for image_name in os.listdir(image_path):
        _, sequence, frame = image_name.split('.')[0].split('_')
        if sequence not in images:
            images[sequence] = {"frames": []}
        if frame not in images[sequence]["frames"]:
            images[sequence]["frames"].append(frame)

    return images


def get_rgb_values(image_path, sequence, frame):
    image = os.path.join(image_path, "rgb_" + sequence + "_" + frame + ".jpg")
    img = cv2.imread(image)
    return img

def load_eef(geom_path):
    data = np.load(geom_path)
    eef_pos = data["eef_pos"]
    
    assert len(eef_pos) == 7, "eef_pos must have length 7"

    if np.array_equal(eef_pos, [-1, -1, -1, -1, -1, -1, -1]):
        return None
    
    pos = eef_pos[:3]
    orient = eef_pos[3:]
    
    output = np.concatenate((pos, pos, orient, orient))
    
    return output 


def group_and_sort(image_path, info_path):
    # Load the images
    images = load_image(image_path)

    # Sort the frames
    for sequence in images:
        images[sequence]["frames"] = sorted(images[sequence]["frames"])

    # Sort the sequences
    images = {k: v for k, v in sorted(images.items(), key=lambda item: int(item[0]))}

    # Load the interaction data
    interaction = load_interaction(info_path)
    assert len(interaction) == len(images), "Number of samples do not match"

    # Combine the data
    for sequence in images:
        images[sequence]["interaction"] = interaction[sequence]

    return images


def get_pointcloud(sequence, frame, image_path, cam_path):
    image = os.path.join(image_path, "rgb_" + sequence + "_" + frame + ".jpg")
    depth = os.path.join(image_path, "depth_" + sequence + "_" + frame + ".png")
    mask = os.path.join(image_path, "seg_" + sequence + "_" + frame + ".jpg")

    intrinsic, extrinsic = load_camera_intrinsics(cam_path)
    pcd = create_masked_point_cloud(image, depth, mask, intrinsic, extrinsic)
    return pcd


def preprocess (config, phase = "val"):
    time_start = time.time()
    print ("Preprocessing starts")

    image_path = config.get("image_path", "PlushSim/interaction_sequence/img")
    info_path = config.get("info_path", "PlushSim/interaction_sequence/info")
    geom_path = config.get("geom_path", "PlushSim/interaction_sequence/geom")
    cam_path = config.get("cam_path", "PlushSim/interaction_sequence/info/scene_meta.json")
    max_points = config.get("max_points", 2000)

    if not os.path.exists(image_path) or not os.path.exists(info_path):
        raise FileNotFoundError("Image or info directory does not exist")


    sequence_groups = group_and_sort(image_path, info_path)

    # Process each sequence
    for sequence in sequence_groups:
        print (f"Processing sequence {int(sequence)+1} of {len(sequence_groups)}")
        os.makedirs(f"initialized/{int(sequence):06d}/", exist_ok=True)

        info = sequence_groups[sequence]["interaction"]

        lift_pc, lift_eef, lift_rgb = [], [], []
        interaction_index = 0

        # Iterate through each frame
        for frame in sequence_groups[sequence]["frames"]:
            if interaction_index >= len(info["start_frames"]):
                break
            
            if int(frame) in range (info["start_frames"][interaction_index], info["static_frames"][interaction_index]+1):
                # Get eef_positions
                eef_pos = load_eef(os.path.join(geom_path, f"{sequence}_{frame}.npz"))
                if eef_pos is not None:
                    lift_eef.append(eef_pos)
                    
                    # Get the point cloud
                    pcd = get_pointcloud(sequence, frame, image_path, cam_path)
                    pcd = pcd.farthest_point_down_sample(max_points)
                    lift_pc.append(np.asarray(pcd.points))

                    if phase == "val":
                        # Get the rgb values
                        img = get_rgb_values(image_path, sequence, frame)
                        lift_rgb.append(img)
                        assert len(lift_pc) == len(lift_rgb), "Length of point cloud and rgb values do not match"

                assert len(lift_pc) == len(lift_eef), "Length of point cloud and eef positions do not match"

            else:
                output = {
                    "action": [],
                    "eef_states": lift_eef, # Lift phase only
                    "info": {
                        "n_cams": 1,
                        "n_particles": max_points,
                        "timestamp": len(lift_pc),
                    },
                    "observations": {
                        "color": {
                            "cam_0": lift_rgb
                        },
                        "depth": {
                            "cam_0": [
                                0.0
                            ]
                        }
                    },
                    "positions": convert_ndarray(lift_pc), # Lift phase only
                    # Additional information
                    # "grasp_point": info["grasp_points"][interaction_index],
                    # "release_point": info["target_points"][interaction_index],
                    # "grasp_pixel": info["grasp_pixels"][interaction_index]
                }

                save_dic_to_h5(output, f"initialized/{int(sequence):06d}/{interaction_index:02d}.h5")
                interaction_index += 1
                lift_pc, lift_eef, lift_rgb= [], [], []

                if int(frame) in range (info["start_frames"][interaction_index], info["static_frames"][interaction_index]+1):
                    # Get eef_positions
                    eef_pos = load_eef(os.path.join(geom_path, f"{sequence}_{frame}.npz"))
                    if eef_pos is not None:
                        lift_eef.append(eef_pos)
                        
                        # Get the point cloud
                        pcd = get_pointcloud(sequence, frame, image_path, cam_path)
                        pcd = pcd.farthest_point_down_sample(max_points)
                        lift_pc.append(np.asarray(pcd.points))

                        if phase == "val":
                            # Get the rgb values
                            img = get_rgb_values(image_path, sequence, frame)
                            lift_rgb.append(img)
                            assert len(lift_pc) == len(lift_rgb), "Length of point cloud and rgb values do not match"

                    assert len(lift_pc) == len(lift_eef), "Length of point cloud and eef positions do not match"

        # Save the last interaction
        output = {
                "action": [],
                "eef_states": lift_eef, # Lift phase only
                "info": {
                    "n_cams": 1,
                    "n_particles": max_points,
                    "timestamp": len(lift_pc),
                },
                "observations": {
                    "color": {
                        "cam_0": lift_rgb
                    },
                    "depth": {
                        "cam_0": [
                            0.0
                        ]
                    }
                },
                "positions": convert_ndarray(lift_pc), # Lift phase only
                # Additional information
                # "grasp_point": info["grasp_points"][interaction_index],
                # "release_point": info["target_points"][interaction_index],
                # "grasp_pixel": info["grasp_pixels"][interaction_index]
            }

        save_dic_to_h5(output, f"initialized/{int(sequence):06d}/{interaction_index:02d}.h5")


        # Save the property parameters
        property_params = {
            "density": 0.009999999776482582, 
            "elasticity_damping": 0.10000000149011612, 
            "dynamic_friction": 0.25, 
            "youngs_modulus": 10000000.0, 
            "poissons_ratio": 0.30000001192092896, 
            "damping_scale": 1.0
        }
        save_dic_to_pkl(property_params, f"initialized/{int(sequence):06d}/property_params.pkl")
        
        print (f"Sequence {sequence} completed")

    # Save camera info
    with open(cam_path) as f:
        data = json.load(f)
        cam_info = data['cam_info']['Camera']
        extrinsic = np.array(cam_info[0])
        intrinsic = np.array(cam_info[1])
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        intrinsic = np.array([fx, fy, cx, cy]).reshape(1, 4)
        extrinsic = extrinsic.reshape(1, 4, 4)
    os.makedirs("initialized/cameras/", exist_ok=True)
    np.save("initialized/cameras/extrinsic.npy", extrinsic)
    np.save("initialized/cameras/intrinsic.npy", intrinsic)

    time_end = time.time()
    print (f"Preprocessing completed in {time_end - time_start} seconds")
        

if __name__ == "__main__":
    # If phase is train, rgb info is omitted
    preprocess({}, phase = "val")
import numpy as np
import glob
import argparse
import time
import os
import re

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


def group_and_sort(image_path, info_path):
    # Load the images
    images = load_image(image_path)

    # Sort the frames
    for sequence in images:
        images[sequence]["frames"] = sorted(images[sequence]["frames"])

    # Load the interaction data
    interaction = load_interaction(info_path)
    assert len(interaction) == len(images), "Number of samples do not match"

    # Combine the data
    for sequence in images:
        images[sequence]["interaction"] = interaction[sequence]

    return images


def get_pointcloud(sequence, frame):
    image_path = "PlushSim/interaction_sequence/img"
    camera_info = "PlushSim/interaction_sequence/info/scene_meta.json"

    image = os.path.join(image_path, "rgb_" + sequence + "_" + frame + ".jpg")
    depth = os.path.join(image_path, "depth_" + sequence + "_" + frame + ".png")
    mask = os.path.join(image_path, "seg_" + sequence + "_" + frame + ".jpg")

    pcd = create_masked_point_cloud(image, depth, mask, load_camera_intrinsics(camera_info))
    return pcd


def preprocess (config):
    time_start = time.time()
    print ("Preprocessing starts")

    image_path = config.get("image_path", "PlushSim/interaction_sequence/img")
    info_path = config.get("info_path", "PlushSim/interaction_sequence/info")
    max_points = config.get("max_points", 3)

    if not os.path.exists(image_path) or not os.path.exists(info_path):
        raise FileNotFoundError("Image or info directory does not exist")


    sequence_groups = group_and_sort(image_path, info_path)

    # Process each sequence
    for sequence in sequence_groups:
        print (f"Processing sequence {int(sequence)} of {len(sequence_groups)}")
        os.makedirs(f"initialized/{int(sequence):06d}/", exist_ok=True)

        info = sequence_groups[sequence]["interaction"]

        lift_pc, release_pc = [], []
        interaction_index = 0

        # Iterate through each frame
        for frame in sequence_groups[sequence]["frames"]:
            if interaction_index >= len(info["start_frames"]):
                break

            if int(frame) in range(info["start_frames"][interaction_index], info["release_frames"][interaction_index]+1):
                pcd = get_pointcloud(sequence, frame)
                pcd = pcd.farthest_point_down_sample(max_points)
                lift_pc.append(np.asarray(pcd.points))

            elif int(frame) in range(info["release_frames"][interaction_index], info["static_frames"][interaction_index]+1):
                pcd = get_pointcloud(sequence, frame)
                pcd = pcd.farthest_point_down_sample(max_points)
                release_pc.append(np.asarray(pcd.points))

                if int(frame) == info["static_frames"][interaction_index]:
                    output = {
                        "action": [],
                        "eef_states": [],
                        "info": {
                            "n_cams": 1,
                            "n_particles": max_points,
                            "timestamp": len(lift_pc),
                        },
                        "observations": {
                            "color": {
                                "cam_0": [
                                    0.0
                                ]
                            },
                            "depth": {
                                "cam_0": [
                                    0.0
                                ]
                            }
                        },
                        "positions": convert_ndarray(lift_pc), # Lift phase only
                        # Additional information
                        # "release": convert_ndarray(release_pc),
                        # "grasp_point": info["grasp_points"][interaction_index],
                        # "release_point": info["target_points"][interaction_index],
                        # "grasp_pixel": info["grasp_pixels"][interaction_index]
                    }

                    save_dic_to_h5(output, f"initialized/{int(sequence):06d}/{interaction_index:02d}.h5")
                    interaction_index += 1
                    lift_pc, release_pc = [], []

        # Save the property parameters
        property_params = {
            # CHANGE THIS 
            "particle_radius": 0.03,
            "num_particles": 2160,
            "length": 2.7744067519636624,
            "thickness": 3.0,
            "dynamic_friction": 0.3,
            "cluster_spacing": 6.860757465489678,
            "global_stiffness": 0.00018607574654896778,
            "stiffness": 0.7151893663724195
        }
        save_dic_to_pkl(property_params, f"initialized/{int(sequence):06d}/property_params.pkl")
        
        print (f"Sequence {sequence} completed")

    time_end = time.time()
    print (f"Preprocessing completed in {time_end - time_start} seconds")
        


if __name__ == "__main__":
    # dic = group_and_sort("PlushSim/interaction_sequence/img", "PlushSim/interaction_sequence/info")
    # with open("test.json", "w") as f:
    #     json.dump(dic, f, indent=4)
    preprocess({})
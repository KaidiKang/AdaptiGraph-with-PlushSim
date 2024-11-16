import h5py
import json
import pickle
import numpy as np

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    else:
        return obj


# Recursively convert the HDF5 group or dataset to a Python dictionary
def h5_to_dict(obj):
    result = {}
    
    if isinstance(obj, h5py.Dataset):
        # For scalar datasets, convert the value
        if obj.shape == ():  # Scalar
            result = obj[()]
            # Convert NumPy types to native Python types
            if isinstance(result, np.generic):  # Catch any NumPy scalar types
                result = result.item()
        else:
            # Convert datasets to list and handle NumPy types
            result = obj[:].tolist()  # Convert NumPy array to list
    
    elif isinstance(obj, h5py.Group):
        # Recursively add group members
        for key, item in obj.items():
            result[key] = h5_to_dict(item)

    return result


def check_h5_compression(h5_path):
    with h5py.File(h5_path, 'r') as file:
        for key in file.keys():
            item = file[key]
            if isinstance(item, h5py.Dataset):
                print(f"Dataset: {key}")
                print(f"Compression: {item.compression}")
                print(f"Compression Options: {item.compression_opts}")
                print(f"Shape: {item.shape}")
                print(f"Type: {item.dtype}")
                print(f"Size: {item.size}")
                print(f"Chunk Size: {item.chunks}")
            elif isinstance(item, h5py.Group):
                print(f"Group: {key}")


# Save a nested dictionary to a single h5 file
def save_dic_to_h5(dictionary, filename):
    with h5py.File(filename, 'w') as f:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Create a group for nested dictionary
                group = f.create_group(key)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        # Handle nested dictionary
                        subgroup = group.create_group(subkey)
                        for k, v in subvalue.items():
                            if isinstance(v, list):
                                # Convert empty or numeric lists to numpy arrays
                                if len(v) == 0:
                                    subgroup.create_dataset(k, data=np.array(v, dtype=np.float32))
                                else:
                                    subgroup.create_dataset(k, data=np.array(v))
                            else:
                                subgroup.create_dataset(k, data=v)
                    elif isinstance(subvalue, list):
                        # Convert empty or numeric lists to numpy arrays
                        if len(subvalue) == 0:
                            group.create_dataset(subkey, data=np.array(subvalue, dtype=np.float32))
                        else:
                            group.create_dataset(subkey, data=np.array(subvalue))
                    else:
                        # Handle numeric or string values
                        if isinstance(subvalue, (int, float)):
                            group.create_dataset(subkey, data=subvalue)
                        elif isinstance(subvalue, str):
                            group.create_dataset(subkey, data=np.string_(subvalue))
            elif isinstance(value, list):
                # Convert empty or numeric lists to numpy arrays
                if len(value) == 0:
                    f.create_dataset(key, data=np.array(value, dtype=np.float32))
                else:
                    f.create_dataset(key, data=np.array(value))
            elif isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            else:
                # Handle numeric or string values
                if isinstance(value, (int, float)):
                    f.create_dataset(key, data=value)
                elif isinstance(value, str):
                    f.create_dataset(key, data=np.string_(value))


def h5_to_json(h5_path, json_path):
    # Open the HDF5 file
    with h5py.File(h5_path, 'r') as file:
        # Convert the entire HDF5 structure into a dictionary
        hdf5_dict = h5_to_dict(file)

    with open(json_path, 'w') as json_file:
        json.dump(hdf5_dict, json_file, indent=4)


def pkl_to_json(pkl_path, json_path):
    # Load the property parameters from the pickle file
    with open(pkl_path, 'rb') as pkl_file:
        property_params = pickle.load(pkl_file)

    with open(json_path, 'w') as json_file:
        json.dump(property_params, json_file, indent=4)


def save_dic_to_pkl(dictionary, filename):
    with open (filename, 'wb') as f:
        pickle.dump(dictionary, f)


def npz_to_json(npz_path, json_path):
    # Load the interaction data from the NumPy file
    data = np.load(npz_path)
    data_dic = {key: data[key].tolist() for key in data}

    with open(json_path, 'w') as json_file:
        json.dump(data_dic, json_file, indent=4)


def npy_reader(npy_path):
    return np.load(npy_path)


if __name__ == '__main__':
    h5_to_json("initialized/000000/01.h5", "example/h5_plush.json")
    # pkl_to_json("rope/000000/property_params.pkl", "example/rope_property.json")
    # check_h5_compression("rope/000000/01.h5")
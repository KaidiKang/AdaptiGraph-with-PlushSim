import h5py
import json
import pickle
import numpy as np

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

# Open the HDF5 file
with h5py.File('cloth/000000/00.h5', 'r') as file:
    # Convert the entire HDF5 structure into a dictionary
    hdf5_dict = h5_to_dict(file)

with open('h5_file.json', 'w') as json_file:
    json.dump(hdf5_dict, json_file, indent=4)


# Load the property parameters from the pickle file
with open('cloth/000000/property_params.pkl', 'rb') as pkl_file:
    property_params = pickle.load(pkl_file)

with open('property_params.json', 'w') as json_file:
    json.dump(property_params, json_file, indent=4)
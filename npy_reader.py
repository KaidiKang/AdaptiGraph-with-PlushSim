import numpy as np

# Load the .npy file
extrinsic = np.load('cloth/cameras/extrinsic.npy')
intrinsic = np.load('cloth/cameras/intrinsic.npy')

# Print the extrinsic and intrinsic matrices
print("Extrinsic matrix:")
print(extrinsic)
print("Intrinsic matrix:")
print(intrinsic)
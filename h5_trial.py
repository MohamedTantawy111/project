import h5py
import numpy as np

# Example: Simulated n frames (e.g., 50 frames)
num_frames = 50  # User-defined
frames_list = [np.random.randint(0, 1024, (1088, 2048), dtype=np.uint16) for _ in range(num_frames)]  
frames_list_two=np.load("list_of_frames.npy", allow_pickle=True)
# Convert list to NumPy array (Shape: [num_frames, height, width])
frames_array = np.stack(frames_list)  # Shape: (num_frames, 1088, 2048)
print("Loaded list_of_frames shape:", frames_list_two.shape)
print(type(frames_list_two))
print(len(frames_list_two))
print(len(frames_list_two[0]))
print(len(frames_list_two[0][0]))

#print(frames_list_2.shape)


# Save to HDF5
with h5py.File("hyperspectral_data.h5", "w") as h5f:
    h5f.create_dataset("frames", data=frames_array, dtype=np.uint16, compression="gzip")
    print("HDF5 file saved successfully! ✅")

with h5py.File("hyperspectral_data.h5", "r") as h5f:
    frames_array_read = h5f["frames"][:]  # Load as NumPy array
    print("Loaded data shape:", frames_array_read)  # (num_frames, 1088, 2048)
# Convert back to list of frames (each frame is a 2D list)
#frames_list = [frame.tolist() for frame in frames_array_read]

# Verify conversion
print(type(frames_array_read))          # Should be <class 'list'>
print(len(frames_array_read))
print(len(frames_array_read[0]))       # Should be <class 'list'>
print(len(frames_array_read[0][0]))    # Should be <class 'list'>
print("Restored list structure successfully! ✅")

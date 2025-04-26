import numpy as np

import numpy as np

def apply_correction_matrix(cube_2d, correction_matrix):
    """
    Apply a spectral correction matrix to a hyperspectral cube.
    
    Args:
        cube_2d (np.ndarray): Input hyperspectral cube of size (1088, 2048, 192).
        correction_matrix (np.ndarray): Correction matrix of size (150, 192).
    
    Returns:
        np.ndarray: Corrected hyperspectral cube of size (1088, 2048, 150).
    """
    # Validate dimensions
    scene_rows, sensor_width, real_filters = cube_2d.shape
    virtual_bands, matrix_filters = correction_matrix.shape
    
    if real_filters != matrix_filters:
        raise ValueError("Number of filters in the cube and correction matrix must match.")
    
    # Reshape the cube for efficient matrix multiplication
    cube_flat = cube_2d.reshape(-1, real_filters)  # Shape: (1088 * 2048, 192)
    
    # Apply correction matrix to each pixel
    corrected_cube_flat = cube_flat @ correction_matrix.T  # Shape: (1088 * 2048, 150)
    
    # Reshape back to the original spatial dimensions
    corrected_cube = corrected_cube_flat.reshape(scene_rows, sensor_width, virtual_bands)  # Shape: (1088, 2048, 150)
    
    return corrected_cube

# Example Usage
if __name__ == "__main__":
    
    
    cube_2d = np.load("step_motor_1st_2d_cube.npy", allow_pickle=True)
    reflection_matrices = np.load("reflection_matrices.npy", allow_pickle=True)
    irradiance_matrices = np.load("irradiance_matrices.npy", allow_pickle=True)
    
    


    # Apply correction
    new_reflection_corrected_step_motor_1st_cube = apply_correction_matrix(cube_2d, reflection_matrices)
    new_irradiance_corrected_step_motor_1st__cube = apply_correction_matrix(cube_2d, irradiance_matrices)

    print("reflection_corrected_cube Shape:", new_reflection_corrected_step_motor_1st_cube.shape)  # Should be (1088, 2048, 150)
    print("irradiance_corrected_cube Shape:", new_irradiance_corrected_step_motor_1st__cube.shape)  # Should be (1088, 2048, 150)
   
    np.save('new_reflection_corrected_cube.npy', new_reflection_corrected_step_motor_1st_cube)
    np.save('new_irradiance_corrected_cube.npy', new_irradiance_corrected_step_motor_1st__cube)
    
from ximea import xiapi
import numpy as np
import matplotlib.pyplot as plt
import serial
import time
import spectral

SCENE_ROWS = 300
frame_rate = 80
exposure = (1000 / frame_rate ) - 10
speed_motor = 6.5*frame_rate
camera_gain = 4


reflection_matrices = np.load("reflection_matrices.npy", allow_pickle=True)
irradiance_matrices = np.load("irradiance_matrices.npy", allow_pickle=True)
wavelengths = np.load("virtual_wavelengths.npy", allow_pickle=True)


NUM_FRAMES = 216 + (SCENE_ROWS - 1)

# Sensor specs
SENSOR_HEIGHT = 1088
SENSOR_WIDTH  = 2048

# Output cube shape: (scene_rows, columns, filters) => (2, 2048, 192)
cube_2d = np.zeros((SCENE_ROWS, SENSOR_WIDTH, 192), dtype=np.float32)

# Initialize serial communication
try:
    arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
except Exception as e:
    print(f"Error: {e}")
    exit()


def send_command(command):
    """Send a command to the Arduino and print the response."""
    arduino.write((command + '\n').encode('utf-8'))  # Send command
    time.sleep(0.1)  # Wait for Arduino to process the command
    while arduino.in_waiting > 0:  # Check if there's data to read
        response = arduino.readline().decode('utf-8').strip()
        print(f"Arduino: {response}")

def set_speed(speed):
    """Send speed command to Arduino."""
    send_command(f"speed {speed}")

def set_direction(direction):
    """Send direction command to Arduino."""
    if direction in [0, 1]:
        send_command(f"direction {direction}")
    else:
        print("Invalid direction value. Use 0 for reverse, 1 for forward.")

##############################################################################
# 1) SETTINGS & GLOBALS
##############################################################################

# We'll build a cube for 100 scene rows:

# Each row in the scene needs 216 frames, offset by 1 for each additional row

##############################################################################
# 2) SINGLE-ROW MAPPING: frame_idx -> (row_start, filter_index)
##############################################################################
def row_and_filter_for_frame(frame_idx):
    """
    Returns (row_start, filter_idx) for a single-row scenario:
      - frame_idx in [0..63]  => Area #1 => filters 0..63
      - frame_idx in [64..87] => gap (24 frames) => filter_idx=None
      - frame_idx in [88..215]=> Area #2 => filters 64..191
    If out of range, raises ValueError.
    """
    if 0 <= frame_idx < 64:
        # Filters #0..63 in Area #1 => row 4..323
        row_start = 4 + frame_idx * 5
        filter_idx = frame_idx
    elif 64 <= frame_idx < 88:
        # 24-frame gap => row 324..443
        row_start = 324 + (frame_idx - 64) * 5
        filter_idx = None
    elif 88 <= frame_idx < 216:
        # Filters #64..191 in Area #2 => row 444..1083
        row_start = 444 + (frame_idx - 88) * 5
        filter_idx = 64 + (frame_idx - 88)
    else:
        raise ValueError("frame_idx out of range (0..215) for single-row logic.")

    return row_start, filter_idx

##############################################################################
# 3) MULTI-ROW MAPPING: (scene_row, frame_idx) -> (row_start, filter_idx)
##############################################################################
def row_and_filter_for_frame_2d(scene_row, frame_idx):
    """
    Because row #1 is offset by +1 frame relative to row #0,
    we effectively do frame_idx - scene_row before calling the single-row logic.
    
    If (frame_idx - scene_row) < 0, we haven't reached filter #0 yet -> return None.
    If (frame_idx - scene_row) > 215, we've gone past filter #191 -> also None.
    """
    single_row_frame = frame_idx - scene_row
    if single_row_frame < 0:
        return None, None  # not valid yet

    try:
        row_start, filter_idx = row_and_filter_for_frame(single_row_frame)
    except ValueError:
        # means single_row_frame >= 216 => out of range
        return None, None

    return row_start, filter_idx


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
 

##############################################################################
# 4) MAIN
##############################################################################
def main():
    # 4.1) Open camera, configure, start acquisition
    cam = xiapi.Camera()
    print('Opening camera...')
    cam.open_device()

    cam.set_imgdataformat('XI_RAW16')
    cam.set_exposure(exposure*1000)  # for example, 10 ms
    cam.set_gain(camera_gain)
    cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
    cam.set_framerate(frame_rate)
    print(f"Frame rate set to {cam.get_framerate()} fps")
    cam.start_acquisition()
    
    # 4.2) Acquire the frames
    print(f'Acquiring {NUM_FRAMES} frames...')
    frames_buffer = []
    set_speed(speed_motor)  # Stop the motor
    for f in range(NUM_FRAMES):

        img = xiapi.Image()
        cam.get_image(img)
        frame_data = np.frombuffer(img.get_image_data_raw(), dtype=np.uint16).reshape(SENSOR_HEIGHT, SENSOR_WIDTH)
        frames_buffer.append(frame_data)
        print(f'Acquiring frame {f}/{NUM_FRAMES} ...')
        
    # Stop & close
    print("Stopping motor...")
    set_speed(0)  # Stop the motor
    arduino.close()
    cam.stop_acquisition()
    cam.close_device()
    print('Capture complete. Processing...')

    # 4.3) Build the (SCENE_ROWS, 2048, 192) cube
    #####################np.save('list_of_frames.npy', frames_buffer)
    print("Saved list_of_frames to 'list_of_frames.npy'.")
    for scene_r in range(SCENE_ROWS):
        for f in range(NUM_FRAMES):
            row_start, fi = row_and_filter_for_frame_2d(scene_r, f)
            if fi is not None:
                # Average the 5 sensor rows => intensity for this filter
                pixel_block = frames_buffer[f][row_start : row_start + 5, :]  # shape (5, 2048)
                row_mean = pixel_block.mean(axis=0)  # shape => (2048,)
                # Store in the last dimension => spectral channel fi
                cube_2d[scene_r, :, fi] = row_mean

    # 4.4) Save the result to disk
    
    print("cube_2d real shape:", cube_2d.shape)  # e.g. (2, 2048, 192)
    
    # Apply correction
    new_reflection_corrected_step_motor_1st_cube = apply_correction_matrix(cube_2d, reflection_matrices)
    new_irradiance_corrected_step_motor_1st__cube = apply_correction_matrix(cube_2d, irradiance_matrices)

    print("reflection_corrected_cube Shape:", new_reflection_corrected_step_motor_1st_cube.shape)  # Should be (1088, 2048, 150)
    print("irradiance_corrected_cube Shape:", new_irradiance_corrected_step_motor_1st__cube.shape)  # Should be (1088, 2048, 150)
       
    new_reflection_corrected_step_motor_1st_cube = (new_reflection_corrected_step_motor_1st_cube - new_reflection_corrected_step_motor_1st_cube.min()) / (new_reflection_corrected_step_motor_1st_cube.max() - new_reflection_corrected_step_motor_1st_cube.min())

    new_irradiance_corrected_step_motor_1st__cube = (new_irradiance_corrected_step_motor_1st__cube - new_irradiance_corrected_step_motor_1st__cube.min()) / (new_irradiance_corrected_step_motor_1st__cube.max() - new_irradiance_corrected_step_motor_1st__cube.min())
    # Save the cube in ENVI format with wavelength metadata
    spectral.envi.save_image(
        'new_reflection_corrected_step_motor_1st_cube.hdr',            # <-- 1st argument: the header filename
        new_reflection_corrected_step_motor_1st_cube,                          # <-- 2nd argument: your data
        metadata={
            'wavelength': wavelengths,
            'wavelength units': 'nm'
        },
        interleave='bsq',              # or 'bil'/'bip' if needed
        dtype=np.float32,              # match your NumPy dtype
        byteorder=0,                   # 0 for little-endian
        force=True,                    # overwrite existing files
        ext='.raw'                     # force data file to be named 'my_hsi_image.dat'
    )

    spectral.envi.save_image(
        'new_irradiance_corrected_step_motor_1st__cube.hdr',            # <-- 1st argument: the header filename
        new_irradiance_corrected_step_motor_1st__cube,                          # <-- 2nd argument: your data
        metadata={
            'wavelength': wavelengths,
            'wavelength units': 'nm'
        },
        interleave='bsq',              # or 'bil'/'bip' if needed
        dtype=np.float32,              # match your NumPy dtype
        byteorder=0,                   # 0 for little-endian
        force=True,                    # overwrite existing files
        ext='.raw'                     # force data file to be named 'my_hsi_image.dat'
    )

    print("ENVI files created: my_hsi_image.hdr and my_hsi_image.dat")

   
if __name__ == '__main__':
    main()




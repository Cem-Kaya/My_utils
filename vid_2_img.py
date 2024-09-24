import cv2
import numpy as np
import os
# ns-process-data  images --data ./output_frames --output-dir ./data --verbose  --camera-type fisheye  --gpu
def is_sharp(image):
    """Returns sharpness value of the image based on Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_sharp_frames(video_path, output_folder, threshold=15, sharp_frame_interval=10, flip_horizontal=False, flip_vertical=False):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    sharp_frame_count = 0

    # Variables to keep track of sharpest frame in each group of sharp frames
    max_variance = 0
    sharpest_frame = None
    
    # Check if output folder exists and create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        variance = is_sharp(frame)
        if variance > threshold:
            sharp_frame_count += 1
            
            if variance > max_variance:
                max_variance = variance
                sharpest_frame = frame

            if sharp_frame_count % sharp_frame_interval == 0:  # After every sharp_frame_interval-th sharp frame
                # Apply conditional flipping
                if flip_horizontal:
                    sharpest_frame = cv2.flip(sharpest_frame, 1)  # 1 denotes flipping around the y-axis
                if flip_vertical:
                    sharpest_frame = cv2.flip(sharpest_frame, 0)  # 0 denotes flipping around the x-axis
                
                saved_count += 1
                save_path = f"{output_folder}/frame_{frame_count}.jpg"
                cv2.imwrite(save_path, sharpest_frame)

                # Reset the variables for the next group of sharp frames
                max_variance = 0
                sharpest_frame = None

        frame_count += 1

    cap.release()
    print(f"Total Frames: {frame_count}, Saved Frames: {saved_count}")

video_path = 'v.mp4'
output_folder = './output_frames'
extract_sharpness_threshold = 20
extract_sharp_frames(video_path, output_folder, extract_sharpness_threshold, sharp_frame_interval=20, flip_horizontal=True, flip_vertical=True)

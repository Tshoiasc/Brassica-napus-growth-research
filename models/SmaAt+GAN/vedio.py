import os
import cv2
import re

# Function to sort files based on the number in their name
def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Directory paths
base_dir = "validation_results/"
output_video_path = "result_video.mp4"

# Collect all image paths
image_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".png"):
            image_paths.append(os.path.join(root, file))

# Sort image paths based on the number in the file name
image_paths = sorted(image_paths, key=numerical_sort)

# Read the first image to get the size
frame = cv2.imread(image_paths[0])
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, 5, (width, height))  # 1 FPS

# Write images to the video
for image_path in image_paths:
    img = cv2.imread(image_path)
    video.write(img)

# Release the video writer
video.release()

output_video_path

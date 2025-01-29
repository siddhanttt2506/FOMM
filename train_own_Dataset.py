import matplotlib
matplotlib.use('Agg')

import os
import sys
yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import torch
import cv2
import numpy as np
from imageio import imread, mimread
from skimage.color import gray2rgb
from skimage.util import img_as_float32
from torchvision.transforms import ToTensor
from PIL import Image

from frames_dataset import FramesDataset
from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
from train import train
from reconstruction import reconstruction
from animate import animate

# Load the configuration file
config_path = "config/vox-256.yaml"  # Replace with your config file path
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print("Config Loaded Successfully")

# Initialize Generator
generator = OcclusionAwareGenerator(**config['model_params']['generator_params'], 
                                    **config['model_params']['common_params'])

# Initialize Discriminator
discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'], 
                                        **config['model_params']['common_params'])

# Initialize Keypoint Detector
kp_detector = KPDetector(**config['model_params']['kp_detector_params'], 
                         **config['model_params']['common_params'])

# Move models to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)
kp_detector.to(device)

print("Models Initialized Successfully")

# Load Dataset
dataset = FramesDataset(is_train=True, **config['dataset_params'])
print("Dataset Loaded Successfully")

# Load checkpoint if available
checkpoint_path = None  # Replace with your checkpoint path if needed
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    print("Checkpoint Loaded Successfully")
else:
    print("No Checkpoint Provided. Starting Fresh.")

# Read and process an image
image_path = "sample.jpg"  # Replace with an actual image path
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (256, 256))
cv2.imwrite("resized_image.jpg", resized_image)

# Test Keypoint Detection
test_image_path = "resized_image.jpg"
test_image = Image.open(test_image_path).convert('RGB')
test_tensor = ToTensor()(test_image).unsqueeze(0).to(device)

with torch.no_grad():
    keypoints = kp_detector(test_tensor)

print("Keypoints Detected Successfully")
print("Keypoints:", keypoints)

# Define log directory
log_dir = "log_directory"  # Replace with your log directory path
os.makedirs(log_dir, exist_ok=True)

# Train 
train(config, generator, discriminator, kp_detector, checkpoint_path, log_dir, dataset, device_ids=[0])

def read_video(name, frame_shape):
    """
    Read video which can be:
      - a single image of concatenated frames
      - a '.mp4', '.gif', or '.mov' video file
      - a folder with image frames
    """
    try:
        if os.path.isdir(name):
            frames = sorted(os.listdir(name))
            video_array = np.array(
                [img_as_float32(imread(os.path.join(name, frames[idx]))) for idx in range(len(frames))]
            )
        elif name.lower().endswith(('.png', '.jpg')):
            image = imread(name)
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = gray2rgb(image)
            if image.shape[2] == 4:
                image = image[..., :3]  # Remove alpha channel
            image = img_as_float32(image)
            video_array = np.moveaxis(image, 1, 0).reshape((-1,) + frame_shape)
            video_array = np.moveaxis(video_array, 1, 2)
        elif name.lower().endswith(('.gif', '.mp4', '.mov')):
            video = np.array(mimread(name))
            if len(video.shape) == 3:
                video = np.array([gray2rgb(frame) for frame in video])
            if video.shape[-1] == 4:
                video = video[..., :3]
            video_array = img_as_float32(video)
        else:
            raise ValueError(f"Unknown file extension: {name}")
        return video_array
    except Exception as e:
        print(f"Error reading {name}: {e}")
        return None

# Define the file path
video_path = "test_video.mp4"  # Replace with your actual video file path
frame_shape = (256, 256, 3)
video_array = read_video(video_path, frame_shape)

if video_array is not None:
    print("Video read successfully!")
    print(f"Video shape: {video_array.shape}")
    for frame in video_array:
        frame = (frame * 255).astype(np.uint8)
        cv2.imshow("Video", frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
else:
    print("Failed to read the video.")

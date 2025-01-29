import os
import sys
import yaml
import warnings
import imageio
import numpy as np
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import gmtime, strftime
from shutil import copy
from argparse import ArgumentParser
from skimage.transform import resize
from IPython.display import HTML
from tqdm import tqdm
from scipy.spatial import ConvexHull

from torch.utils.data import DataLoader

from frames_dataset import FramesDataset, PairedDataset
from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback
from logger import Logger, Visualizer

from train import train
from reconstruction import reconstruction
from animate import animate

# Set matplotlib backend
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings("ignore")

# Enable inline plotting (if using Jupyter Notebook)
%matplotlib inline


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    # Load configuration
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize models
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    # Move models to CPU if specified or if CUDA is not available
    if cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        generator.to(device)
        kp_detector.to(device)
    else:
        device = torch.device('cuda')
        generator.to(device)
        kp_detector.to(device)

    # Load checkpoint
    if cpu or not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    # Load state dicts into models
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    # Wrap models with DataParallel if CUDA is available and not running on CPU
    if not cpu and torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    # Set models to evaluation mode
    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def display(source, driving, generated):
    """Show the source image and the driving video using OpenCV."""
    # Convert source to BGR (for OpenCV)
    if source.dtype != np.uint8:  # Check if the image is not 8-bit unsigned integer
        source = cv2.normalize(source, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)

    # Iterate over the frames of the driving video
    for i in range(len(driving)):
        # Convert driving frame to BGR (for OpenCV)
        if driving[i].dtype != np.uint8:  # Check if the frame is not 8-bit unsigned integer
            driving[i] = cv2.normalize(driving[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        driving_frame = cv2.cvtColor(driving[i], cv2.COLOR_RGB2BGR)
        
        # Create the frame to be shown
        frame = np.concatenate([source, driving_frame], axis=1)

        # If generated frames are provided, concatenate them as well
        if generated is not None:
            if generated[i].dtype != np.uint8:  # Check if the frame is not 8-bit unsigned integer
                generated[i] = cv2.normalize(generated[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            generated_frame = cv2.cvtColor(generated[i], cv2.COLOR_RGB2BGR)
            frame = np.concatenate([source, driving_frame, generated_frame], axis=1)
        
        # Display the concatenated frame
        cv2.imshow('Source + Driving + Generated', frame)
        
        # Wait for a key press to move to the next frame
        key = cv2.waitKey(50)  # 50ms delay between frames
        if key == 27:  # Press 'Esc' to exit the video display
            break

    # Release the OpenCV window
    cv2.destroyAllWindows()

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cpu()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cpu()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

import cv2
import numpy as np

def display_and_save(source, driving, generated, output_path='output.mp4', fps=20):
    """Show the source image and the driving video using OpenCV, and save the video."""
    # Convert source to BGR (for OpenCV)
    if source.dtype != np.uint8:
        source = cv2.normalize(source, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    
    # Get frame dimensions for VideoWriter
    height, width, _ = source.shape
    total_width = width * (3 if generated is not None else 2)
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, height))

    for i in range(len(driving)):
        # Convert driving frame to BGR (for OpenCV)
        if driving[i].dtype != np.uint8:
            driving[i] = cv2.normalize(driving[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        driving_frame = cv2.cvtColor(driving[i], cv2.COLOR_RGB2BGR)
        
        # Create the frame to be shown
        frame = np.concatenate([source, driving_frame], axis=1)

        # If generated frames are provided, concatenate them as well
        if generated is not None:
            if generated[i].dtype != np.uint8:
                generated[i] = cv2.normalize(generated[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            generated_frame = cv2.cvtColor(generated[i], cv2.COLOR_RGB2BGR)
            frame = np.concatenate([frame, generated_frame], axis=1)
        
        # Write the frame to the video file
        video_writer.write(frame)
        
        # Display the concatenated frame
        cv2.imshow('Source + Driving + Generated', frame)
        
        # Wait for a key press to move to the next frame
        key = cv2.waitKey(50)  # 50ms delay between frames
        if key == 27:  # Press 'Esc' to exit the video display
            break

    # Release the VideoWriter and destroy OpenCV windows
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Video saved at: {output_path}")

generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                            checkpoint_path="Your checkpoint path")

source_image = imageio.imread("source_image_path")
driving_video = imageio.mimread("driving video path")

#Resize inputs to 256x256
source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True,
                             adapt_movement_scale=True)

print("Source image shape:", source_image.shape)

print("Driving video shape:", len(driving_video), "frames with shape:", driving_video[0].shape)

print("Predictions shape:", len(predictions), "frames with shape:", predictions[0].shape)

print("Source image dtype:", source_image.dtype)
print("Driving video frame dtype:", driving_video[0].dtype)
print("Predictions frame dtype:", predictions[0].dtype)

import numpy as np

# Normalize and convert source image
if source_image.dtype != np.uint8:
    source_image = cv2.normalize(source_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Normalize and convert driving video frames
driving_video = [cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) for frame in driving_video]

# Normalize and convert prediction frames
predictions = [cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) for frame in predictions]

print("Source image shape:", source_image.shape, "dtype:", source_image.dtype)
print("Driving video shape:", len(driving_video), "frames with shape:", driving_video[0].shape, "dtype:", driving_video[0].dtype)
print("Predictions shape:", len(predictions), "frames with shape:", predictions[0].shape, "dtype:", predictions[0].dtype)

print("Is source image empty?", np.all(source_image == 0))
print("Is driving video frame empty?", np.all(driving_video[0] == 0))
print("Is predictions frame empty?", np.all(predictions[0] == 0))

display(source_image, driving_video, predictions)

display_and_save(source_image, driving_video, predictions)
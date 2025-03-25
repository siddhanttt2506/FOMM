

# First Order Motion Model for Image Animation

This repository contains an implementation of the [First Order Motion model](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation). This project allows you to animate images from an input driving video.

## Example animations
The videos on the left show the driving videos. The first row on the right for each dataset shows the source videos. The bottom row contains the animated sequences with motion transferred from the driving video and object taken from the source image. We trained a separate network for each task.

### VoxCeleb Dataset
[![Watch the video](https://raw.githubusercontent.com/shrutisekhar02/FOMM/main/input/Monalisa.png)](https://raw.githubusercontent.com/shrutisekhar02/FOMMy/main/output.mp4)

![Demo](output.gif)

### Installation

To install the dependencies run:
```
pip install -r requirements.txt
```

### YAML configs

See ```config/vox-256.yaml``` to get description of each parameter.


### Animation Demo
To run a demo, download checkpoints and ensure the paths of the driving video, source image and config file are set. The script assumes that the input image and video are resized to 256x256 pixels. The script will display the animation in a window. Press Esc to exit the display.The generated video will be saved as output.mp4 in the current directory. 

```
python main.py
```
                                                                                                                                                                    
### Training

To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_own_Dataset.py 
```
Ensure the path to the datset and other training parameters are adjusted in the config file.
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder.
To check the loss values during training see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.

### Dataset Generation 
To make your own datset, follow the instructions from https://github.com/AliaksandrSiarohin/video-preprocessing.

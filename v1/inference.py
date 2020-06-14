
import os
import sys
import glob
import fnmatch
from timeit import default_timer as timer
from pathlib import Path
from shutil import copyfile

import matplotlib
import numpy as np
from PIL import Image
import cv2
import hydra
from omegaconf import DictConfig

from ..tools.cropper import crop_from_raw_input

@hydra.main(config_path="config.yaml")
def demo_plant_inference(cfg : DictConfig):
    image = cv2.imread(cfg.DEMO_PLANT.INPUT_PATH)
    plant_inference(image)

@hydra.main(config_path="config.yaml")
def plant_inference(cfg: DictConfig, image: np.ndarray):
    """
    Args:
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.

    Returns:
        predictions (dict): the output of the model.
        vis_output (VisImage): the visualized image output.

    plant means set of pipline i.e. detector as a whole
    1. ATTENTION EVALUATION
    2. ACTIVE CROP SELECTION
    3. FINAL EVALUATION
    """
    print(cfg.pretty())

    # 1. first crop
    try:
        first_crops = crop_from_raw_input(image)
    except CropperSizeError as e:
        print('Image size is not supported or there are some bugs about shape.')
    
        
if __name__ == '__main__':
    demo_plant_inference()
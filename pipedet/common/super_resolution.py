
import cv2
from cv2 import dnn_superres

import numpy as np

def super_resolution(image: np.ndarray) -> np.ndarray:
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Read the desired model
    path = "/home/appuser/data/weight_for_super_resolution/EDSR_x4.pb"
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 4)

    # Upscale the image
    upsampled = sr.upsample(image)

    return upsampled
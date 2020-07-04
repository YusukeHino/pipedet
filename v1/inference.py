
import os
import sys
import glob
import base64
import io
import json
import fnmatch
from timeit import default_timer as timer
from pathlib import Path
from shutil import copyfile

import requests
import matplotlib
import numpy as np
from PIL import Image
import cv2
import hydra
from omegaconf import DictConfig

from ..tools import cropper

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

@hydra.main(config_path="config.yaml")
def internal_inference():
    pass

def container_predict(image_file_path, image_key, port_number=8503):
    """Sends a prediction request to TFServing docker container REST API.

    Args:
        image_file_path: Path to a local image for the prediction request.
        image_key: Your chosen string key to identify the given image.
        port_number: The port number on your device to accept REST API calls.
    Returns:
        The response of the prediction request.
    """

    with io.open(image_file_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # The example here only shows prediction with one image. You can extend it
    # to predict with a batch of images indicated by different keys, which can
    # make sure that the responses corresponding to the given image.
    instances = {
            'instances': [
                    {'image_bytes': {'b64': str(encoded_image)},
                      'key': image_key}
            ]
    }

    # This example shows sending requests in the same server that you start
    # docker containers. If you would like to send requests to other servers,
    # please change localhost to IP of other servers.
    url = 'http://localhost:{}/v1/models/default:predict'.format(port_number)

    response = requests.post(url, data=json.dumps(instances))
    print(response.json())
        
if __name__ == '__main__':
    # demo_plant_inference()
    PATH_TO_IMAGE = "/usr/src/app/data/for_rsm_detection/sample_images/sample_image_mirror_detection_0001.png"
    container_predict(PATH_TO_IMAGE, "inference_test")
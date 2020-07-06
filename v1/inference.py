
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

from pipedet.structure import cropper
from pipedet.structure import cv2_to_base64_string

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

def container_predict(image_file_path, image_key, port_number=8501):
    """Sends a prediction request to TFServing docker container REST API.

    Args:
        image_file_path: Path to a local image for the prediction request.
        image_key: Your chosen string key to identify the given image.
        port_number: The port number on your device to accept REST API calls.
    Returns:
        The response of the prediction request.
        {'predictions': [{'key': 'inference_test', 'detection_multiclass_scores': [[0.00344526768, 0.997], [0.00373196602, 0.477866769], [0.00257599354, 0.137169182], ...], 'detection_classes': [1.0, 1
.0, 1.0, 1.0, ...], 'num_detections': 40.0, 'image_info': [512, 512, 1, 0, 512, 512], 'detection_boxes': [[0.488796, 0.85602057, 0.594647
646, 0.953933], [0.483729571, 0.976985693, 0.587875247, 1.0], [0.372124255, 0.0, 0.479369521, 0.00482739
229], [0.356921822, 0.0, 0.488324493, 0.00728783756], [0.40906927, 0.000657424098, 0.512865424, 0.004928
97443], [0.33150664, 0.0, 0.486344904, 0.00281080115], [0.502710104, 0.973039031, 0.601225376, 0.9989502
43], [0.358610392, 0.000676913653, 0.502567172, 0.00986657478], [0.333896577, 0.00413428, 0.512988269, 0
.0157458391], [0.447005063, 0.000295822276, 0.536275148, 0.00365979434], [0.407772958, 0.000505766366, 0
.523653388, 0.00771948649], [0.485350043, 0.964359462, 0.576633513, 1.0], [0.48660171, 0.984374464, 0.59
0121627, 0.999776661], [0.300048769, 0.0, 0.472841799, 0.00200588442], [0.717599332, 0.996715546, 0.7908
79905, 1.0], [0.475850761, 0.989976227, 0.583369195, 0.998669922], [0.689930916, 0.996614695, ...

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
    url = 'http://automl_20200614:{}/v1/models/default:predict'.format(port_number)

    response = requests.post(url, data=json.dumps(instances))
    print(response.json())
        
if __name__ == '__main__':
    # demo_plant_inference()
    PATH_TO_IMAGE = "/home/appuser/data/for_rsm_detection/cropped_1024_1024/test/20181205_005_0000001_0.png"
    container_predict(PATH_TO_IMAGE, "inference_test")

import os
import unittest

import numpy as np
import cv2
from cv2 import dnn_superres

_input_path = "/home/appuser/src/pipedet/tests/demo_sr/original/0069.png"
_output_path = "/home/appuser/src/pipedet/tests/demo_sr/demo/0069.png"


class TestSuperResolution(unittest.TestCase):

    def test_super_resolution(self):

        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()

        # Read image
        image = cv2.imread(_input_path)

        # Read the desired model
        path = "/home/appuser/data/weight_for_super_resolution/EDSR_x4.pb"
        sr.readModel(path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("edsr", 4)

        # Upscale the image
        result = sr.upsample(image)

        # Save the image
        cv2.imwrite(_output_path, result)
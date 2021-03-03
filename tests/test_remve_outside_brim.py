import os
import unittest

import numpy as np
import cv2


denominators_of_beta = (10, 10)

beta = (1/denominators_of_beta[0], 1/denominators_of_beta[1])

_ROOT_INPUT = '/home/appuser/data/facing_via_mirror/for_ellipse_tuning'

_FILENAMES_INPUT = [
    '20200918_024_0077.png',
    '20210120_030_0120.png'
]

_ROOT_OUT = '/home/appuser/src/pipedet/tests/demo_remove'

_out_dirname = f'{denominators_of_beta[0]}_{denominators_of_beta[1]}'

assert os.path.isdir(_ROOT_OUT)


class TestRemover(unittest.TestCase):
    def test_remove(self):
        for filename in _FILENAMES_INPUT:
            path_input = os.path.join(_ROOT_INPUT, filename)
            image = cv2.imread(path_input)
            image = cv2.resize(image, (512, 512))
            image = remove_outside_brim_with_ellipse(image)
            root_output = os.path.join(_ROOT_OUT, _out_dirname)
            os.makedirs(root_output, exist_ok=True)
            path_output = os.path.join(root_output, filename)
            cv2.imwrite(path_output, image)
            self.assertTrue(os.path.isfile(path_output))

def remove_outside_brim_with_ellipse(image):
    width, height = (512, 512)
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if (x - 1/2 * width) ** 2 / (1/2 * width - beta[0]*width) ** 2 + (y - 1/2 * height) ** 2 / (1/2*height-beta[1]*height) ** 2 <= 1:
                image[y, x, :] = 0
    return image
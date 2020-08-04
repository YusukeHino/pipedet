
import os
import unittest

import numpy as np
import cv2

from pipedet.structure.large_image import LargeImage

TEST_ROOT = os.path.dirname(__file__)

IMAGE_ROOT = os.path.join(TEST_ROOT, 'demo_images')
RESULT_ROOT = os.path.join(TEST_ROOT, 'demo_result_images')

IMAGE_NAMES = [
    "demo_3840_2160.jpg",
    "demo_3840_2160_0065.jpg",
]

PATH_TO_IMAGES = [os.path.join(IMAGE_ROOT, image_name) for image_name in IMAGE_NAMES]

PATH_TO_RESULT_IMAGES = [os.path.join(RESULT_ROOT, image_name) for image_name in IMAGE_NAMES]

ROOT_MINIMUM = "/home/appuser/data/facing_via_mirror/3840_2160_60fps/minimum/20200124_022_minimum/frames"

ROOT_MINIMUM_RES = "/home/appuser/src/pipedet/tests/demo_result_images/frames"

SLIDES = [
    (0, 0),
    (-200, 0),
    (-200, 200),
]

class TestLargeImage(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, count):
        original_image = cv2.imread(PATH_TO_IMAGES[0])
        self.dx, self.dy = SLIDES[count]
        rows, cols, _ = original_image.shape
        M = np.float32([[1,0,self.dx],[0,1,self.dy]])
        shifted_image = cv2.warpAffine(original_image, M, (cols,rows))
        self.large_image = LargeImage(shifted_image)

    def tearDown(self):
        del self.large_image

    def test_pipedet_1(self):
        self.setup(0)
        cv2.imwrite(PATH_TO_IMAGES[0][:-4] + f'_slide_{self.dx}_{self.dy}.jpg', self.large_image.image)
        self.large_image.pipe_det(first_thre=0.5, second_thre=0.5, patch_width=1024, patch_height=1024)
        cv2.imwrite(PATH_TO_RESULT_IMAGES[0][:-4] + f'_slide_{self.dx}_{self.dy}.jpg', self.large_image.image_drawn)

    def test_pipedet_2(self):
        self.setup(1)
        cv2.imwrite(PATH_TO_IMAGES[0][:-4] + f'_slide_{self.dx}_{self.dy}.jpg', self.large_image.image)
        self.large_image.pipe_det(first_thre=0.5, second_thre=0.5, patch_width=1024, patch_height=1024)
        cv2.imwrite(PATH_TO_RESULT_IMAGES[0][:-4] + f'_slide_{self.dx}_{self.dy}.jpg', self.large_image.image_drawn)

    def test_pipedet_3(self):
        self.setup(2)
        cv2.imwrite(PATH_TO_IMAGES[0][:-4] + f'_slide_{self.dx}_{self.dy}.jpg', self.large_image.image)
        self.large_image.pipe_det(first_thre=0.5, second_thre=0.5, patch_width=1024, patch_height=1024)
        cv2.imwrite(PATH_TO_RESULT_IMAGES[0][:-4] + f'_slide_{self.dx}_{self.dy}.jpg', self.large_image.image_drawn)

    def test_pipedet_4(self):
        for frame_num in range(1, 337):
            image_name = str(frame_num).zfill(4) + ".jpg"
            path_image = os.path.join(ROOT_MINIMUM, image_name)
            image = cv2.imread(path_image)
            large_image = LargeImage(image)
            large_image.pipe_det(first_thre=0.5, second_thre=0.5, patch_width=1024, patch_height=1024)
            cv2.imwrite(os.path.join(ROOT_MINIMUM_RES, image_name), large_image.image_drawn)

    def test_depict_track_1(self):
        self.setup(0)
        cv2.imwrite(PATH_TO_IMAGES[0][:-4] + f'_slide_{self.dx}_{self.dy}_depict_track.jpg', self.large_image.image)
        self.large_image.pipe_det(first_thre=0.5, second_thre=0.5, patch_width=1024, patch_height=1024)
        self.large_image.track_ids = [_ for _ in range(1, 1 + len(self.large_image.bboxes))]
        cv2.imwrite(PATH_TO_RESULT_IMAGES[0][:-4] + f'_slide_{self.dx}_{self.dy}_depict_track.jpg', self.large_image.image_drawn)
            

    if __name__ == '__main__':
        unittest.main()
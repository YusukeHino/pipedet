
import unittest
import os
import logging

import cv2

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker
from pipedet.solver.hooks import MirrorDetection, Recorder, ImageWriter, MOTWriter
from pipedet.structure.large_image import LargeImage
from pipedet.solver.get_mirror_seq import MirrorSeq

_root_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20201016_001/frames_png"
_root_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_crop_results/mirror_seq_images"
# _root_output_images_w_road_objects = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_crop_results/w_road_objects"
_input_mot_txt = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/gt.txt"
# _input_road_objects_mot_txt = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq_sr/20200918_038/road_objects_tracked/gt.txt"
_target_tracking_id = 1

class TestMirrorSeq(unittest.TestCase):
    def test_load_w_mot_txt(self):
        mirror_seq = MirrorSeq(_root_images)
        mirror_seq.load_w_mot_txt(_input_mot_txt)
        mirror_seq.crop_n_get_mirror_seq(_target_tracking_id)
        for frame_num, cropped_image in mirror_seq.mapped_mirror_seq.items():
            path_output_image = os.path.join(_root_output_images, str(frame_num).zfill(4) + ".png")
            # cropped_image.super_resolution()
            cv2.imwrite(path_output_image, cropped_image.image)
            logger = logging.getLogger(__name__)
            logger.info(f'frame| {frame_num}')
            print(frame_num)

        # mirror_seq.get_road_objects_from_mot_txt(_input_road_objects_mot_txt)
        # for frame_num, cropped_image in mirror_seq.mapped_mirror_seq.items():
        #     path_output_image = os.path.join(_root_output_images_w_road_objects, str(frame_num).zfill(4) + ".png")
        #     cv2.imwrite(path_output_image, cropped_image.image_drawn)
            
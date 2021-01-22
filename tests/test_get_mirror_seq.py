
import unittest
import os
import logging

import cv2

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker
from pipedet.solver.hooks import MirrorDetection, Recorder, ImageWriter, MOTWriter
from pipedet.structure.large_image import LargeImage
from pipedet.solver.get_mirror_seq import MirrorSeq



# -- for 20201016_001 -- #
# _root_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20201016_001/frames_png"
# _root_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_crop_results/mirror_seq_images"
# _input_mot_txt = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/gt.txt"
# _target_tracking_id = 1

# -- for 20200918_002 -- #
# _root_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20200918_002/frames_png"
# _root_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_crop_results/mirror_seq_images"
# _input_mot_txt = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/gt.txt"
# _target_tracking_id = 3 # The right mirror. There are two mirrors(1 and 3) in the right side.

# -- for 20200918_024 -- #
# _root_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20200918_024/frames_png"
# _root_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_crop_results/mirror_seq_images"
# _input_mot_txt = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/gt.txt"
# _target_tracking_id = 1 # The litle bit right mirror. There are two mirrors(2 and 1) in the center.

# -- for 20210120_001 --#
# _root_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20210120_001/frames_png"
# _root_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_crop_results/mirror_seq_images"
# _input_mot_txt = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/gt.txt"
# _target_tracking_id = 1

# -- for 20210120_008 --#
_root_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20210120_008/frames_png"
_root_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_crop_results/mirror_seq_images"
_input_mot_txt = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/gt.txt"
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
            

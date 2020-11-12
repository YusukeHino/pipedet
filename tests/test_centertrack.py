
import unittest
import os
import logging

import cv2

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker, MOTReadingAsTracker
from pipedet.solver.hooks import MirrorDetection, Recorder, ImageWriter, MOTWriter, MOTReader
from pipedet.structure.large_image import LargeImage
from pipedet.solver.get_mirror_seq import MirrorSeq

# _root_images = "/home/appuser/data/general_in_vehicle_seqs/0016"
_root_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq_sr_512_512/20200918_038/frames"
# _root_output_images = "/home/appuser/src/pipedet/tests/demo_centertrack/w_tracklet_depicted"
_root_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_crop_results/w_road_objects"
# _input_mot_txt = "/home/appuser/data/general_in_vehicle_seqs/0016_txt_tracked/gt.txt"
_input_mot_txt = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq_sr_512_512/20200918_038/road_objects_tracked/gt.txt"

class TestCenterTracker(unittest.TestCase):
    def test_tracking(self):
        tracker = MOTReadingAsTracker()
        tracker.load_frames(root_images=_root_images, frame_num_start=1)
        hooks = [
            Recorder(),
            MOTReader(path_mot_txt=_input_mot_txt),
            ImageWriter(root_output_images=_root_output_images)
        ]
        tracker.register_hooks(hooks)
        tracker.track(0, 509)

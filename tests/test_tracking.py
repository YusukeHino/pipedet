
import unittest
import logging

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker
from pipedet.solver.hooks import MirrorDetection, RoadObjectDetection, BoxCoordinateNormalizer, ApproachingInitializer, AreaCalculator, MidpointCalculator, WidthAndHeihtCalculator, Recorder, ImageWriter, ImageWriterForApproaching, VideoWriter, MOTWriter
from pipedet.structure.large_image import LargeImage

_root_mirror_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20201016_001/frames_png"
_root_mirror_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/frames_for_tracking"
_root_mirror_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/"


# _root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20201016_001/frames"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"
_root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_038/frames"
_root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
_root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
_root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
_root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"

logging.basicConfig(level=logging.INFO)
class TestTrackingFrameLoader(unittest.TestCase):

    def test_loader(self):
        loader = TrackingFrameLoader(root_images=_root_mirror_images)
        loader_iter = loader
        # loader_iter = iter(loader)
        data = next(loader_iter)
        self.assertIsInstance(data, LargeImage)

class TestIoUTracker(unittest.TestCase):

    def test_mirror_tracking(self):
        tracker = IoUTracker()
        tracker.load_frames(root_images=_root_mirror_images)
        hooks = [
            MirrorDetection(0.5, 0.5),
            Recorder(),
            ImageWriter(root_output_images=_root_mirror_output_images),
            MOTWriter(root_output_mot=_root_mirror_output_mot)
        ]
        tracker.register_hooks(hooks)
        tracker.track()

    def test_road_object_tracking(self):
        tracker = IoUTracker()
        tracker.load_frames(root_images=_root_road_object_images)
        hooks = [
            RoadObjectDetection(score_thre = 0.2),
            BoxCoordinateNormalizer(),
            ApproachingInitializer(),
            # AreaCalculator(),
            MidpointCalculator(),
            WidthAndHeihtCalculator(),
            Recorder(),
            ImageWriter(root_output_images=_root_road_object_output_images),
            ImageWriterForApproaching(root_output_images=_root_road_object_output_approaching_images),
            VideoWriter(root_output_video=_root_road_object_output_video),
            MOTWriter(root_output_mot=_root_road_object_output_mot)
        ]
        tracker.register_hooks(hooks)
        tracker.track()
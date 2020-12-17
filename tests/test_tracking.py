
import unittest
import logging

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker
from pipedet.solver.hooks import MirrorDetection, RoadObjectDetection, BoxCoordinateNormalizer, ApproachingInitializer, FeatureMatcher, HorizontalMovementCounter, AreaCalculator, MidpointCalculator, WidthAndHeihtCalculator, Recorder, ImageWriter, ImageWriterForApproaching, VideoWriterForTracking, VideoWriterForApproaching, VideoWriterForMatching, MOTWriter
from pipedet.structure.large_image import LargeImage

# -- for 20200918_002 --#
# _root_mirror_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20200918_002/frames_png"
# _root_mirror_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/frames_for_tracking"
# _root_mirror_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/"


# _root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20201016_001/frames"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"

# -- for 20200918_038 --#
# _root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_038/frames"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"

# -- for 20200918_002 --#
_root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_002/mirror_2/frames"
_root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
_root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
_root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
_root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
_root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
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
        tracker.load_frames(root_images=_root_road_object_images, frame_num_start=-1, frame_num_end=-1)
        hooks = [
            RoadObjectDetection(score_thre = 0.2),
            BoxCoordinateNormalizer(),
            ApproachingInitializer(),
            FeatureMatcher(),
            # HorizontalMovementCounter(right_trend_is_approaching=True),
            # AreaCalculator(),
            MidpointCalculator(),
            # WidthAndHeihtCalculator(),
            Recorder(),
            ImageWriter(root_output_images=_root_road_object_output_images),
            ImageWriterForApproaching(root_output_images=_root_road_object_output_approaching_images),
            VideoWriterForTracking(root_output_video=_root_road_object_output_video, do_resize=True, fps=30.0),
            VideoWriterForApproaching(root_output_video=_root_road_object_output_approaching_video, do_resize=True, fps=30.0),
            VideoWriterForMatching(root_output_video=_root_road_object_output_feature_matching_video, size=(1024,512) ,do_resize=True, fps=30.0),
            MOTWriter(root_output_mot=_root_road_object_output_mot)
        ]
        tracker.register_hooks(hooks)
        tracker.track()
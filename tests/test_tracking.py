
import unittest
import logging

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker, NullTracker
from pipedet.solver.hooks import MirrorDetection, RoadObjectDetection, MOTReader, BoxCoordinateNormalizer, ApproachingInitializer, FeatureMatcher, TransformedMidpointCalculator, TransformedMidpointDifferencer, HorizontalMovementCounter, AreaCalculator, MidpointCalculator, WidthAndHeihtCalculator, RiskyJudger, Recorder, ImageWriter, ImageWriterForApproaching, VideoWriterForTracking, VideoWriterForApproaching, VideoWriterForMatching, MOTWriter, MOTWriterForApproaching, MOTWriterForRisky, FeatureStatWriter
from pipedet.structure.large_image import LargeImage

# -- mir tracking for 20200918_002 --#
# _root_mirror_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20200918_002/frames_png"
# _root_mirror_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/frames_for_tracking"
# _root_mirror_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/"

# -- mir tracking for 20200918_024 --#
# _root_mirror_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20200918_024/frames_png"
# _root_mirror_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/frames_for_tracking"
# _root_mirror_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/video_for_tracking"
# _root_mirror_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/"

# -- mir tracking for 20210120_001 --#
# _root_mirror_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20210120_001/frames_png"
# _root_mirror_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/frames_for_tracking"
# _root_mirror_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/video_for_tracking"
# _root_mirror_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/"

# -- mir tracking for 20210120_008 --#
# _root_mirror_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20210120_008/frames_png"
# _root_mirror_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/frames_for_tracking"
# _root_mirror_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/demo_tracking_result/video_for_tracking"
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
# _root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
# _root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"

# -- for 20200918_002 --#
# _root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_002/mirror_2/frames"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
# _root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"

# -- for 20200918_024 --#
# _root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames"
# _path_input_mot_txt = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/gt.txt"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
# _root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"

# -- for 20210120_001 --#
# _root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20210120_001/mirror_1/frames"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
# _root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"
# _root_road_object_output_approaching_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_approaching"
# _root_road_object_output_risky_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_risky"

# -- for 20210120_008 --#
# _root_road_object_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20210120_008/mirror_1/frames"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
# _root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_tracking"
# _root_road_object_output_approaching_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_approaching"
# _root_road_object_output_risky_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_risky"

# -- for 20210120_018 --#
# _root_road_object_images = "/home/appuser/src/pipedet/experiments/exp/20210120_018/mirror_seqs/00001/frames"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
# _root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"
# _root_road_object_output_approaching_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_approaching"
# _root_road_object_output_risky_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_risky"

# -- for 20210120_004 --#
# _root_road_object_images = "/home/appuser/src/pipedet/experiments/exp/20210120_004/mirror_seqs/00001/frames"
# _root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
# _root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
# _root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
# _root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
# _root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
# _root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"
# _root_road_object_output_approaching_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_approaching"
# _root_road_object_output_risky_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_risky"

# -- for 20210120_029 --#
_root_road_object_images = "/home/appuser/src/pipedet/experiments/exp/20210120_029/mirror_seqs/00001/frames"
_root_road_object_output_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_tracking"
_root_road_object_output_approaching_images = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/frames_for_approaching"
_root_road_object_output_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_tracking"
_root_road_object_output_approaching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_approaching"
_root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/video_for_feature_matching"
_root_road_object_output_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/"
_root_road_object_output_approaching_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_approaching"
_root_road_object_output_risky_mot = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/mot_for_risky"
_root_feature_stat = "/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/feature_stat"

# -- for 20210120_005 --#
# _root_mirror_images = "/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20210120/20210120_005/frames_png"
# _root_mirror_output_video = "/home/appuser/data/board"


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
        tracker.load_frames(root_images=_root_mirror_images, frame_num_start=-1, frame_num_end=-1)
        hooks = [
            MirrorDetection(0.3, 0.3),
            Recorder(),
            ImageWriter(root_output_images=_root_mirror_output_images),
            VideoWriterForTracking(root_output_video=_root_mirror_output_video, do_resize=True, fps=30.0, size=(3840, 2160)),
            MOTWriter(root_output_mot=_root_mirror_output_mot)
        ]
        tracker.register_hooks(hooks)
        tracker.track()

    def test_exp_images_2_video(self):
        tracker = IoUTracker()
        tracker.load_frames(root_images=_root_mirror_images, frame_num_start=-1, frame_num_end=-1)
        hooks = [
            Recorder(),
            VideoWriterForTracking(root_output_video=_root_mirror_output_video, do_resize=True, fps=30.0, size=(3840, 2160)),
        ]
        tracker.register_hooks(hooks)
        tracker.track()

    def test_road_object_tracking(self):
        tracker = IoUTracker()
        tracker.load_frames(root_images=_root_road_object_images, frame_num_start=-1, frame_num_end=-1)
        hooks = [
            RoadObjectDetection(score_thre = 0.17),
            BoxCoordinateNormalizer(),
            ApproachingInitializer(),
            FeatureMatcher(interval=6, box_remove_margin=0.1),
            # TransformedMidpointCalculator(),
            # TransformedMidpointDifferencer(),
            HorizontalMovementCounter(right_trend_is_approaching=False),
            # AreaCalculator(),
            # MidpointCalculator(),
            # WidthAndHeihtCalculator(),
            RiskyJudger(),
            Recorder(),
            ImageWriter(root_output_images=_root_road_object_output_images),
            ImageWriterForApproaching(root_output_images=_root_road_object_output_approaching_images),
            VideoWriterForTracking(root_output_video=_root_road_object_output_video, do_resize=True, fps=30.0),
            VideoWriterForApproaching(root_output_video=_root_road_object_output_approaching_video, do_resize=True, fps=30.0),
            VideoWriterForMatching(root_output_video=_root_road_object_output_feature_matching_video, size=(1024,512) ,do_resize=True, fps=30.0),
            MOTWriter(root_output_mot=_root_road_object_output_mot, labels=["vehicle"]),
            MOTWriterForApproaching(root_output_mot=_root_road_object_output_approaching_mot),
            MOTWriterForRisky(root_output_mot=_root_road_object_output_risky_mot),
        ]
        tracker.register_hooks(hooks)
        tracker.track()
    
    def test_road_object_tracking_transdiff(self):
        tracker = IoUTracker()
        tracker.load_frames(root_images=_root_road_object_images, frame_num_start=-1, frame_num_end=-1)
        hooks = [
            RoadObjectDetection(score_thre = 0.17),
            BoxCoordinateNormalizer(),
            ApproachingInitializer(),
            FeatureMatcher(interval=6, box_remove_margin=0.1),
            # TransformedMidpointCalculator(),
            TransformedMidpointDifferencer(),
            # HorizontalMovementCounter(right_trend_is_approaching=False),
            # AreaCalculator(),
            # MidpointCalculator(),
            # WidthAndHeihtCalculator(),
            RiskyJudger(),
            Recorder(),
            ImageWriter(root_output_images=_root_road_object_output_images),
            ImageWriterForApproaching(root_output_images=_root_road_object_output_approaching_images),
            VideoWriterForTracking(root_output_video=_root_road_object_output_video, do_resize=True, fps=30.0),
            VideoWriterForApproaching(root_output_video=_root_road_object_output_approaching_video, do_resize=True, fps=30.0),
            VideoWriterForMatching(root_output_video=_root_road_object_output_feature_matching_video, size=(1024,512) ,do_resize=True, fps=30.0),
            MOTWriter(root_output_mot=_root_road_object_output_mot, labels=["vehicle"]),
            MOTWriterForApproaching(root_output_mot=_root_road_object_output_approaching_mot),
            MOTWriterForRisky(root_output_mot=_root_road_object_output_risky_mot),
            FeatureStatWriter(root_output_feature_stat=_root_feature_stat),
        ]

        tracker.register_hooks(hooks)
        tracker.track()



class TestNullTracker(unittest.TestCase):

    def test_null_road_object_tracking(self):
        tracker = NullTracker()
        tracker.load_frames(root_images=_root_road_object_images, frame_num_start=-1, frame_num_end=-1)
        hooks = [
            MOTReader(path_mot_txt=_path_input_mot_txt),
            BoxCoordinateNormalizer(),
            ApproachingInitializer(),
            FeatureMatcher(interval=30),
            TransformedMidpointCalculator(),
            # HorizontalMovementCounter(right_trend_is_approaching=True),
            # AreaCalculator(),
            # MidpointCalculator(),
            # WidthAndHeihtCalculator(),
            RiskyJudger(),
            Recorder(),
            ImageWriter(root_output_images=_root_road_object_output_images),
            ImageWriterForApproaching(root_output_images=_root_road_object_output_approaching_images),
            VideoWriterForTracking(root_output_video=_root_road_object_output_video, do_resize=True, fps=30.0),
            VideoWriterForApproaching(root_output_video=_root_road_object_output_approaching_video, do_resize=True, fps=30.0),
            VideoWriterForMatching(root_output_video=_root_road_object_output_feature_matching_video, size=(1024,512) ,do_resize=True, fps=30.0),
            # MOTWriter(root_output_mot=_root_road_object_output_mot)
            MOTWriterForApproaching(root_output_mot=_root_road_object_output_approaching_mot),
            MOTWriterForRisky(root_output_mot=_root_road_object_output_risky_mot)
        ]
        tracker.register_hooks(hooks)
        tracker.track()

import logging
import datetime
import os
import cv2

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker, NullTracker
from pipedet.solver.hooks import MirrorDetection, RoadObjectDetection, MOTReader, BoxCoordinateNormalizer, ApproachingInitializer, FeatureMatcher, TransformedMidpointCalculator, TransformedMidpointDifferencer, HorizontalMovementCounter, AreaCalculator, MidpointCalculator, WidthAndHeihtCalculator, RiskyJudger, Recorder, ImageWriter, ImageWriterForApproaching, VideoWriterForTracking, VideoWriterForApproaching, VideoWriterForMatching, MOTWriter, MOTWriterForApproaching, MOTWriterForRisky, FeatureStatWriter
from pipedet.structure.large_image import LargeImage
from pipedet.solver.get_mirror_seq import MirrorSeq

_root_cropped_images = "" # TODO designate

_root_road_object_output_tracking_images = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/frames_for_tracking"
_root_road_object_output_approaching_images = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/frames_for_approaching"
_root_road_object_output_video_for_tracking = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/video_for_tracking"
_root_road_object_output_approaching_video = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/video_for_approaching"
_root_road_object_output_feature_matching_video = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/video_for_feature_matching"
_root_road_object_output_tracking_mot = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/mot_for_tracking"
_root_road_object_output_approaching_mot = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/mot_for_approaching"
_root_road_object_output_risky_mot = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/mot_for_risky"
_root_feature_stat = "/home/appuser/src/pipedet/simple_demonstration/output_risky_including_road_object_recognition/feature_stat"

if __name__ == '__main__':
    tracker = IoUTracker()
    tracker.load_frames(root_images=_root_cropped_images, frame_num_start=-1, frame_num_end=-1)
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
        ImageWriter(_root_road_object_images=_root_road_object_output_tracking_images),
        ImageWriterForApproaching(root_output_images=_root_road_object_output_approaching_images),
        VideoWriterForTracking(root_output_video=_root_road_object_output_video_for_tracking, do_resize=True, fps=30.0),
        VideoWriterForApproaching(root_output_video=_root_road_object_output_approaching_video, do_resize=True, fps=30.0),
        VideoWriterForMatching(root_output_video=_root_road_object_output_feature_matching_video, size=(1024,512) ,do_resize=True, fps=30.0),
        MOTWriter(root_output_mot=_root_road_object_output_tracking_mot, labels=["vehicle"]),
        MOTWriterForApproaching(root_output_mot=_root_road_object_output_approaching_mot),
        MOTWriterForRisky(root_output_mot=_root_road_object_output_risky_mot),
        FeatureStatWriter(root_output_feature_stat=_root_feature_stat),
    ]
    tracker.register_hooks(hooks)
    tracker.track()
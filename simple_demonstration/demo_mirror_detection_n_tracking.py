import logging
import datetime
import os
import cv2

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker, NullTracker
from pipedet.solver.hooks import MirrorDetection, RoadObjectDetection, MOTReader, BoxCoordinateNormalizer, ApproachingInitializer, FeatureMatcher, TransformedMidpointCalculator, HorizontalMovementCounter, AreaCalculator, MidpointCalculator, WidthAndHeihtCalculator, RiskyJudger, Recorder, ImageWriter, ImageWriterForApproaching, VideoWriterForTracking, VideoWriterForApproaching, VideoWriterForMatching, MOTWriter, MOTWriterForApproaching, MOTWriterForRisky
from pipedet.structure.large_image import LargeImage
from pipedet.solver.get_mirror_seq import MirrorSeq

"""
Mirror tracking demonstration.
"""


_root_in_vehicle_frames = '/home/appuser/src/pipedet/simple_demonstration/demo_in_vehicle_frame'

_root_output_video_for_tracking = '/home/appuser/src/pipedet/simple_demonstration/output_mirror_tracking/video_for_tracking'

_root_output_images_for_tracking = '/home/appuser/src/pipedet/simple_demonstration/output_mirror_tracking/images_for_tracking'

_root_output_mot_for_tracking = '/home/appuser/src/pipedet/simple_demonstration/output_mirror_tracking/mot_for_tracking'

if __name__ == '__main__':
    tracker = IoUTracker()
    tracker.load_frames(root_images=_root_in_vehicle_frames, frame_num_start=-1, frame_num_end=-1)
    hooks = [
        MirrorDetection(0.3, 0.3),
        Recorder(),
        ImageWriter(root_output_images=_root_output_images_for_tracking),
        VideoWriterForTracking(root_output_video=_root_output_video_for_tracking, do_resize=True, fps=30.0, size=(3840, 2160)),
        MOTWriter(root_output_mot=_root_output_mot_for_tracking)
    ]
    tracker.register_hooks(hooks)
    tracker.track()
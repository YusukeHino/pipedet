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
Mirror cropping after mirror detection and tracking.
"""


_root_in_vehicle_frames = '/home/appuser/src/pipedet/simple_demonstration/demo_in_vehicle_frame'

_root_input_mot_for_tracking = '/home/appuser/src/pipedet/simple_demonstration/output_mirror_tracking/mot_for_tracking'

_root_output_images_for_cropping = '/home/appuser/src/pipedet/simple_demonstration/demo_mirror_seq'

_mir_num = 1 # track id of mirror that you would like to crop

if __name__ == '__main__':

    mirror_seq = MirrorSeq(_root_in_vehicle_frames)
    mirror_seq.load_w_mot_txt(os.path.join(_root_input_mot_for_tracking, 'gt.txt'))
    mirror_seq.crop_n_get_mirror_seq(_mir_num)
    padded_mir_num = str(_mir_num).zfill(5)
    root_seqs = os.path.join(_root_output_images_for_cropping, padded_mir_num)
    root_cropped_images = os.path.join(root_seqs, 'frames')
    if len(mirror_seq.mapped_mirror_seq) == 0:
        logging.info(f'Mirror cropping was terminated at mirror seq: {_mir_num-1}')
        break
    os.makedirs(root_cropped_images, exist_ok=True)
    for frame_num, cropped_image in mirror_seq.mapped_mirror_seq.items():
        path_output_image = os.path.join(root_cropped_images, str(frame_num).zfill(4) + ".png")
        cv2.imwrite(path_output_image, cropped_image.image)
        assert os.path.isfile(path_output_image), f'{path_output_image} was not generated.'
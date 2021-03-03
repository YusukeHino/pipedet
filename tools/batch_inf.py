
import logging
import datetime
import os
import cv2

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker, NullTracker
from pipedet.solver.hooks import MirrorDetection, RoadObjectDetection, MOTReader, BoxCoordinateNormalizer, ApproachingInitializer, FeatureMatcher, TransformedMidpointCalculator, HorizontalMovementCounter, AreaCalculator, MidpointCalculator, WidthAndHeihtCalculator, RiskyJudger, Recorder, ImageWriter, ImageWriterForApproaching, VideoWriterForTracking, VideoWriterForApproaching, VideoWriterForMatching, MOTWriter, MOTWriterForApproaching, MOTWriterForRisky
from pipedet.structure.large_image import LargeImage
from pipedet.solver.get_mirror_seq import MirrorSeq

_ROOT_MIRROR_EXP = '/home/appuser/src/pipedet/experiments'
assert os.path.isdir(_ROOT_MIRROR_EXP)

_IS_NEW_EXP = False

_ROOT_MIRROR_IMAGES = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20210120'
assert os.path.isdir(_ROOT_MIRROR_IMAGES)

_IF_POSSIBLE_SKIP_MIR_RECOGNITION = True

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    path_exp = os.path.join(_ROOT_MIRROR_EXP, 'exp')

    if _IS_NEW_EXP:
        if os.path.isdir(path_exp):
            with open(os.path.join(path_exp, 'exp_date.txt'), "r") as exp_date_file:
                cnt = 0
                for line in exp_date_file: # per-object
                    if cnt == 0:
                        exp_yyyymmddhhmmss = line.rstrip('\r\n')
                    cnt += 1        
            os.rename(path_exp, path_exp + '.' + exp_yyyymmddhhmmss)
        else:
            os.mkdir(path_exp)
            with open(os.path.join(path_exp, 'exp_date.txt'), "wt") as exp_date_file:
                exp_date_file.write(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))


    for seq_name in sorted(os.listdir(_ROOT_MIRROR_IMAGES)):
        logging.info(f'Seq: {seq_name}')
        yyyymmdd = seq_name[:8]
        seq_num_str = seq_name[9:]
        try:
            seq_num = int(seq_num_str)
        except ValueError:
            # logger = logging.getLogger(__name__)
            # logger.error(f"Cannot extract frame_number from frame_name: {frame_name}")
            raise

        roots_for_input = dict(
            root_mirror_images = os.path.join(_ROOT_MIRROR_IMAGES, seq_name, 'frames_png')
        )

        roots_for_output = dict(
            root_mirror_output_images = os.path.join(path_exp, seq_name, 'mir_tracking_result', 'frames_for_tracking'),
            root_mirror_output_video = os.path.join(path_exp, seq_name, 'mir_tracking_result', 'video_for_tracking'),
            root_mirror_output_mot = os.path.join(path_exp, seq_name, 'mir_tracking_result', 'mot_for_tracking'),
            root_mirror_seqs = os.path.join(path_exp, seq_name, 'mirror_seqs')
        )

        for root_name, root in roots_for_input.items():
            assert os.path.isdir(root), f'Not exists: {root} for {seq_name}'
            
        for root_name, root in roots_for_output.items():
            os.makedirs(root, exist_ok=True)

        tracker = IoUTracker()
        tracker.load_frames(root_images=roots_for_input['root_mirror_images'], frame_num_start=-1, frame_num_end=-1)
        hooks = [
            MirrorDetection(0.3, 0.3),
            Recorder(),
            ImageWriter(root_output_images=roots_for_output['root_mirror_output_images']),
            VideoWriterForTracking(root_output_video=roots_for_output['root_mirror_output_video'], do_resize=True, fps=30.0, size=(3840, 2160)),
            MOTWriter(root_output_mot=roots_for_output['root_mirror_output_mot'])
        ]
        tracker.register_hooks(hooks)

        if _IF_POSSIBLE_SKIP_MIR_RECOGNITION and (len(tracker.loader) == len(os.listdir(roots_for_output['root_mirror_output_images']))):
           logging.info('Mirror recoginition stage was skipped.')
        else: 
            tracker.track()

        for mir_num in range(1, 501):
            logging.info(f'Seq: {seq_name}, Mirror seq: {mir_num}')
            mirror_seq = MirrorSeq(roots_for_input['root_mirror_images'])
            mirror_seq.load_w_mot_txt(os.path.join(roots_for_output['root_mirror_output_mot'], 'gt.txt'))
            mirror_seq.crop_n_get_mirror_seq(mir_num)
            padded_mir_num = str(mir_num).zfill(5)
            root_seqs = os.path.join(roots_for_output['root_mirror_seqs'], padded_mir_num)
            root_cropped_images = os.path.join(root_seqs, 'frames')
            if len(mirror_seq.mapped_mirror_seq) == 0:
                logging.info(f'Mirror cropping was terminated at mirror seq: {mir_num-1}')
                break
            os.makedirs(root_cropped_images, exist_ok=True)
            for frame_num, cropped_image in mirror_seq.mapped_mirror_seq.items():
                path_output_image = os.path.join(root_cropped_images, str(frame_num).zfill(4) + ".png")
                cv2.imwrite(path_output_image, cropped_image.image)
                assert os.path.isfile(path_output_image), f'{path_output_image} was not generated.'
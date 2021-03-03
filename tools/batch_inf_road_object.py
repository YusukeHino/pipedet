
import logging
import datetime
import os
import cv2

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.solver.iou_tracker import IoUTracker, NullTracker
from pipedet.solver.hooks import MirrorDetection, RoadObjectDetection, MOTReader, BoxCoordinateNormalizer, ApproachingInitializer, FeatureMatcher, TransformedMidpointCalculator, TransformedMidpointDifferencer, HorizontalMovementCounter, AreaCalculator, MidpointCalculator, WidthAndHeihtCalculator, RiskyJudger, Recorder, ImageWriter, ImageWriterForApproaching, VideoWriterForTracking, VideoWriterForApproaching, VideoWriterForMatching, MOTWriter, MOTWriterForApproaching, MOTWriterForRisky, FeatureStatWriter
from pipedet.structure.large_image import LargeImage
from pipedet.solver.get_mirror_seq import MirrorSeq

logging.basicConfig(level=logging.INFO)

_ROOT_MIRROR_EXP = '/home/appuser/src/pipedet/experiments'
assert os.path.isdir(_ROOT_MIRROR_EXP)

_IS_NEW_RO_EXP = True

now_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# suffix = 'transmiddiff_box_margin_0p1_distance_thre_10000'
suffix = 'transmiddiff_ivl_6_box_margin_0p1_distance_thre_2000'

if __name__ == '__main__':

    path_exp = os.path.join(_ROOT_MIRROR_EXP, 'exp')

    if _IS_NEW_RO_EXP:
        ro_exp_all_acc_pr_rc_tp_tn_fp_fn = os.path.join(path_exp, 'ro_exp_all_acc_pr_rc_tp_tn_fp_fn.txt')
        if os.path.isfile(ro_exp_all_acc_pr_rc_tp_tn_fp_fn):
            os.rename(ro_exp_all_acc_pr_rc_tp_tn_fp_fn, ro_exp_all_acc_pr_rc_tp_tn_fp_fn + '.' + suffix)

    for seq_name in sorted(os.listdir(path_exp)): # for each seq
        logging.info(f'Seq: {seq_name}')
        yyyymmdd = seq_name[:8]
        seq_num_str = seq_name[9:]
        try:
            seq_num = int(seq_num_str)
        except ValueError as err:
            logging.info(f'{err}')
            # logger = logging.getLogger(__name__)
            # logger.error(f"Cannot extract frame_number from frame_name: {frame_name}")
            continue

        roots_for_input = dict(
            root_road_object_images = os.path.join(path_exp, seq_name, 'mirror_seqs')
        )

        for root_name, root in roots_for_input.items():
            assert os.path.isdir(root), f'Not exists: {root} for {seq_name}'
        
        # if seq_name < '20210120_014': # TODO: REMOVE
        #     continue

        for mir_seq_name in sorted(os.listdir(roots_for_input['root_road_object_images'])): # for each mirror seq
            try:
                mir_seq_num = int(mir_seq_name)
            except ValueError:
                raise
            logging.info(f'Seq: {seq_name}, Mirror seq: {mir_seq_num}')

            # if mir_seq_num < 3: # TODO: remove
            #     continue
            
            path_mirror_seq = os.path.join(roots_for_input['root_road_object_images'], mir_seq_name)
            assert os.path.isdir(path_mirror_seq)
            root_cropped_images = os.path.join(path_mirror_seq, 'frames')
            assert os.path.isdir(root_cropped_images)
            if len(os.listdir(root_cropped_images)) < 30:
                continue

            path_ro_exp = os.path.join(path_mirror_seq, 'ro_exp')
            if _IS_NEW_RO_EXP:
            # if _IS_NEW_RO_EXP and (seq_name != '20210120_014'): # TODO: Remove
                if os.path.isdir(path_ro_exp):
                    if os.path.isfile(os.path.join(path_ro_exp, 'ro_exp_date.txt')):
                        with open(os.path.join(path_ro_exp, 'ro_exp_date.txt'), "r") as ro_exp_date_file:
                            cnt = 0
                            for line in ro_exp_date_file: # per-object
                                if cnt == 0:
                                    exp_yyyymmddhhmmss = line.rstrip('\r\n')
                                cnt += 1        
                        os.rename(path_ro_exp, path_ro_exp + '.' + exp_yyyymmddhhmmss + '_' + suffix)
                    else:
                        os.rename(path_ro_exp, path_ro_exp + '.' + suffix)
                try:
                    os.mkdir(path_ro_exp)
                except FileExistsError:
                    pass

                with open(os.path.join(path_ro_exp, 'ro_exp_date.txt'), "wt") as ro_exp_date_file:
                    ro_exp_date_file.write(now_datetime)
            else:
                try:
                    os.mkdir(path_ro_exp)
                except FileExistsError:
                    pass

                with open(os.path.join(path_ro_exp, 'ro_exp_date.txt'), "wt") as ro_exp_date_file:
                    ro_exp_date_file.write(now_datetime)

            roots_for_output = dict(
                root_road_object_output_images = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'frames_for_tracking'),
                root_road_object_output_video = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'video_for_tracking'),
                root_road_object_output_mot = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'mot_for_tracking'),
                root_road_object_output_approaching_images = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'frames_for_approaching'),
                root_road_object_output_approaching_video = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'video_for_approaching'),
                root_road_object_output_approaching_mot = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'mot_for_approaching'),
                root_road_object_output_feature_matching_video = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'video_for_feature_matching'),
                root_road_object_output_risky_mot = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'mot_for_risky'),
                root_road_object_feature_stat = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'feature_stat'),
            )

            for root_name, root in roots_for_output.items():
                os.makedirs(root, exist_ok=True)

            tracker = IoUTracker()
            tracker.load_frames(root_images=root_cropped_images, frame_num_start=-1, frame_num_end=-1)
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
                ImageWriter(root_output_images=roots_for_output['root_road_object_output_images']),
                ImageWriterForApproaching(root_output_images=roots_for_output['root_road_object_output_approaching_images']),
                VideoWriterForTracking(root_output_video=roots_for_output['root_road_object_output_video'], do_resize=True, fps=30.0),
                VideoWriterForApproaching(root_output_video=roots_for_output['root_road_object_output_approaching_video'], do_resize=True, fps=30.0),
                VideoWriterForMatching(root_output_video=roots_for_output['root_road_object_output_feature_matching_video'], size=(1024,512) ,do_resize=True, fps=30.0),
                MOTWriter(root_output_mot=roots_for_output['root_road_object_output_mot'], labels=["vehicle"]),
                MOTWriterForApproaching(root_output_mot=roots_for_output['root_road_object_output_approaching_mot']),
                MOTWriterForRisky(root_output_mot=roots_for_output['root_road_object_output_risky_mot']),
                FeatureStatWriter(root_output_feature_stat=roots_for_output['root_road_object_feature_stat']),
            ]
            tracker.register_hooks(hooks)
            tracker.track()
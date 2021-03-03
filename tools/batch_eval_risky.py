
import logging
import datetime
import os
import cv2
import numpy as np


logging.basicConfig(level=logging.INFO)

_ROOT_MIRROR_EXP = '/home/appuser/src/pipedet/experiments'
assert os.path.isdir(_ROOT_MIRROR_EXP)

_ROOT_REAL_GT = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/trimed/20210120'
assert os.path.isdir(_ROOT_REAL_GT)

DO_FEATURE_STAT = True

if __name__ == '__main__':
    """
    for each seq:
        read mot for risky(real gt)
        for each mir_seq:
            read mot for risky(inf)
        merge all risky and caluculate risky
        calculate Precision-Recall
        Write precision-recall

    Compile all score
    """

    path_exp = os.path.join(_ROOT_MIRROR_EXP, 'exp')

    score_list = []

    seq_list = [f for f in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, f))]

    if DO_FEATURE_STAT:
        feature_stat_list = []

    for seq_name in sorted(seq_list): # for each seq
        logging.info(f'Seq: {seq_name}')
        yyyymmdd = seq_name[:8]
        seq_num_str = seq_name[9:]
        try:
            seq_num = int(seq_num_str)
        except ValueError:
            raise



        ### Read mot for risky(real gt) ###
        path_real_gt_mot = os.path.join(_ROOT_REAL_GT, seq_name, 'real_gt_for_risky', 'gt.txt')
        assert os.path.isfile(path_real_gt_mot), f'Real gt mot file cannot be found: {path_real_gt_mot}'

        real_gt_for_risky = np.full(2000, False, dtype=bool)

        max_frame_num = 1
        with open(path_real_gt_mot, "r") as mot_txt:
            for line in mot_txt: # per-object
                line = line.rstrip('\r\n')
                listed_line = line.split(',')
                frame_num = int(listed_line[0])
                real_gt_for_risky[frame_num] = True if int(listed_line[7]) == 2 else False
                max_frame_num = max(frame_num, max_frame_num)



        ### Read mot for risky(inf) ###

        root_mir_seqs = os.path.join(path_exp, seq_name, 'mirror_seqs')

        assert os.path.isdir(root_mir_seqs), f'Not exists: {root_mir_seqs}'

        inf_for_risky = np.full(2000, False, dtype=bool)

        for mir_seq_name in sorted(os.listdir(root_mir_seqs)): # for each mirror seq
            try:
                mir_seq_num = int(mir_seq_name)
            except ValueError:
                raise

            logging.info(f'Seq: {seq_name}, Mirror seq: {mir_seq_num}')
            
            path_mirror_seq = os.path.join(root_mir_seqs, mir_seq_name)
            assert os.path.isdir(path_mirror_seq)
            root_cropped_images = os.path.join(path_mirror_seq, 'frames')
            assert os.path.isdir(root_cropped_images)

            path_ro_exp = os.path.join(path_mirror_seq, 'ro_exp')

            if not os.path.isdir(path_ro_exp):
                continue

            root_road_object_output_risky_mot = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'mot_for_risky')
            assert os.path.isdir(root_road_object_output_risky_mot)
            path_road_object_output_risky_mot = os.path.join(root_road_object_output_risky_mot, 'gt.txt')
            assert os.path.isfile(path_road_object_output_risky_mot)
            
            with open(path_road_object_output_risky_mot, "r") as mot_txt:
                for line in mot_txt: # per-object
                    line = line.rstrip('\r\n')
                    listed_line = line.split(',')
                    frame_num = int(listed_line[0])
                    if not inf_for_risky[frame_num]:
                        inf_for_risky[frame_num] = True if int(listed_line[7]) == 2 else False

            ### Read feature stat ### format: [iter_num, is_risky, num_previous_keypoints, num_current_keypoints, is_homograpy_found]
            if DO_FEATURE_STAT:
                root_road_object_output_feature_stat = os.path.join(path_ro_exp, seq_name, mir_seq_name, 'feature_stat')
                assert os.path.isdir(root_road_object_output_feature_stat)
                path_road_object_output_feature_stat = os.path.join(root_road_object_output_feature_stat, 'feature_stat.txt')
                assert os.path.isfile(path_road_object_output_feature_stat)
                with open(path_road_object_output_feature_stat, "r") as feature_stat_txt:
                    for line in feature_stat_txt: # per-object
                        line = line.rstrip('\r\n')
                        listed_line = line.split(',')
                        feature_stat_list.append(listed_line[2:])

            


        ### Calculate Precision Recall ###
        num_tp = 0
        num_tn = 0
        num_fp = 0
        num_fn = 0

        for frame_num in range(1, max_frame_num+1):
            if inf_for_risky[frame_num]: # positive
                if real_gt_for_risky[frame_num]:
                    num_tp += 1
                else:
                    num_fp += 1
            else: # negative
                if real_gt_for_risky[frame_num]:
                    num_fn += 1
                else:
                    num_tn += 1

        accuracy = (num_tp + num_tn) / (num_tp + num_fp + num_tn + num_fn)
        assert accuracy <= 1.0
        
        try:
            precision = num_tp / (num_tp + num_fp)
        except ZeroDivisionError:
            precision = -1.0
            logging.info(f'Precision is NA, because tp + fp = 0. There is no positive.')

        try:
            recall = num_tp / (num_tp + num_fn)
        except ZeroDivisionError:
            recall = -1.0
            logging.info(f'Recall is NA, because tp + fn = 0. There is no ground-true positive.')
        

        path_road_object_output_acc_pr_rc_tp_tn_fp_fn = os.path.join(path_exp, seq_name, 'acc_pr_rc_tp_tn_fp_fn.txt')
     
        with open(path_road_object_output_acc_pr_rc_tp_tn_fp_fn, "wt") as out_txt:
            line_as_list = [seq_name, accuracy, precision, recall, num_tp, num_tn, num_fp, num_fn]
            line_as_str = ','.join([str(_) for _ in line_as_list]) + "\n"
            score_list.append(line_as_str)
            out_txt.write(line_as_str)


    ### Compile all score ###

    ro_exp_all_acc_pr_rc_tp_tn_fp_fn = os.path.join(path_exp, 'ro_exp_all_acc_pr_rc_tp_tn_fp_fn.txt')

    with open(ro_exp_all_acc_pr_rc_tp_tn_fp_fn, 'wt') as out_txt:
        for line_as_str in score_list:
            out_txt.write(line_as_str)

    if DO_FEATURE_STAT:
        ro_exp_all_num_previous_keypoints_num_current_keypoints_is_homograpy_found = os.path.join(path_exp, 'num_previous_keypoints_num_current_keypoints_is_homograpy_found.txt')
        with open(ro_exp_all_num_previous_keypoints_num_current_keypoints_is_homograpy_found, 'wt') as out_txt:
            for feature_stat in feature_stat_list:
                line_as_str = ','.join([str(_) for _ in feature_stat]) + "\n"
                out_txt.write(line_as_str)
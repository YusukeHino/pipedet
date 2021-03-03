
import logging
import datetime
import os
import cv2
from collections import defaultdict
from collections import OrderedDict
from typing import List, Tuple, Optional, Union, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from pipedet.data.image_loader import TrackingFrameLoader
from pipedet.data.dataset import Dataset
from pipedet.structure.large_image import LargeImage

ROOT_SEPARATED = '/home/appuser/data/for_rsm_detection/separated'

ROOT_ANNOTATION = '/home/appuser/data/for_rsm_detection/separated/annotations'

DATA_ROOT = "/home/appuser/data"

CVAT_XML_ROOT = "/home/appuser/data/for_rsm_detection/separated/annotations"

ROOT_RESULT = "/home/appuser/src/pipedet/results_for_mirror_det"

FIRST_THRE = 0.3

SECOND_THRE = 0.3

IOU_THRE = 0.5 # for eval

# class ImageLoader:
#     IMAGE_EXTENTIONS = ['.JPG', '.jpg', '.PNG', '.png']
#     def __init__(self, root_separated: str):
#         self.root_separated = root_separated
#         self.full_paths_to_frame: OrderedDict[int, str] = OrderedDict()
#         assert os.path.isdir(self.root_separated)
#         image_count = 0
#         for size_n_type in sorted(os.listdir(self.root_separated)):
#             # size_n_type is ex.) 4000_3000_for_train
#             if not size_n_type.endswith('_for_test'):
#                 continue
#             size_str = size_n_type[:-len('_for_test')]
#             root_images = os.path.join(self.root_separated, size_n_type)

#             for fname in sorted(os.listdir(root_images)):
#                 if not (os.path.splitext(fname)[1] in self.IMAGE_EXTENTIONS):
#                     continue
#                 full_path_to_frame = os.path.join(root_images, fname)
#                 self.full_paths_to_frame[image_count] = full_path_to_frame
#                 image_count += 1

#         self.frame_num_iter = 0
#         self.frame_num_end = image_count
    
#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.frame_num_iter > self.frame_num_end: # TODO
#             raise StopIteration

#         frame_path = self.full_paths_to_frame[self.frame_num_iter]
#         image = cv2.imread(frame_path)
#         large_image = LargeImage(image)
#         self.frame_num_iter += 1
#         return large_image

#     def __len__(self):
#         return len(self.full_paths_to_frame)

# class AnnotationLoader:
#     def __init__(self, root_annotation: str):
#         self.root_annotation = root_annotation

def box_iou(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left(x1), up(y1), right(x2), bottom(y2)
    HINO x2 and y2 are bigger than x1 and y1, respectively.
    '''
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])

    return float(s_intsec)/(s_a + s_b -s_intsec)

def iou_matching(boxes_a_arg: np.ndarray, boxes_b_arg: np.ndarray, iou_thre: float=0.8) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Args:
    Retrun:
        (a_ind, b_ind) (Tuple[np.ndarray, np.ndarray]):
    '''
    # TODO: set iou thre

    boxes_a = np.array(boxes_a_arg)
    boxes_b = np.array(boxes_b_arg)
    # Get IoU i.e. cost matrix
    iou_matrix = np.zeros((len(boxes_a),len(boxes_b)),dtype=np.float32)
    for a, box_a in enumerate(boxes_a):
        for b, box_b in enumerate(boxes_b):
            iou_matrix[a, b] = -1 * box_iou(box_a, box_b)

    a_ind, b_ind = linear_sum_assignment(iou_matrix)

    a_ind_filtered = np.zeros(0, int)
    b_ind_filtered = np.zeros(0, int)
    for a, b in zip(a_ind, b_ind):
        if iou_matrix[a, b] * (-1) >= iou_thre:
            a_ind_filtered = np.append(a_ind_filtered, a)
            b_ind_filtered = np.append(b_ind_filtered, b)

    return a_ind_filtered, b_ind_filtered

if __name__ == '__main__':
    print('Loading.')
    dataset = Dataset(DATA_ROOT, tag_to_use=["test"])
    xml_files = [os.path.join(CVAT_XML_ROOT, xml_dir_name, "annotations.xml") for xml_dir_name in os.listdir(CVAT_XML_ROOT)]
    dataset.load_cvat_xmls(xml_files)
    print('load_cvat_xmls done.')
    dataset.set_tag_from_rel_path()
    tp_all = 0
    fp_all = 0
    fn_all = 0
    image_count = 0
    for large_image in dataset.images:
        if large_image.tag == 'TEST':
            print(f'Detecting: {large_image.rel_path}\nimage_count {str(image_count)}')
            gt_bboxes = large_image.bboxes
            large_image.bboxes = []
            large_image.class_confidences = []
            #TODO: config
            large_image.pipe_det(first_thre=FIRST_THRE, second_thre=SECOND_THRE, patch_width=1024, patch_height=1024)
            # large_image.inference_of_objct_detection(server='FASTER-RCNN')
            # large_image.adapt_class_confidence_thre(0.5)
            # cv2.imwrite('/home/appuser/data/board/demo_feng_rsm_det_client_depict.png', large_image.image_drawn)
            detections = large_image.bboxes
            print(f'Evaluating: {large_image.rel_path}\n')
            gt_bboxes_array: np.ndarray = np.array(gt_bboxes, dtype=int)
            detections_array: np.ndarray = np.array(detections, dtype=int)
            matched_boxes_indxes, matched_detections_indxes = iou_matching(gt_bboxes_array, detections_array, iou_thre=IOU_THRE)
            matched_detections = detections_array[matched_detections_indxes]
            unmatched_detections = np.delete(detections_array, matched_detections_indxes, 0)

            tp = len(matched_detections)
            fp = len(unmatched_detections)
            fn = len(gt_bboxes_array) - len(matched_detections)

            tp_all += tp
            fp_all += fp
            fn_all += fn
            image_count += 1
            # if image_count > 5: #TODO: Remove
            #     break
    print(f'Final Caluculating')
    precision = tp_all / (tp_all+fp_all)
    recall = tp_all / (tp_all+fn_all)

    path_result = os.path.join(ROOT_RESULT, 'pr_rc_tp_fp_fn.txt')
    with open(path_result, 'wt') as out_txt:
        line_as_list = [precision, recall, tp_all, fp_all, fn_all]
        line_as_str = ','.join([str(_) for _ in line_as_list]) + "\n"
        out_txt.write(line_as_str)
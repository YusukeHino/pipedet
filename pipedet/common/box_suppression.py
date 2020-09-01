
from typing import List, Tuple

import numpy as np


def suppress_crop_locations(crop_locations: List[List[int]], iou_thre: float=0.5) -> List[List[int]]:
    suppressed_crop_locations = crop_locations.copy()
    while True:
        has_been_suppressed = False
        for index, crop_location in enumerate(crop_locations):
            crop_locations = suppressed_crop_locations.copy()
            need_to_del = []
            if index == len(crop_locations) - 1:
                continue
            for index1_5, crop_location2 in enumerate(crop_locations[index+1:]):
                index2 = index + 1  + index1_5
                if get_iou(crop_location, crop_location2) >= iou_thre:
                    need_to_del.append(index2)
                    has_been_suppressed = True
            for index_to_del in sorted(need_to_del, reverse=True):
                del suppressed_crop_locations[index_to_del]
        if not has_been_suppressed:
            break
    return suppressed_crop_locations


def non_max_suppression_fast(boxes_arg: List[List[int]], confidences_arg: List[Tuple[float, float]], overlapThresh: float=0.5):

    if not len(boxes_arg):
        return [], []
    
    boxes = np.array(boxes_arg)
    confidences = np.array(confidences_arg)

	# if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
	# initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = confidences[:,1]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    # keep looping while some indexes still remain in the indexes
    # list
    while len(order) > 0:
        i = order[0]
        pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
		# compute the width and height of the bounding box
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
		# compute the ratio of overlap
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
		# delete all indexes from the index list that have
        inds = np.where(ovr <= overlapThresh)[0]
        order = order[inds + 1]

    return boxes[pick].astype("int").tolist(), confidences[pick].tolist()


def get_iou(box1: List[int], box2: List[int]) -> float:
    """
    Args:
        box1(List[int]): bounding box in format x1,y1,x2,y2
        box2(List[int]): bounding box in format x1,y1,x2,y2

    Returns:
        float: intersection-over-onion of bbox1, bbox2
    """

    box1 = [float(x) for x in box1]
    box2 = [float(x) for x in box2]

    (x0_1, y0_1, x1_1, y1_1) = box1
    (x0_2, y0_2, x1_2, y1_2) = box2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0.

    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union
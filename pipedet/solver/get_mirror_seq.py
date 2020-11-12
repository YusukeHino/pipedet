
import logging
import time
import weakref
from collections import defaultdict
from typing import List, Tuple, Optional, Union, Any, DefaultDict, Dict

import numpy as np

from ..data.image_loader import TrackingFrameLoader
from ..structure.large_image import LargeImage, _RawBoxType
from ..solver.hooks import HookBase

class MirrorSeq:
    def __init__(self, root_images: str):
        self.mapped_large_images: Dict[int, LargeImage] = {}
        self.frame_loader = TrackingFrameLoader(root_images=root_images)

    def load_w_mot_txt(self, mot_txt_file: str):
        mapped_annotations = self.get_mapped_annotatons_from_mot_txt(mot_txt_file)

        frame_loader_iter = iter(self.frame_loader)
        while True: # for each frame
            frame_num = frame_loader_iter.frame_num_iter
            annotations = mapped_annotations[frame_num]
            try:
                large_image = next(frame_loader_iter)
            except StopIteration:
                break
            if len(annotations) == 0:
                continue
            large_image.bboxes = []
            large_image.track_ids = []
            for listed_line in annotations: # for each object
                bbox = [float(x) for x in listed_line[2:6]]
                bbox[0] -= 1.
                bbox[1] -= 1.
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bbox = [int(x) for x in bbox]
                large_image.bboxes.append(bbox)
                large_image.track_ids.append(int(listed_line[1]))
            self.mapped_large_images[frame_num] = large_image
                
    def get_mapped_annotatons_from_mot_txt(self, mot_txt_file: str) -> DefaultDict[int, List[List[str]]]:
        with open(mot_txt_file, "r") as mot_txt:
            annotations: DefaultDict[int, List[List[str]]] = defaultdict(lambda: [])
            for line in mot_txt: # per-object
                line = line.rstrip('\r\n')
                listed_line = line.split(',')
                frame_num = int(listed_line[0])
                annotations[frame_num].append(listed_line)
            return annotations

    def crop_n_get_mirror_seq(self, target_tracking_id: int):
        self.mapped_mirror_seq: Dict[int, LargeImage] = {}
        for frame_num, large_image in self.mapped_large_images.items():
            if len(large_image.bboxes) == 0:
                continue
            if target_tracking_id in large_image.track_ids:
                for obj_num, track_id in enumerate(large_image.track_ids):
                    if track_id == target_tracking_id:
                        bbox = large_image.bboxes[obj_num]
                        cropped_mirror = large_image.get_crop(bbox).copy()
                        self.mapped_mirror_seq[frame_num] = LargeImage(cropped_mirror)
        
        frame_nums = list(self.mapped_mirror_seq.keys())
        frame_nums.sort()
        tmp = frame_nums[0] - 1
        for frame_num_element in frame_nums:
            assert tmp + 1 == frame_num_element
            tmp = frame_num_element

    def get_road_objects_from_mot_txt(self, mot_txt_file: str):
        mapped_annotations = self.get_mapped_annotatons_from_mot_txt(mot_txt_file)
        for frame_num, mirror_image in self.mapped_mirror_seq.items():
            annotations = mapped_annotations[frame_num]
            mirror_image.bboxes = []
            mirror_image.track_ids = []
            mirror_image.class_confidences = []
            for listed_line in annotations: # for each object
                bbox = [float(x) for x in listed_line[2:6]]
                bbox[0] -= 1.
                bbox[1] -= 1.
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bbox = [int(x) for x in bbox]
                class_confidence = float(listed_line[6])
                mirror_image.bboxes.append(bbox)
                mirror_image.track_ids.append(int(listed_line[1]))
                mirror_image.class_confidences.append((0.0, class_confidence))
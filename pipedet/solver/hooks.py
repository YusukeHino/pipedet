
import datetime
import itertools
import logging
import os
import tempfile
import time
import copy
from collections import Counter
import zipfile
from typing import List, Tuple, Optional, Union, Any

import cv2


class HookBase:
    """

    """

    def before_track(self):
        """
        Called before the first iteration.
        """
        pass

    def after_track(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class Detection(HookBase):

    def __init__(self, first_thre: float, second_thre: float):
        self.first_thre = first_thre
        self.second_thre = second_thre

    def before_step(self):
        self.tracker.data.pipe_det(first_thre=self.first_thre, second_thre=self.second_thre, patch_width=1024, patch_height=1024)
        self.tracker.state_detections = self.tracker.data.bboxes

class Recorder(HookBase):

    def after_step(self):
        record = copy.deepcopy(self.tracker.data)
        record.track_ids = self.tracker.state_track_ids
        record.bboxes = self.tracker.state_boxes
        self.tracker.record = record

class ImageWriter(HookBase):

    def __init__(self, root_output_images: str):
        self.root_output_images = root_output_images

    def after_step(self):
        image_name = str(self.tracker.iter).zfill(4) + ".jpg"
        self.output_image_path = os.path.join(self.root_output_images, image_name)
        self.tracker.record.clear_patches()
        cv2.imwrite(self.output_image_path, self.tracker.record.image_drawn)

class MOTWriter(HookBase):

    def __init__(self, root_output_mot: str):
        self.path_to_zip = os.path.join(root_output_mot, 'gt.zip')
        self.path_to_txt = os.path.join(root_output_mot, 'gt.txt')
        self.path_to_labels = os.path.join(root_output_mot, 'labels.txt')
        self.lines_as_lists: List[List[Any]] = []

    def after_step(self):
        for track_id, bbox in zip(self.tracker.record.track_ids, self.tracker.record.bboxes):
            line_as_list = [self.tracker.iter + 1, track_id, bbox[0] + 1, bbox[1] + 1, bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0, 1, 1.0]
            self.lines_as_lists.append(line_as_list)

    def after_track(self):
        lines_as_str_list = []
        for line_as_list in self.lines_as_lists:
            line_as_str = ','.join([str(_) for _ in line_as_list]) + "\n"
            lines_as_str_list.append(line_as_str)
        with open(self.path_to_txt, "wt") as out_mot:
            for line_as_str in lines_as_str_list:
                out_mot.write(line_as_str)

        with open(self.path_to_labels, "wt") as out_labels:
            out_labels.write("mirror")

        with zipfile.ZipFile(self.path_to_zip, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
            new_zip.write(self.path_to_txt, arcname='gt/gt.txt')
            new_zip.write(self.path_to_labels, arcname='gt/labels.txt')
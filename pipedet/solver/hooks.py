
import datetime
import itertools
import logging
import os
import tempfile
import time
import copy
from collections import Counter
import zipfile
from collections import defaultdict
from typing import List, Tuple, Optional, Union, Any, DefaultDict, Dict

import numpy as np
import cv2

from ..structure.large_image import BoxMode

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


class MirrorDetection(HookBase):

    def __init__(self, first_thre: float, second_thre: float):
        self.first_thre = first_thre
        self.second_thre = second_thre

    def before_step(self):
        self.tracker.data.pipe_det(first_thre=self.first_thre, second_thre=self.second_thre, patch_width=1024, patch_height=1024)
        self.tracker.state_detections = self.tracker.data.bboxes

class RoadObjectDetection(HookBase):

    def __init__(self, score_thre: float=0.):
        self.score_thre = score_thre

    def before_step(self):
        self.tracker.data.inference_of_objct_detection(server = "EFFICIENTDET")
        self.tracker.data.adapt_class_confidence_thre(self.score_thre)
        self.tracker.state_detections = self.tracker.data.bboxes

class BoxCoordinateNormalizer(HookBase):
    """
    If the size of image varies in the sequence, this hook should be inserted after detector and before recorder.
    """

    def before_step(self):
        """
        XYXY_ABS to XYXYREL
        """
        self.width = self.tracker.data.width
        self.height = self.tracker.data.height

        self.tracker.state_detections = BoxMode.convert_boxes(self.tracker.state_detections, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYXY_REL, width=self.width, height=self.height)
        self.tracker.state_boxes = BoxMode.convert_boxes(self.tracker.state_boxes, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYXY_REL, width=self.width, height=self.height)

    def after_step(self):
        """
        XYXY_REL to XYXY_ABS
        """

        self.tracker.state_boxes = BoxMode.convert_boxes(self.tracker.state_boxes, from_mode=BoxMode.XYXY_REL, to_mode=BoxMode.XYXY_ABS, width=self.width, height=self.height)

class AreaCalculator(HookBase):
    """
    After BoxCoordinate
    """

    def __init__(self):
        self.buffer_areas = {}

    def before_track(self):
        self.tracker.state_approaching = []

    def after_step(self):
        """
        """
        self.width = self.tracker.data.width
        self.height = self.tracker.data.height

        bboxes = BoxMode.convert_boxes(self.tracker.state_boxes, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYXY_REL, width=self.width, height=self.height)

        self.tracker.state_approaching = []

        for track_id, bbox in zip(self.tracker.state_track_ids, bboxes):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if track_id in self.buffer_areas: # existing track
                if self.buffer_areas[track_id] - area < 0: # approaching
                    self.tracker.state_approaching.append(True)
                else: # not approaching
                    self.tracker.state_approaching.append(False)
            else: # new track
                self.tracker.state_approaching.append(False)
            self.buffer_areas[track_id] = area

class MidpointCalculator(HookBase):
    """
    After BoxCoordinate
    """

    def __init__(self):
        self.buffer_midpoint = {}

    def before_track(self):
        self.tracker.state_approaching = []

    def after_step(self):
        """
        """
        self.width = self.tracker.data.width
        self.height = self.tracker.data.height

        bboxes = BoxMode.convert_boxes(self.tracker.state_boxes, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYXY_REL, width=self.width, height=self.height)

        self.tracker.state_approaching = []

        for track_id, bbox in zip(self.tracker.state_track_ids, bboxes):
            midpoint = ((bbox[2]-bbox[0]) // 2, bbox[3])
            if track_id in self.buffer_midpoint: # existing track
                if self.buffer_midpoint[track_id][1] - midpoint[1] < 0: # approaching
                    self.tracker.state_approaching.append(True)
                else: # not approaching
                    self.tracker.state_approaching.append(False)
            else: # new track
                self.tracker.state_approaching.append(False)
            self.buffer_midpoint[track_id] = midpoint

class WidthAndHeihtCalculator(HookBase):
    """
    After BoxCoordinate
    """

    def __init__(self):
        self.buffer_size = {}

    def before_track(self):
        self.tracker.state_approaching = []

    def after_step(self):
        """
        """
        self.width = self.tracker.data.width
        self.height = self.tracker.data.height

        bboxes = BoxMode.convert_boxes(self.tracker.state_boxes, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYXY_REL, width=self.width, height=self.height)
        cnt = 0
        for track_id, bbox in zip(self.tracker.state_track_ids, bboxes):
            size = (bbox[2]-bbox[0], bbox[3]-bbox[1])
            if track_id in self.buffer_size: # existing track
                if self.buffer_size[track_id][0] >= size[0]: # not approaching
                    self.tracker.state_approaching[cnt] = False
                if self.buffer_size[track_id][1] >= size[1]: # not approaching
                    self.tracker.state_approaching[cnt] = False
            else: # new track
                self.tracker.state_approaching[cnt] = False
            self.buffer_size[track_id] = size
            cnt += 1

# class RiskPredictor(HookBase):
#     def __init__(self):
    
#     def before_track(self):


class Recorder(HookBase):

    def after_step(self):
        record = copy.deepcopy(self.tracker.data)
        record.track_ids = self.tracker.state_track_ids
        record.bboxes = self.tracker.state_boxes
        if hasattr(self.tracker, 'state_approaching'):
            record.approaching = self.tracker.state_approaching
        self.tracker.record = record


class ImageWriter(HookBase):

    def __init__(self, root_output_images: str, image_extention: str='.png'):
        self.root_output_images = root_output_images
        self.image_extention = image_extention

    def after_step(self):
        image_name = str(self.tracker.iter).zfill(4) + self.image_extention
        self.output_image_path = os.path.join(self.root_output_images, image_name)
        self.tracker.record.clear_patches()
        cv2.imwrite(self.output_image_path, self.tracker.record.image_drawn)
        assert os.path.exists(self.output_image_path)

class ImageWriterForApproaching(HookBase):

    def __init__(self, root_output_images: str, image_extention: str='.png'):
        self.root_output_images = root_output_images
        self.image_extention = image_extention

    def after_step(self):
        image_name = str(self.tracker.iter).zfill(4) + self.image_extention
        self.output_image_path = os.path.join(self.root_output_images, image_name)
        cv2.imwrite(self.output_image_path, self.tracker.record.image_drawn_as_approaching)
        assert os.path.exists(self.output_image_path)

class VideoWriter(HookBase):
    """
    # TODO: this is not effective.
    """
    def __init__(self, root_output_video: str):
        self.root_output_video = root_output_video
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        frame_rate = 30 # TODO: make cofigureble
        self.size = (512, 512)
        self.video_writer = cv2.VideoWriter(os.path.join(self.root_output_video, 'tracked.mp4'), fmt, frame_rate, self.size)

    def after_step(self):
        frame = self.tracker.record.image_drawn
        BLACK = (0, 0, 0)
        width = self.tracker.record.width
        height = self.tracker.record.height
        padded_frame = np.full((*self.size, 3), BLACK, dtype=np.uint8)
        x_c = (self.size[0] - width) // 2
        y_c = (self.size[1] - height) // 2
        padded_frame[y_c:y_c+height, x_c:x_c+width] = frame
        self.video_writer.write(frame)

    def after_track(self):
        self.video_writer.release()

class MOTReader(HookBase):
    def __init__(self, path_mot_txt: str):
        self.path_mot_txt = path_mot_txt
        assert os.path.isfile(self.path_mot_txt), self.path_mot_txt
    
    def before_step(self):
        self.read_from_mot_txt(self.path_mot_txt)

    def read_from_mot_txt(self, mot_txt_file: str):
        mapped_annotations = self.get_mapped_annotatons_from_mot_txt(mot_txt_file)
        frame_num = self.tracker.iter + 1
        annotations = mapped_annotations[frame_num]
        self.tracker.state_boxes = []
        self.tracker.state_track_ids = []
        self.tracker._data.class_confidences = []
        for listed_line in annotations: # for each object
            bbox = [float(x) for x in listed_line[2:6]]
            bbox[0] -= 1.
            bbox[1] -= 1.
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bbox = [int(x) for x in bbox]
            class_confidence = float(listed_line[6])
            self.tracker.state_boxes.append(bbox)
            self.tracker.state_track_ids.append(int(listed_line[1]))
            self.tracker._data.class_confidences.append((0.0, class_confidence))
                
    def get_mapped_annotatons_from_mot_txt(self, mot_txt_file: str) -> DefaultDict[int, List[List[str]]]:
        with open(mot_txt_file, "r") as mot_txt:
            annotations: DefaultDict[int, List[List[str]]] = defaultdict(lambda: [])
            for line in mot_txt: # per-object
                line = line.rstrip('\r\n')
                listed_line = line.split(',')
                frame_num = int(listed_line[0])
                annotations[frame_num].append(listed_line)
            return annotations
        

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
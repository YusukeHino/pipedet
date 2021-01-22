
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
import queue
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
        self.tracker.data.remove_too_big_boxes(width_thre=0.8, height_thre=0.8)
        self.tracker.state_detections = self.tracker.data.bboxes

class MOTReader(HookBase):
    def __init__(self, path_mot_txt: str):
        self.path_mot_txt = path_mot_txt
        assert os.path.isfile(self.path_mot_txt), self.path_mot_txt
    
    def before_track(self):
        self.mapped_annotations = self.get_mapped_annotatons_from_mot_txt(self.path_mot_txt)

    def before_step(self):
        self.read_from_mot_txt()

    def read_from_mot_txt(self):
        frame_num = self.tracker.iter
        annotations = self.mapped_annotations[frame_num]
        self.tracker.state_boxes = []
        self.tracker.state_track_ids = []
        self.tracker._data.class_confidences = []
        for listed_line in annotations: # for each object
            bbox = [float(x) for x in listed_line[2:6]]
            bbox[0] -= 1.
            bbox[1] -= 1.
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bbox = [float(x) for x in bbox]
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

class BigBoxRemover(HookBase):

    def __init__(self, max_rel_width: float = 0.9, max_rel_height: float = 0.9):
        self.max_rel_width = max_rel_width
        self.max_rel_height = max_rel_height

    def before_step(self):
        state_detections = [] 
        for detection in self.tracker.state_detections:
            if (detection[2] - detection[0] > self.max_rel_width) and (detection[3] - detection[1] > self.max_rel_height):
                continue
            else:
                state_detections.append(detection)
        self.tracker.state_detections = state_detections

class ApproachingInitializer(HookBase):

    def before_track(self):
        self.tracker.state_approaching = []
    
    def after_step(self):
        self.tracker.state_approaching = [False] * len(self.tracker.state_track_ids)

class FeatureMatcher(HookBase):
    def __init__(self, min_hessian: int=400, interval:int=1):
        """
        TODO: Is min_hessian meaningful?
        """
        self.min_hessian = min_hessian
        self.detector = cv2.ORB_create(nfeatures = 10000, edgeThreshold = 17, patchSize=17)
        self.size = (512, 512)
        self.interval = interval
        
    @property
    def previous_frame(self):
        return self._previous_frame

    @property
    def previous_keypoints_n_descriptors(self):
        return self._previous_keypoints_n_descriptors

    @property
    def previous_boxes(self):
        return self._previous_boxes
        
    @property
    def previous_track_ids(self):
        return self._previous_track_ids

    @property
    def previous_size(self):
        return self._previous_size

    def before_track(self):
        self.tracker.matching_interval = self.interval
        assert self.interval > 0
        self.buffer_frames = queue.Queue()
        for i in range(self.interval):
            self.buffer_frames.put_nowait(None)
        self.buffer_boxes = queue.Queue()
        for i in range(self.interval):
            self.buffer_boxes.put_nowait(None)
        self.buffer_track_ids = queue.Queue()
        for i in range(self.interval):
            self.buffer_track_ids.put_nowait(None)
        self.buffer_sizes = queue.Queue()
        for i in range(self.interval):
            self.buffer_sizes.put_nowait(None)
        self.buffer_keypoints_n_descriptors = queue.Queue()
        for i in range(self.interval):
            self.buffer_keypoints_n_descriptors.put_nowait(None)


    def after_step(self):
        self._previous_frame = self.buffer_frames.get_nowait()
        self._previous_boxes = self.buffer_boxes.get_nowait()
        self._previous_track_ids = self.buffer_track_ids.get_nowait()
        self._previous_size = self.buffer_sizes.get_nowait()
        self._previous_keypoints_n_descriptors = self.buffer_keypoints_n_descriptors.get_nowait()
        frame = self.tracker.data.image
        frame = cv2.resize(frame, self.size)
        keypoints2, descriptors2 = self.detector.detectAndCompute(frame, None)
        keypoints2, descriptors2 = self.remove_outside_brim_with_ellipse(keypoints2, descriptors2, self.size)
        assert descriptors2 is not None
        keypoints2, descriptors2 = self.remove_inside_bboxes(keypoints2, descriptors2, self.size)
        assert descriptors2 is not None
        if self.previous_frame is not None: # not first loop
            assert self.previous_keypoints_n_descriptors is not None
            keypoints1, descriptors1 = self.previous_keypoints_n_descriptors
            assert descriptors1 is not None
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
            # Match descriptors.
            try:
                good_feature_matches = bf.match(descriptors1,descriptors2)
            except cv2.error as err:
                self.tracker.logger.debug(f"Reagarding to matching of features:\n{err}")
                good_feature_matches = []
            # Sort them in the order of their distance.
            good_feature_matches = sorted(good_feature_matches, key = lambda x:x.distance)
            # good_feature_matches = good_feature_matches[:20]

            MIN_MATCH_COUNT = 6
            if len(good_feature_matches)>MIN_MATCH_COUNT:
                src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_feature_matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_feature_matches ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                height, width = self.size
                pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                # left-top, left-bottom, right-bottom, right-top
                self.tracker.state_quadruple_of_points = dst
                self.tracker.state_matchesMask = matchesMask
                quadruples_of_points_for_boxes = []

                bboxes = BoxMode.convert_boxes(self.previous_boxes, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYXY_REL, width=self.previous_size[0], height=self.previous_size[1])
                for bbox in bboxes:
                    x1, y1, x2, y2 = (rel*self.size[0] if cnt%2==0 else rel*self.size[1] for cnt,rel in enumerate(bbox))
                    pts_for_box = np.float32([[x1,y1],[x1,y2],[x2,y2],[x2,y1]]).reshape(-1,1,2)
                    dst_for_box = cv2.perspectiveTransform(pts_for_box,M)
                    quadruples_of_points_for_boxes.append(dst_for_box)
                self.tracker.state_quadruples_of_points_for_boxes = quadruples_of_points_for_boxes
                self.tracker.state_previous_track_ids = self.previous_track_ids
            else:
                self.tracker.logger.debug("Not enough matches are found - %d/%d" % (len(good_feature_matches), MIN_MATCH_COUNT))
                matchesMask = None
                self.tracker.state_matchesMask = matchesMask
                self.tracker.state_quadruple_of_points = None

            self.tracker.state_keypoints_n_descriptors = ((keypoints1, descriptors1), (keypoints2, descriptors2))
            self.tracker.state_good_feature_matches = good_feature_matches

        self.buffer_boxes.put_nowait(self.tracker.state_boxes)
        self.buffer_track_ids.put_nowait(self.tracker.state_track_ids)
        self.buffer_sizes.put_nowait((self.tracker.data.width, self.tracker.data.height))
        self.buffer_frames.put_nowait(frame)
        self.buffer_keypoints_n_descriptors.put_nowait((keypoints2, descriptors2))

    def remove_out_of_center(self, keypoints, descriptors, original_size):
        ret_keypoints = []
        ret_indxs_of_descriptor = np.array([], dtype=np.int8)
        ret_descriptors = np.array([])

        for cnt, (keypoint, descriptor) in enumerate(zip(keypoints, descriptors)):
            if self.see_if_is_located_in_center(keypoint.pt, original_size):
                ret_keypoints.append(keypoint)
                ret_indxs_of_descriptor = np.append(ret_indxs_of_descriptor, cnt)
        if len(ret_indxs_of_descriptor):
            ret_descriptors = descriptors[ret_indxs_of_descriptor]
        return ret_keypoints, ret_descriptors
    
    def remove_outside_brim_with_ellipse(self, keypoints, descriptors, original_size):
        ret_keypoints = []
        ret_indxs_of_descriptor = np.array([], dtype=np.int8)
        ret_descriptors = np.array([])
        
        width, height = original_size
        alpha = (1/6,)
        beta = (1/16, 1/8)
        for cnt, (keypoint, descriptor) in enumerate(zip(keypoints, descriptors)):
            x, y = keypoint.pt
            if (x - 1/2 * width) ** 2 / (1/2 * width - beta[0]*width) ** 2 + (y - 1/2 * height) ** 2 / (1/2*height-beta[1]*height) ** 2 <= 1:
                ret_keypoints.append(keypoint)
                ret_indxs_of_descriptor = np.append(ret_indxs_of_descriptor, cnt)
        if len(ret_indxs_of_descriptor):
            ret_descriptors = descriptors[ret_indxs_of_descriptor]
        return ret_keypoints, ret_descriptors
    
    def remove_inside_bboxes(self, keypoints, descriptors, original_size):
        ret_keypoints = keypoints
        ret_descriptors = descriptors
        self.width = self.tracker.data.width
        self.height = self.tracker.data.height
        bboxes = BoxMode.convert_boxes(self.tracker.state_boxes, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYXY_REL, width=self.width, height=self.height)
        for bbox in bboxes:
            ret_keypoints, ret_descriptors = self.remove_inside_bbox(ret_keypoints, ret_descriptors, original_size, bbox)
        return ret_keypoints, ret_descriptors

    def remove_inside_bbox(self, keypoints, descriptors, original_size, bbox):
        ret_keypoints = []
        ret_indxs_of_descriptor = np.array([], dtype=np.int8)
        ret_descriptors = np.array([])

        for cnt, (keypoint, descriptor) in enumerate(zip(keypoints, descriptors)):
            if not self.see_if_is_inside_bbox(keypoint.pt, original_size, bbox):
                ret_keypoints.append(keypoint)
                ret_indxs_of_descriptor = np.append(ret_indxs_of_descriptor, cnt)
        if len(ret_indxs_of_descriptor):
            ret_descriptors = descriptors[ret_indxs_of_descriptor]
        return ret_keypoints, ret_descriptors

    def see_if_is_located_in_center(self, coordinate, size):
        """
        Args:
            coordinate: Tuple[int, int]
            size: Tuple[int, int]
                width, height
        Returns:
            bool
        """
        width, height = size
        x, y = coordinate
        if x <= 1/4 * width:
            return False
        if x >= 3/4 * width:
            return False
        if y <= 1/4 * height:
            return False
        if y >= 3/4 * height:
            return False
        return True

    def see_if_is_inside_bbox(self, coordinate, size, bbox):
        width, height = size
        x, y = coordinate
        x1, y1, x2, y2 = bbox
        if x/width < x1:
            return False
        if y/height < y1:
            return False
        if x/width > x2:
            return False
        if y/height > y2:
            return False
        return True

class TransformedMidpointCalculator(HookBase):

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

        if hasattr(self.tracker.record, 'good_feature_matches'): # not first loop
            if True: # TODO: flag meaning effective transform matrix
                for track_id, quadruples_of_points_for_box in zip(self.tracker.state_previous_track_ids, self.tracker.state_quadruples_of_points_for_boxes):
                    rel_quadruples_of_points_for_box = quadruples_of_points_for_box / 512 # TODO: get size
                    self.buffer_midpoint[track_id] = ((rel_quadruples_of_points_for_box[1][0][0] + rel_quadruples_of_points_for_box[2][0][0]) / 2, (rel_quadruples_of_points_for_box[1][0][1] + rel_quadruples_of_points_for_box[2][0][1]) / 2)
                
        cnt = 0
        for track_id, bbox in zip(self.tracker.state_track_ids, bboxes):
            midpoint = ((bbox[2]-bbox[0]) / 2, bbox[3])
            if track_id in self.buffer_midpoint: # existing track
                if self.buffer_midpoint[track_id][1] - midpoint[1] < 0: # approaching
                    self.tracker.state_approaching[cnt] = True
                else: # not approaching
                    self.tracker.state_approaching[cnt] = False
            else: # new track
                self.tracker.state_approaching[cnt] = False
            self.buffer_midpoint[track_id] = midpoint
            cnt += 1

    
class HorizontalMovementCounter(HookBase):
    """
    """

    def __init__(self, right_trend_is_approaching: bool=True):
        self.right_trend_is_approaching = right_trend_is_approaching
        self.buffer_midpoint = {}
        self.buffer_right_move_num = {}

    def before_track(self):
        pass

    def after_step(self):
        """
        """
        self.width = self.tracker.data.width
        self.height = self.tracker.data.height

        bboxes = BoxMode.convert_boxes(self.tracker.state_boxes, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYXY_REL, width=self.width, height=self.height)

        cnt = 0
        for track_id, bbox in zip(self.tracker.state_track_ids, bboxes):
            midpoint = ((bbox[2]-bbox[0]) / 2, bbox[3])
            if track_id in self.buffer_right_move_num: # existing track
                if self.buffer_midpoint[track_id][0] - midpoint[0] < -0.01: # move right
                    self.buffer_right_move_num[track_id] += 1
                    self.tracker.logger.debug(f"\n track id: {track_id} moved right {self.buffer_midpoint[track_id][0] - midpoint[0]}, and right_move_num {self.buffer_right_move_num[track_id]}.")
                elif self.buffer_midpoint[track_id][0] - midpoint[0] > 0.01: # move left
                    self.buffer_right_move_num[track_id] -= 1
                    self.tracker.logger.debug(f"\n track id: {track_id} moved left {self.buffer_midpoint[track_id][0] - midpoint[0]}, and right_move_num {self.buffer_right_move_num[track_id]}.")
                else: # same horizontal position
                    pass 

                if self.right_trend_is_approaching:
                    if self.buffer_right_move_num[track_id] > 0:
                        self.tracker.state_approaching[cnt] = True
                    else:
                        self.tracker.state_approaching[cnt] = False
                else:
                    if self.buffer_right_move_num[track_id] < 0:
                        self.tracker.state_approaching[cnt] = True
                    else:
                        self.tracker.state_approaching[cnt] = False
            else: # new track
                self.buffer_right_move_num[track_id] = 0
                self.tracker.state_approaching[cnt] = False
            self.buffer_midpoint[track_id] = midpoint
            cnt += 1

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
        assert len(bboxes) == len(self.tracker.state_track_ids)
        cnt = 0
        for track_id, bbox in zip(self.tracker.state_track_ids, bboxes):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if track_id in self.buffer_areas: # existing track
                if self.buffer_areas[track_id] - area < 0: # approaching
                    self.tracker.state_approaching[cnt] = True
                else: # not approaching
                    self.tracker.state_approaching[cnt] = False
            else: # new track
                self.tracker.state_approaching[cnt] = False
            self.buffer_areas[track_id] = area
            cnt += 1

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

        cnt = 0
        for track_id, bbox in zip(self.tracker.state_track_ids, bboxes):
            midpoint = ((bbox[2]-bbox[0]) / 2, bbox[3])
            if track_id in self.buffer_midpoint: # existing track
                if self.buffer_midpoint[track_id][1] - midpoint[1] < 0: # approaching
                    self.tracker.state_approaching[cnt] = True
                else: # not approaching
                    self.tracker.state_approaching[cnt] = False
            else: # new track
                self.tracker.state_approaching[cnt] = False
            self.buffer_midpoint[track_id] = midpoint
            cnt += 1


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
        if hasattr(self.tracker, 'state_good_feature_matches'):
            record.good_feature_matches = self.tracker.state_good_feature_matches
            record.keypoints_n_descriptors = self.tracker.state_keypoints_n_descriptors
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

class _VideoWriterBase(HookBase):
    """
    """
    def __init__(self, root_output_video: str, fps: float, do_resize: bool=False, size: Tuple[int, int]=(512,512), out_filename: str='tracked.mp4'):
        self.root_output_video = root_output_video
        assert os.path.isdir(self.root_output_video)
        self.out_filename = out_filename
        self.do_resize = do_resize
        # fmt = cv2.VideoWriter_fourcc(*'XVID')
        # fmt = cv2.VideoWriter_fourcc(*"x264")
        # fmt = cv2.VideoWriter_fourcc(*"MPEG")
        # fmt = cv2.VideoWriter_fourcc(*'H264')
        # fmt = cv2.VideoWriter_fourcc('mp4v') # for macOS and mp4
        # fmt = cv2.CAP_FFMPEG
        fmt = cv2.VideoWriter_fourcc(*'avc1') # for ubuntu 18.04 and mp4
        frame_rate = fps
        self.size = size
        self.video_writer = cv2.VideoWriter(os.path.join(self.root_output_video, self.out_filename), fmt, frame_rate, self.size)

    def after_step(self):
        
        frame, width, height = self.get_frame_n_width_height()
        if self.do_resize:
            resized_frame = cv2.resize(frame, self.size)
            self.video_writer.write(resized_frame)
        else: # padding
            padded_frame = np.full((*self.size, 3), 0, dtype=np.uint8)
            x_c = (self.size[0] - width) // 2
            y_c = (self.size[1] - height) // 2
            padded_frame[y_c:y_c+height, x_c:x_c+width] = frame
            self.video_writer.write(padded_frame)

    def get_frame_n_width_height(self):
        """
        returns:
            frame (np.Ndarray)
            width: int
            height: int
        """
        raise NotImplementedError

    def after_track(self):
        self.video_writer.release()
        assert os.path.isfile(os.path.join(self.root_output_video, self.out_filename)), "mp4 file was not generated."

class VideoWriterForTracking(_VideoWriterBase):
    def get_frame_n_width_height(self):
        return self.tracker.record.image_drawn, self.tracker.record.width, self.tracker.record.height

class VideoWriterForApproaching(_VideoWriterBase):
    def __init__(self, *args, **kwargs):
        if not ('out_filename' in kwargs):
            kwargs['out_filename'] = 'approaching.mp4'
        super().__init__(*args, **kwargs)

    def get_frame_n_width_height(self):
        return self.tracker.record.image_drawn_as_approaching, self.tracker.record.width, self.tracker.record.height

class VideoWriterForMatching(_VideoWriterBase):
    def __init__(self, *args, **kwargs):
        if not ('out_filename' in kwargs):
            kwargs['out_filename'] = 'feature_matching.mp4'
        super().__init__(*args, **kwargs)
    
    def before_track(self):
        self.buffer_data = queue.Queue()
        for i in range(self.tracker.matching_interval):
            self.buffer_data.put_nowait(None)

    def after_step(self):
        self.data = self.tracker.record
        self.previous_data = self.buffer_data.get_nowait()
        super().after_step()
        self.buffer_data.put_nowait(self.tracker.record)
            
    def get_frame_n_width_height(self):
        if not hasattr(self.tracker.record, 'good_feature_matches'): # first loop
            return np.full((*self.size, 3), 0, dtype=np.uint8), self.size[0], self.size[1]
        # not first loop
        resized_frame = cv2.resize(self.data.image_drawn, (512, 512)) #TODO: get size by somehow
        resized_previous_frame = cv2.resize(self.previous_data.image_drawn, (512, 512)) #TODO: get size by somehow
        # img_matches = np.empty((*self.size, 3), dtype=np.uint8) # This is bad.
        img_matches = np.empty((max(resized_previous_frame.shape[0], resized_frame.shape[0]), resized_previous_frame.shape[1]+resized_frame.shape[1], 3), dtype=np.uint8)
        keypoints1 = self.tracker.record.keypoints_n_descriptors[0][0]
        keypoints2 = self.tracker.record.keypoints_n_descriptors[1][0]
        if self.tracker.state_quadruple_of_points is not None:
            dst = self.tracker.state_quadruple_of_points # abs
            resized_frame = cv2.polylines(resized_frame, [np.int32(dst)],True,255,3, cv2.LINE_AA)
            dsts_for_boxes = self.tracker.state_quadruples_of_points_for_boxes
            for dst_for_box in dsts_for_boxes:
                resized_frame = cv2.polylines(resized_frame, [np.int32(dst_for_box)],True,255,3, cv2.LINE_AA)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = self.tracker.state_matchesMask, # draw only inliers
                flags = cv2.DrawMatchesFlags_DEFAULT)
        # cv2.drawMatches(resized_previous_frame, keypoints1, resized_frame, keypoints2, self.tracker.record.good_feature_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_matches = cv2.drawMatches(resized_previous_frame, keypoints1, resized_frame, keypoints2, self.tracker.record.good_feature_matches, None, **draw_params)
        return img_matches, self.size[0], self.size[1]
        

class MOTWriter(HookBase):

    def __init__(self, root_output_mot: str):
        self.path_to_zip = os.path.join(root_output_mot, 'gt.zip')
        self.path_to_txt = os.path.join(root_output_mot, 'gt.txt')
        self.path_to_labels = os.path.join(root_output_mot, 'labels.txt')
        self.lines_as_lists: List[List[Any]] = []

    def after_step(self):
        for track_id, bbox in zip(self.tracker.record.track_ids, self.tracker.record.bboxes):
            line_as_list = [self.tracker.iter, track_id, bbox[0] + 1, bbox[1] + 1, bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0, 1, 1.0]
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
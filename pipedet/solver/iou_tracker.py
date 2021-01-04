
import logging
import time
import weakref
from typing import List, Tuple, Optional, Union, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data.image_loader import TrackingFrameLoader
from ..structure.large_image import LargeImage, _RawBoxType
from ..solver.hooks import HookBase


class BBox:
    """

    """
    def __init__(self, box: _RawBoxType, superclass_id: Optional[int], class_confidence: int) -> None:
        self.box: _RawBoxType = box
        self._superclass_id = superclass_id
        self.class_confidence = class_confidence

    @property
    def superclass_id(self) -> Any:
        if self._superclass_id != None:
            return self._superclass_id
        else:
            raise TypeError

class BBoxAsARecord(BBox):
    """
    Properties:
    age: (0-based) how many frames passed after firstly detected
    """
    def __init__(self, box: _RawBoxType, superclass_id: Optional[int], class_confidence: int, birth_year: Optional[int]) -> None:
        super().__init__(box, superclass_id, class_confidence)
        self._age: int = 0
        self._birth_year = birth_time

    @property
    def birth_time(self) -> Any:
        return self._birth_year

    @property
    def age(self) -> int:
        return self._age
    
    def get_old(self) -> None:
        self._age += 1

class Trajectory:
    """

    """
    def __init__(self, first_head: BBoxAsARecord) -> None:
        self._track_id = self._init_pop_id_from_queue()
        self._bbox_buffer: Deque[BBoxAsARecord] = deque() # if you have to cinfigure max length, please refer to maxlen argument
        first_head.get_old # TODO examine that get_old() heare is really needed.
        self._bbox_buffer.append(first_head)

    def track_and_get_old(self, new_head: BBoxAsARecord) -> None:
        assert self.head.superclass_id == new_head.superclass_id, f'Object with superclass {new_head.superclass_id} cannot be tracked to the track, which superclass is {self.head.superclass_id}.'
        self._bbox_buffer.append(new_head)
        self.get_old_all() # TODO examine that get_old() heare is really needed.

    def get_old_all(self) -> None:
        for bbox in self._bbox_buffer:
            bbox.get_old()
    
    # TODO __getitem__()

    @property
    def head(self) -> BBoxAsARecord:
        return self._bbox_buffer[-1]
    
    @property
    def superclass_id(self) -> int:
        return self.head.superclass_id


class TrackerBase:
    
    def __init__(self):
        self._hooks = []
        self.record: Optional[LargeImage] = None
        self.available_id: int = 1

    def register_hooks(self, hooks):
        """

        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.tracker = weakref.proxy(self)
        self._hooks.extend(hooks)

    def track(self):
        """
        Args:
        """

        self.start_iter = self.loader.frame_num_start
        self.end_iter = self.loader.frame_num_end
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Starting tracking from iteration {}".format(self.start_iter))

        try:
            self.before_track()
            for self.iter in range(self.start_iter, self.end_iter+1):
                self.before_step()
                self.run_step()
                self.after_step()
        except Exception:
            self.logger.exception("Exception during training:")
            raise
        finally:
            self.after_track()

    def before_track(self):
        for h in self._hooks:
            h.before_track()

    def after_track(self):
        for h in self._hooks:
            h.after_track()

    def before_step(self):
        self._data = next(self.loader_iter)
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        self.logger.info(f"iter: {self.iter}")

    def run_step(self):
        raise NotImplementedError

    def load_frames(self, *args, **kwargs):
        self.loader = TrackingFrameLoader(*args, **kwargs)
        self.loader_iter = iter(self.loader)

    @classmethod
    def _init_pop_id_from_queue(cls) -> int:
        ret = cls.available_id
        cls.available_id += 1
        return ret

    @property
    def data(self) -> Any:
        return self._data

class IoUTracker(TrackerBase):
    """
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.state_boxes: List[List[Union[int, float]]] = []
        self.state_track_ids: List[int] = []
        self.state_ages: List[int] = []
        self.state_detections: List[List[Union[int, float]]] = []
        # self.state_approaching: List[bool] = []

    def run_step(self):
        state_boxes: np.ndarray = np.array(self.state_boxes)
        state_track_ids: np.ndarray = np.array(self.state_track_ids, dtype=int)
        state_ages: np.ndarray = np.array(self.state_ages)
        state_detections: np.ndarray = np.array(self.state_detections)

        matched_boxes_indxes, matched_detections_indxes = self.iou_matching(state_boxes, state_detections, iou_thre=0.5)

        unmatched_detections = np.delete(state_detections, matched_detections_indxes, 0)

        state_boxes = state_detections[matched_detections_indxes]
        track_ids = state_track_ids[matched_boxes_indxes]

        track_ids_for_unmatched_detections = np.array([_ for _ in range(self.available_id, self.available_id+len(unmatched_detections))], dtype=int)
        self.available_id += len(unmatched_detections)

        if state_boxes.ndim != unmatched_detections.ndim:
            if state_boxes.ndim == 1:
                state_boxes = unmatched_detections
            elif unmatched_detections.ndim == 1:
                pass
            else:
                raise NotImplementedError
        else:
            state_boxes = np.concatenate((state_boxes, unmatched_detections))
        state_track_ids = np.concatenate((track_ids, track_ids_for_unmatched_detections))

        self.state_boxes = state_boxes.tolist()
        self.state_track_ids = state_track_ids.tolist()

    def iou_matching(self, boxes_a_arg: np.ndarray, boxes_b_arg: np.ndarray, iou_thre: float=0.5) -> Tuple[np.ndarray, np.ndarray]:

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
                iou_matrix[a, b] = -1 * self.box_iou(box_a, box_b)

        a_ind, b_ind = linear_sum_assignment(iou_matrix)

        a_ind_filtered = np.zeros(0, int)
        b_ind_filtered = np.zeros(0, int)
        for a, b in zip(a_ind, b_ind):
            if iou_matrix[a, b] * (-1) >= iou_thre:
                a_ind_filtered = np.append(a_ind_filtered, a)
                b_ind_filtered = np.append(b_ind_filtered, b)

        return a_ind_filtered, b_ind_filtered

    def box_iou(self, a, b):
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

class NullTracker(TrackerBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.state_boxes: List[List[Union[int, float]]] = []
        self.state_track_ids: List[int] = []
        self.state_ages: List[int] = []
        self.state_detections: List[List[Union[int, float]]] = []

    def run_step(self):
        pass
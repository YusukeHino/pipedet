
import logging
import time
import weakref
from typing import List, Tuple, Optional, Union, Any

import numpy as np

from ..data.image_loader import TrackingFrameLoader
from ..structure.large_image import LargeImage


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

    available_id: int = 1
    
    def __init__(self):
        self._hooks = []
        self.tracker.record: Optional[LargeImage] = None

    def register_hooks(self, hooks):
        """

        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.tracker = weakref.proxy(self)
        self._hooks.extend(hooks)

    def track(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        try:
            self.before_track()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
        except Exception:
            logger.exception("Exception during training:")
            raise
        finally:
            self.after_track()

    def before_track(self):
        for h in self._hooks:
            h.before_track()

    def after_track(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        self._data = next(self.loader_iter)
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def load_frames(self, root_images: str):
        self.loader = TrackingFrameLoader(root_images=root_images)
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
        self.state_boxes: List[List[int]] = []
        self.state_track_ids: List[int] = []
        self.state_ages: List[int] = []

    def run_step(self):
        self.iou_tracking(self.state_boxes, self.data.boxes, iou_thre=0.5)

    def iou_tracking(boxes_a: List[List[int]], boxes_b: List[List[int]]):
        '''
        '''
        pass # TODO
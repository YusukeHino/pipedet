
from typing import List, Tuple, Optional, Union, Any

import numpy as np

from ..data.image_loader import TrackingFrameLoader


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


class BaseTracker:

    available_id: int = 1

    def __init__(self) -> None:
        pass

    def load_frames(self, root_images: str):
        self.loader = TrackingFrameLoader(root_images=root_images)
        self.loader_iter = iter(self.loader)

    @classmethod
    def _init_pop_id_from_queue(cls) -> int:
        ret = cls.available_id
        cls.available_id += 1
        return ret

class IoUTracker(BaseTracker):
    """
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def linear
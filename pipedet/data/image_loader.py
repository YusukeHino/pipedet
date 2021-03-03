
import os
import logging
from collections import defaultdict
from collections import OrderedDict
from typing import List, Tuple, Optional, Union, Any

from ..structure.large_image import LargeImage

import cv2

class TrackingFrameLoader:

    def __init__(self, root_images: str, frame_num_start: int=-1, frame_num_end: int=-1):
        self._frame_num_start = frame_num_start
        self._frame_num_end = frame_num_end
        self.root_images = root_images
        self.biggest_frame_num = 0
        self.full_paths_to_frame: OrderedDict[int, str] = OrderedDict()
        is_first_loop = True
        for frame_name in sorted(os.listdir(self.root_images)):
            frame_str = os.path.splitext(frame_name)[0]
            try:
                frame_num = int(frame_str)
            except ValueError:
                logger = logging.getLogger(__name__)
                logger.error(f"Cannot extract frame_number from frame_name: {frame_name}")
                raise
            if is_first_loop:
                frame_num_count = frame_num
                self.smallest_frame_num = frame_num
                assert self.frame_num_start >= self.smallest_frame_num, f"frame_num_start {frame_num_start} is smaller than the smallest frame number in the directory."
            assert frame_num == frame_num_count, f"frame_num_count: {frame_num_count}, frame_num: {frame_num}."
            self.biggest_frame_num = max(self.biggest_frame_num, frame_num)
            full_path_to_frame = os.path.join(root_images, frame_name)
            self.full_paths_to_frame[frame_num] = full_path_to_frame
            is_first_loop = False
            frame_num_count += 1

        self.frame_num_iter = self.frame_num_start

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_num_iter > self.frame_num_end:
            raise StopIteration

        frame_path = self.full_paths_to_frame[self.frame_num_iter]
        image = cv2.imread(frame_path)
        large_image = LargeImage(image)
        self.frame_num_iter += 1
        return large_image

    def __len__(self):
        return len(self.full_paths_to_frame)

    @property
    def frame_num_start(self):
        if self._frame_num_start == -1:
            return self.smallest_frame_num
        else:
            return self._frame_num_start
    @property
    def frame_num_end(self):
        if self._frame_num_end == -1:
            return self.biggest_frame_num
        else:
            return self._frame_num_end
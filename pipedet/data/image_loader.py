
import os
import logging
from collections import defaultdict
from collections import OrderedDict
from typing import List, Tuple, Optional, Union, Any

from ..structure.large_image import LargeImage

import cv2

class TrackingFrameLoader:

    def __init__(self, root_images: str, frame_num_start: int=1, frame_num_end: int=-1):
        self.root_images = root_images
        frame_num_count = frame_num_start
        self.full_paths_to_frame: OrderedDict[int, str] = OrderedDict()
        for frame_name in sorted(os.listdir(self.root_images)):
            frame_str = os.path.splitext(frame_name)[0]
            try:
                frame_num = int(frame_str)
            except ValueError:
                logger = logging.getLogger(__name__)
                logger.error(f"Cannot extract frame_number from frame_name: {frame_name}")
                raise
            assert frame_num == frame_num_count, f"frame_num_count: {frame_num_count}, frame_num: {frame_num}."
            full_path_to_frame = os.path.join(root_images, frame_name)
            self.full_paths_to_frame[frame_num_count] = full_path_to_frame
            frame_num_count += 1
        self.frame_num_iter = frame_num_start

    def __iter__(self):
        return self

    def __next__(self):
        frame_path = self.full_paths_to_frame[self.frame_num_iter]
        image = cv2.imread(frame_path)
        large_image = LargeImage(image=image)
        self.frame_num_iter += 1
        return large_image
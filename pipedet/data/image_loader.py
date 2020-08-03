
import os
import logging
from collections import defaultdict
from collections import OrderedDict
from typing import List, Tuple, Optional, Union, Any

from ..structure.frame import Frame

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

    def __iter__(self):
        return self

    def __next__(self):
        for frame_num, frame_path in self.full_paths_to_frame.items():
            image = cv2.imread(frame_path)
            frame = Frame(image=imgae, frame_num=frame_num)
            yield frame
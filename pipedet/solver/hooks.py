
import datetime
import itertools
import logging
import os
import tempfile
import time
import copy
from collections import Counter
from typing import List, Tuple, Optional, Union, Any


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
    def before_step(self):
        self.tracker.data.pipe_det(first_thre=0.5, second_thre=0.5, patch_width=1024, patch_height=1024)

class Recorder(HookBase):

    def __init__(self, root_output_images: str):
        self.root_output_images = root_output_images

    def after_step(self):
        record = copy.deepcopy(self.tracker.data)
        image_name = str(self.tracker.iter).zfill(4) + ".jpg"
        self.tracker.output_image_path = os.path.join(self.root_output_images, image_name)
        record.track_ids = self.tracker.state_track_ids
        self.tracker.record = record

class ImageWriter(HookBase):

    def after_step(self):
        self.tracker.record.clear_pathces()
        cv2.imwrite(self.tracker.output_image_path, self.tracker.record.image_drawn)
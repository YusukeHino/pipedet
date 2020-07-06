
import os
from typing import List, Tuple, Optional
import random

import numpy as np
import cv2

class CropperSizeError(Exception):
    """
    Usupported image size is received.
    """
    pass

def crop_from_raw_input(image: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.

    Returns:
        list of crops(list of np.ndarray)
    """
    crops = []
    if (image.shape[1] == 3840 and image.shape[0] == 2160):
        # restrict W (i.e. x)
        crops.append(image[:][:2160])
        # Since crops.append(image[:][3839:]) have one, we need 3840 - 2160 = 1680
        crops.append(image[:][1680:])
    else:
        raise CropperSizeError("the size is not 3840 * 2160")

    return crops

class Image():
    """
    image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        This is the format used by OpenCV.
    bbox: x1 y1 x2 y2
    rel_path: ex) "for_rsm_detection/separated/5472_3672_for_train/capture_121.jpg"
    """
    def __init__(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[int]]]=None,
        rel_path: Optional[str]=None
        ) -> None:

        self.image = image
        self.bboxes = bboxes
        self.rel_path = rel_path
        if not self.bboxes:
            for bbox in self.bboxes:
                assert bbox[0] >= 0 and bbox[0] < self.width-1, "x1 of bbox is out of image width"
                assert bbox[2] >= 1 and bbox[2] < self.width, "x2 of bbox is out of image width"
                assert bbox[1] >= 0 and bbox[1] < self.height-1, "y1 of bbox is out of image width"
                assert bbox[3] >= 1 and bbox[3] < self.height, "y2 of bbox is out of image width"

    @property
    def filename(self) -> str:
        if self.rel_path is None:
            raise AttributeError
        return os.path.basename(self.rel_path)

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def image_drawn(self) -> np.ndarray:
        image_copy = self.image.copy()
        for bbox in self.bboxes:
            cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(image_copy, str(bbox[2]-bbox[0]) + "_" + str(bbox[3]-bbox[1]), (bbox[2]+10, bbox[3]),0,0.3,(0,255,0))
        return image_copy

    def show_image_drawn(self):
        cv2.imshow("image_drwan", self.image_drawn)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    
class LargeImage(Image):
    def __init__(self, image: np.ndarray, bboxes: List[List[int]]):
        super().__init__(image, bboxes)
    
    def crop_comprehensively(self, desired_width: int, desired_height: int) -> List[Image]:
        
        return
        
    def make_dataset(self, desired_width: int, desired_height: int) -> List[Image]:
        """
        convert from 4K image w/ bboxes to desired size images w/ bboxes randomly
        """
        dataset = []
        for bbox in self.bboxes:
            # for x1
            x1_lower_bound = bbox[2]- desired_width
            if x1_lower_bound < 0:
                x1_lower_bound = 0
            x1_upper_bound = bbox[0]
            if x1_upper_bound + desired_width >= self.width:
                x1_upper_bound = self.width - desired_width
            x1 = random.randint(x1_lower_bound, x1_upper_bound)
            # for y1
            y1_lower_bound = bbox[3] - desired_height
            if y1_lower_bound < 0:
                y1_lower_bound = 0
            y1_upper_bound = bbox[1]
            if y1_upper_bound + desired_width >= self.height:
                y1_upper_bound = self.height - desired_height
            y1 = random.randint(y1_lower_bound, y1_upper_bound)    
            cropped_image = self.image[y1:y1+desired_height, x1:x1+desired_width, :]
            cropped_bboxes = []
            for original_bbox in self.bboxes:
                cropped_bbox = [original_bbox[0]-x1, original_bbox[1]-y1, original_bbox[2]-x1, original_bbox[3]-y1]

                if cropped_bbox[0] < 0:
                    cropped_bbox[0] = 0
                elif cropped_bbox[0] >= desired_width - 1:
                    continue

                if cropped_bbox[2] < 1:
                    continue
                elif cropped_bbox[2] >= desired_width:
                    cropped_bbox[2] = desired_width - 1

                if cropped_bbox[1] < 0:
                    cropped_bbox[1] = 0
                elif cropped_bbox[1] >= desired_height - 1:
                    continue

                if cropped_bbox[3] < 1:
                    continue
                elif cropped_bbox[3] >= desired_height:
                    cropped_bbox[3] = desired_height - 1

                cropped_bboxes.append(cropped_bbox)
            try:
                data = Image(cropped_image.copy(), cropped_bboxes)
            except AssertionError:
                print("some of bbox is out of image")
            dataset.append(data)

        return dataset
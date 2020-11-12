
import os
import random
import json
import copy
from enum import IntEnum, unique
from typing import List, Tuple, Optional, Union, Any

import requests
import numpy as np
import cv2

from .image_encoder import cv2_to_base64_string
from ..common.box_suppression import suppress_crop_locations, non_max_suppression_fast
from ..common.super_resolution import super_resolution

_RawBoxType = Union[List[Union[float, int]], np.ndarray]

# color name to rgb from https://www.rapidtables.com/web/color/RGB_Color.html
COLOR = ['CORAL', 'ORANGE', 'GREEN', 'AQUA_MARINE', 'DEEP_SKY_BLUE', 'VIOLET', 'DEEP_PINK', 'RED', 'LIME', 'BLUE', 'YELLOW', 'CYAN', 'MAGENTA']
COLOR_NAME_TO_RGB = {'BLACK': (0, 0, 0),
              'WHITE': (255,255,255),
              'RED': (255,0,0),
              'LIME': (0,255,0),
              'BLUE': (0,0,255),
              'YELLOW': (255,255,0),
              'CYAN': (0,255,255),
              'MAGENTA': (255,0,255),
              'CORAL': (255,127,80),
              'ORANGE': (255,165,0),
              'GREEN': (0,128,0),
              'AQUA_MARINE': (127,255,212),
              'DEEP_SKY_BLUE': (0,191,255),
              'VIOLET': (238,130,238),
              'DEEP_PINK': (255,20,147)}

#HINO above dic is correct if (,,) = (R,G,B), so
COLOR_NAME_TO_BGR = {}
for c in COLOR:
    rgb = COLOR_NAME_TO_RGB[c]
    COLOR_NAME_TO_BGR[c] = (rgb[2],rgb[1],rgb[0])

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

@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    YXYX_REL = 4
    """
    Not yet supported!
    (y0, x0, y1, x1) in range [0, 1]. They are relative to the size of the image.
    """
    YXYX_ABS = 5
    """
    Not yet supported!
    (y0, x0, y1, x1) in absolute floating points coordinates.
    """

    @classmethod
    def convert_boxes(cls, boxes: List[_RawBoxType], from_mode: "BoxMode", to_mode: "BoxMode", width: int=None, height: int=None) -> List[_RawBoxType]:
        converted_boxes = []
        for box in boxes:
            converted_boxes.append(cls.convert(box, from_mode, to_mode, width, height))
        return converted_boxes

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode", width: int=None, height: int=None) -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)

        assert len(box) == 4, (
            "BoxMode.convert takes either a k-tuple/list, where k == 4"
        )
        if isinstance(box, List):
            arr = box.copy()

        if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
            arr[:, 2] += arr[:, 0]
            arr[:, 3] += arr[:, 1]
        elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
            arr[:, 2] -= arr[:, 0]
            arr[:, 3] -= arr[:, 1]
        elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYXY_REL:
            arr[0] = box[0] / width
            arr[1] = box[1] / height
            arr[2] = box[2] / width
            arr[3] = box[3] / height
        elif from_mode == BoxMode.XYWH_REL and to_mode == BoxMode.XYXY_ABS:
            assert not width is None and not height is None, "If you convert over REL<->ABS, you need width and height"
            arr[0] = int(arr[0]*width)
            arr[1] = int(arr[1]*height)
            arr[2] = arr[0] + int(arr[2]*width)
            arr[3] = arr[1] + int(arr[3]*height)
        elif from_mode == BoxMode.XYXY_REL and to_mode == BoxMode.XYXY_ABS:
            assert not width is None and not height is None, "If you convert over REL<->ABS, you need width and height"
            arr[0] = int(box[0]*width)
            arr[1] = int(box[1]*height)
            arr[2] = int(box[2]*width)
            arr[3] = int(box[3]*height)
        elif from_mode == BoxMode.YXYX_REL and to_mode == BoxMode.XYXY_ABS:
            assert not width is None and not height is None, "If you convert over REL<->ABS, you need width and height"
            arr[0] = int(box[1]*height)
            arr[1] = int(box[0]*width)
            arr[2] = int(box[3]*height)
            arr[3] = int(box[2]*width)
        elif from_mode == BoxMode.YXYX_ABS and to_mode == BoxMode.XYXY_ABS:
            arr[0] = int(box[1])
            arr[1] = int(box[0])
            arr[2] = int(box[3])
            arr[3] = int(box[2])

        else:
            raise NotImplementedError(
                "Conversion from BoxMode {} to {} is not supported yet".format(
                    from_mode, to_mode
                )
            )

        return arr

class Image():
    """
    image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        This is the format used by OpenCV.
    bbox: x1 y1 x2 y2
    rel_path: ex) "for_rsm_detection/separated/5472_3672_for_train/capture_121.jpg"
    """
    def __init__(
        self,
        image: np.ndarray=None,
        box_mode: "BoxMode"=BoxMode.XYXY_ABS,
        bboxes: Optional[List[List[int]]]=None,
        rel_path: Optional[str]=None,
        track_ids: Optional[List[int]]=None,
        ) -> None:

        assert isinstance(image, np.ndarray), f"argument 'image' is {type(image)}, expected np.ndarray"
        self.image = image

        if box_mode == BoxMode.XYXY_ABS:
            self.bboxes = bboxes
        elif not bboxes is None:
            self.bboxes = []
            for box in bboxes:
                xyxy_abs_box = BoxMode.convert(box, from_mode = box_mode, to_mode = BoxMode.XYXY_ABS, width=self.width, height=self.height)
                self.bboxes.append(xyxy_abs_box)
        else:
            self.bboxes = None

        self.rel_path = rel_path
        if self.bboxes:
            for bbox in self.bboxes:
                assert bbox[0] >= 0 and bbox[0] < self.width-1, "x1 of bbox is out of image width"
                assert bbox[2] >= 1 and bbox[2] < self.width, "x2 of bbox is out of image width"
                assert bbox[1] >= 0 and bbox[1] < self.height-1, "y1 of bbox is out of image width"
                assert bbox[3] >= 1 and bbox[3] < self.height, "y2 of bbox is out of image width"

        self.tag: str = ""

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
        try:
            image_copy = self.image_w_patch_grid.copy()
        except NotImplementedError:
            image_copy = self.image.copy()
        if not self.bboxes is None:
            assert len(self.bboxes) == len(self.class_confidences) or len(self.class_confidences) == 0, "length of bboxes and class_confidences are not same in spite of not being zero of len(class_confidences)"
            for idx, bbox in enumerate(self.bboxes):
                if hasattr(self, "class_confidences"):
                    class_confidence = self.class_confidences[idx]
                else:
                    class_confidence = (0., 1.)
                if hasattr(self, "track_ids"):
                    track_id = self.track_ids[idx]
                else:
                    track_id = None
                self.depict_bbox(image_copy, bbox, class_confidence, track_id)
        return image_copy
    
    def depict_bbox(self, image: np.ndarray, bbox: List[int], class_confidence: Optional[Tuple[float, float]]=None, track_id: Optional[int]=None, color: Optional[Tuple[int, int, int]]=None) -> None:

        bbox_size_str = str(bbox[2]-bbox[0]) + "*" + str(bbox[3]-bbox[1])

        if class_confidence is not None:
            class_confidence_str = str(class_confidence[1])
        if (track_id is None) or (track_id == -1):
            if color is None:
                color = (0, 255, 0)
            self.depict_rectange(image, bbox, color)
            if class_confidence is not None:
                text_to_depict = bbox_size_str + "_" + class_confidence_str
            else:
                text_to_depict = bbox_size_str
            cv2.putText(image, text_to_depict, (bbox[2]+10, bbox[3]),0,1.0,color)
        else:
            color = COLOR_NAME_TO_BGR[COLOR[int(track_id % len(COLOR))]]
            text_to_depict = 'ID:'+ str(track_id) + " " + bbox_size_str
            self.depict_rectange(image, bbox, color)
            cv2.rectangle(image, (bbox[0]-2, bbox[1]-55), (bbox[2]+330, bbox[1]), color, -1, 1)
            cv2.putText(image, text_to_depict,(bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

            
    def depict_rectange(self, image: np.ndarray, bbox: List[int], color: Tuple[int, int, int]) -> None:
        cv2.rectangle(
            img=image,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color,
            thickness=5
        )

    @property
    def image_w_patch_grid(self) -> Any:
        raise NotImplementedError()

    def show_image_drawn(self, is_patch_grid: bool=False):
        if not is_patch_grid:
            cv2.imshow("image_drwan", self.image_drawn)
        else:
            cv2.imshow("image_drawn_w_patch_grid", self.image_drawn_w_patch_grid)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def get_crop(self, box: List[int]) -> np.ndarray:
        return self.image[box[1]:box[3],box[0]:box[2],:]

    def super_resolution(self):
        self.image = super_resolution(self.image)

    def inference_of_objct_detection(self, server = "AUTOML"):
        if server == "AUTOML":
            self.container_predict() # TODO: set config flow
        elif server == "EFFICIENTDET":
            self.client_efficientdet()
        else:
            raise NotImplementedError

    def container_predict(self):
        encoded_image = cv2_to_base64_string(self.image)
        instances = {
                'instances': [
                        {'image_bytes': {'b64': encoded_image},
                        'key': 'any'}
                ]
        }
        port_number=8501
        url = f'http://automl_20200729:{port_number}/v1/models/default:predict'
        response = requests.post(url, data=json.dumps(instances))
        try:
            response.raise_for_status()
        except:
            print(response.content)
        pre_convert_bboxes = response.json()['predictions'][0]['detection_boxes']
        self.class_confidences = response.json()['predictions'][0]['detection_multiclass_scores']
        assert not self.bboxes, "Imgae object has some bbox attribute, so ObjDet has been canceled."
        self.bboxes = BoxMode.convert_boxes(pre_convert_bboxes, from_mode=BoxMode.YXYX_REL, to_mode=BoxMode.XYXY_ABS, width=self.width, height=self.height)
    
    def client_efficientdet(self):
        encoded_image = cv2_to_base64_string(self.image)
        instances = {
                'instances': [
                        {'image_bytes': {'b64': encoded_image},
                        'key': 'any'}
                ]
        }
        port_number=8501
        url = f'http://efficientdet:{port_number}'
        response = requests.post(url, data=json.dumps(instances))
        try:
            response.raise_for_status()
        except:
            print(response.content)
        content = response.json()
        pre_convert_bboxes = content['predictions'][0]['detection_boxes']
        self.class_confidences = [(0, x) for x in content['predictions'][0]['class_confidences']]
        self.class_ids = []
        self.class_ids = content['predictions'][0]['class_ids']
        assert not self.bboxes, "Imgae object has some bbox attribute, so ObjDet has been canceled."
        self.bboxes = BoxMode.convert_boxes(pre_convert_bboxes, from_mode=BoxMode.YXYX_ABS, to_mode=BoxMode.XYXY_ABS, width=self.width, height=self.height)

    # def tf_serving(self):
    #     tmp_image = self.image.copy()
    #     tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
    #     port_number=8501
    #     url = f'http://tf_serving_for_efficientdet:{port_number}/v1/models/efficientdet/versions/1:predict'
    #     # headers = {"content-type": "application/json"}
    #     data = json.dumps({"signature_name": "serving_default", "instances": tmp_image[0:3].tolist()})
    #     response = requests.post(url, data=data)
    #     try:
    #         response.raise_for_status()
    #     except:
    #         print(response.content)
    #     jsoned_response = response.json()
    #     pre_convert_bboxes = jsoned_response['predictions'][0]
    #     self.class_confidences = [(0., x[5]) for x in pre_convert_bboxes]
    #     pre_convert_bboxes = [x[1:5] for x in pre_convert_bboxes]
    #     self.bboxes = BoxMode.convert_boxes(pre_convert_bboxes, from_mode=BoxMode.YXYX_REL, to_mode=BoxMode.XYXY_ABS, width=self.width, height=self.height)


    def get_global_bboxes(self, global_location_of_image: List[int]) -> List[List[int]]:
        global_bboxes = []
        for bbox in self.bboxes:
            global_bboxes.append([
                bbox[0] + global_location_of_image[0],
                bbox[1] + global_location_of_image[1],
                bbox[2] + global_location_of_image[0],
                bbox[3] + global_location_of_image[1]
            ])
        return global_bboxes

    def adapt_class_confidence_thre(self, thre: float) -> None:
        filtered_bboxes = []
        filtered_class_confidences = []
        for bbox, class_confidence in zip(self.bboxes, self.class_confidences):
            if class_confidence[1] >= thre:
                filtered_bboxes.append(bbox)
                filtered_class_confidences.append(class_confidence)
        self.bboxes = filtered_bboxes
        self.class_confidences = filtered_class_confidences

    def nms(self, iou_thre: float=0.5):
        self.bboxes, self.class_confidences = non_max_suppression_fast(self.bboxes, self.class_confidences, overlapThresh=0.5)
        

    def find_and_pop_edge_bbox(self) -> Tuple[List[List[int]], List[List[float]], List[List[bool]]]:
        """
        Return:
            is_on_edge_list: List of is_on_edge, that is list of bool [left, top, right, bottom] are on image edges, respectively.
        """
        filtered_bboxes = []
        filtered_class_confidences = []
        edge_bboxes = []
        edge_class_confidences = []
        is_on_edge_list = []

        for bbox, class_confidence in zip(self.bboxes, self.class_confidences):
            is_on_edge = [False] * 4
            if bbox[0] <= 3:
                is_on_edge[0] = True
            if bbox[1] <= 3:
                is_on_edge[1] = True
            if bbox[2] >= self.width - 1 - 3:
                is_on_edge[2] = True
            if bbox[3] >= self.height - 1 - 3:
                is_on_edge[3] = True
            if not any(is_on_edge):
                filtered_bboxes.append(bbox)
                filtered_class_confidences.append(class_confidence)
            else:
                edge_bboxes.append(bbox)
                edge_class_confidences.append(class_confidence)
                is_on_edge_list.append(is_on_edge)
                
        self.bboxes = filtered_bboxes
        self.class_confidences = filtered_class_confidences
        return edge_bboxes, edge_class_confidences, is_on_edge_list
    
    def is_out_of_frame(self, bbox: List[int]) -> bool:
        if (bbox[0] < 0) or (bbox[1] < 0) or (bbox[2] > self.width) or (bbox[3] > self.height):
            return True
        return False


class LargeImage(Image):
    def __init__(self, image: np.ndarray, bboxes: Optional[List[List[int]]]=None, rel_path: Optional[str]=None):
        super().__init__(image=image, bboxes=bboxes, rel_path=rel_path)
        self.first_crop_locations: List[List[int]] = []
        self.second_crop_locations: List[List[int]] = []

    @property
    def image_w_patch_grid(self) -> np.ndarray:
        image_copy = self.image.copy()
        if not self.first_crop_locations is None:
            for crop in self.first_crop_locations:
                cv2.rectangle(image_copy, (crop[0], crop[1]), (crop[2], crop[3]), (255, 0, 0), 10)
                cv2.putText(image_copy, str(crop[2]-crop[0]) + "_" + str(crop[3]-crop[1]), (crop[2]+10, crop[3]),0,0.3,(255, 0, 0)) # Blue
        if not self.second_crop_locations is None:
            for crop in self.second_crop_locations:
                cv2.rectangle(image_copy, (crop[0], crop[1]), (crop[2], crop[3]), (0,0,255), 10)
                cv2.putText(image_copy, str(crop[2]-crop[0]) + "_" + str(crop[3]-crop[1]), (crop[2]+10, crop[3]),0,0.3,(0,0,255))
        return image_copy
    
    def clear_patches(self) -> None:
        self.first_crop_locations = []
        self.second_crop_locations = []
    
    def calculate_first_crop_location(self, desired_width: int, desired_height: int) -> List[List[int]]:
        """
        How to crop:
            as for x:
                width = div1 * desired_width + mod1
                mod1 = div2 * div1 + mod2
            as for y:
                height = div3 * desired_height + mod3
        """
        x_ys = []
        div1, mod1 = divmod(self.width, desired_width)
        div2, mod2 = divmod(mod1, div1)
        div3, mod3 = divmod(self.height, desired_height)
        for x_crop_num in range(div1+1):
            for y_crop_num in range(div3):
                if x_crop_num == div1:
                    x_ys.append(((x_crop_num-1)*desired_width + mod1, y_crop_num*desired_height))
                else:
                    x_ys.append((x_crop_num*desired_width, y_crop_num*desired_height))    
        return [[x, y, x+desired_width, y+desired_width] for x, y in x_ys]

    def get_crops(self, crop_locations:List[List[int]], desired_width: int, desired_height: int) -> List[Image]:
        cropped_images = []
        for location in crop_locations:
            
            image = Image(image=self.get_crop(location))
            cropped_images.append(image)
        return cropped_images

    def first_crop(self, desired_width: int=1024, desired_height: int=1024) -> Tuple[List[List[int]], List[Image]]:
        crop_locations = self.calculate_first_crop_location(desired_width, desired_height)
        crops = self.get_crops(crop_locations, desired_width, desired_height)
        return crop_locations, crops

    def pipe_det(self, patch_width: int, patch_height: int, first_thre: float=0.9, second_thre: float=0.9) -> None:
        assert not self.first_crop_locations, "already have not empty first_crop_locations"
        self.first_crop_locations, crops = self.first_crop()
        assert not self.bboxes, "have bboxes"
        self.bboxes = []
        self.class_confidences = []
        assert not self.second_crop_locations, "already have not empty second_crop_locations"
        for crop, location in zip(crops, self.first_crop_locations):
            crop.inference_of_objct_detection()
            crop.adapt_class_confidence_thre(thre=first_thre)
            for edge_bbox, edge_class_confidence, is_on_edge in zip(*crop.find_and_pop_edge_bbox()):
                if (is_on_edge[0] and is_on_edge[2]) or (is_on_edge[1] and is_on_edge[3]):
                    continue # ignore logic; removing bboxes that have patch width/height
                second_location = location.copy()
                if is_on_edge[0]:                    
                    second_location[0] -= patch_width//2
                    second_location[2] -= patch_width//2
                elif is_on_edge[2]:
                    second_location[0] += patch_width//2
                    second_location[2] += patch_width//2
                if is_on_edge[1]:
                    second_location[1] -= patch_height//2
                    second_location[3] -= patch_height//2
                elif is_on_edge[3]:
                    second_location[1] += patch_height//2
                    second_location[3] += patch_height//2
                if self.is_out_of_frame(second_location):
                    continue
                self.second_crop_locations.append(second_location)
            self.bboxes.extend(crop.get_global_bboxes(location))
            self.class_confidences.extend(crop.class_confidences)
            
        self.second_crop_locations = suppress_crop_locations(self.second_crop_locations)
        second_crops = self.get_crops(crop_locations=self.second_crop_locations, desired_width=patch_width, desired_height=patch_height)
        for second_crop, second_location in zip(second_crops, self.second_crop_locations):
            second_crop.inference_of_objct_detection()
            second_crop.adapt_class_confidence_thre(thre=second_thre)
            self.bboxes.extend(second_crop.get_global_bboxes(second_location))
            self.class_confidences.extend(second_crop.class_confidences)
        self.nms(iou_thre=0.5)

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
                data = Image(cropped_image.copy(), bboxes=cropped_bboxes)
            except AssertionError:
                print("some of bbox is out of image")
            dataset.append(data)

        return dataset
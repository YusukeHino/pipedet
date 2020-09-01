
import json
import os
import glob
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union, Optional

import cv2
import pandas as pd

from pipedet.structure.large_image import Image, LargeImage, BoxMode

if __name__ == "__main__":
    PATH_ROOT = os.path.join(os.environ["HOME"], "data")
    REL_PATH = os.path.join("for_rsm_detection", "separated")
    PATH_ANNOTATIONS = os.path.join(PATH_ROOT, REL_PATH)
    annotations_dir_list = [i for i in os.listdir(PATH_ANNOTATIONS)]
    for annotations_dir in annotations_dir_list: # for _ in [widht_height, (ex)4000_3000, ... ]
        path_annotations_file = os.path.join(PATH_ANNOTATIONS, annotations_dir, 'annotations.xml')
        tree = ET.parse(path_annotations_file)
        root = tree.getroot()
        for member in root.findall('object'):
            pass


class Dataset:
    """
    """

    def __init__(self, input_data_root: str) -> None :
        """
        Args:
            input_data_root (str): e.g. "/home/appuser/data"
        """
        self._input_data_root = input_data_root
        self.images: List[Union[Image, LargeImage]] = []
    
    def __iter__(self):
        return iter(self.images)

    def load_cvat_xmls(self, xml_files: List[str]) -> None:
        """
        Args:
            xml_files (List[str]): e.g. "/home/appuser/data/for_rsm_detection/separated/annotations/4000_3000/annotations.xml"
        """
        for xml_file in xml_files:
            self._load_cvat_xml(xml_file)

    def _load_cvat_xml(self, xml_file: str) -> None:
        """
        Load a single "annotations.xml" that is "cvat" format.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for tag_image in root.findall('image'):
            # rel_path is "for_rsm_detection/separated/4104_3006_for_test/20181205_007_0000004.png"
            rel_path = tag_image.attrib['name']
            width = int(tag_image.attrib['width'])
            height = int(tag_image.attrib['height'])
            bboxes = []
            class_confidences = []
            for tag_box in tag_image.findall('box'):
                bbox = [0] * 4
                bbox[0] = int(float(tag_box.attrib['xtl']))
                bbox[1] = int(float(tag_box.attrib['ytl']))
                bbox[2] = int(float(tag_box.attrib['xbr']))
                bbox[3] = int(float(tag_box.attrib['ybr']))
                if bbox[2] >= width:
                    bbox[2] -= 1
                if bbox[3] >= height:
                    bbox[3] -= 1
                bboxes.append(bbox)
                class_confidences.append((0., 1.))

            # Get image
            abs_image_path = os.path.join(self._input_data_root, rel_path)
            try:
                image = cv2.imread(abs_image_path)
            except:
                raise
            assert image.ndim == 3, image.ndim
            assert image.shape[:2] == (height, width), image.shape
            large_image = LargeImage(image = image, bboxes=bboxes, rel_path=rel_path)
            large_image.class_confidences = class_confidences
            self.images.append(large_image)

    def set_tag_from_rel_path(self) -> None:
        for image in self.images:
            dir_name = os.path.dirname(image.rel_path)
            assert dir_name.endswith(("train", "validate", "test")), "Cannot extract tag(train, test, or validate)."
            if dir_name.endswith("train"):
                tag = "TRAIN"
            elif dir_name.endswith("validate"):
                tag = "VALIDATE"
            elif dir_name.endswith("test"):
                tag = "TEST"
            image.tag = tag

    def generate_random_patch_dataset(self, root_of_rel_path_to_locate: str, desired_width: int=1024, desired_height: int=1024) -> "Dataset":
        random_patch_dataset = Dataset(self._input_data_root)
        for image in self.images:
            patches = image.make_dataset(desired_width, desired_height)
            assert image.rel_path is not None, f"rel_path is None."
            assert image.rel_path[-4] == ".", f"rel_path is {image.rel_path}"
            for i, patch in enumerate(patches):
                patch_name = image.filename[:-4] + "_" + str(i) + image.filename[-4:]
                patch.rel_path = os.path.join(root_of_rel_path_to_locate, patch_name)
                assert image.tag
                patch.tag = image.tag
                random_patch_dataset.images.append(patch)
        return random_patch_dataset

    def write_csv_for_automl(self, output_csv_path: str, location_in_gcp: str ,class_name: str="mirror") -> None:
        """
        Args:
            location_in_gcp (str): e.g. "gs://mirror-images/img/20200614_1024_1024/"
        """
        annotations: List[Tuple[str, str, str, float, float, None, None, float, float, None, None]] = []
        for image in self.images:
            for bbox in image.bboxes:
                assert image.tag
                annotation = (image.tag,
                    location_in_gcp + image.filename,
                    class_name,
                    float(bbox[0])/image.width, float(bbox[1])/image.height,None,None,
                    float(bbox[2])/image.width, float(bbox[3])/image.height,None,None)
                annotations.append(annotation)
        
        xml_df = pd.DataFrame(annotations)
        xml_df.to_csv(output_csv_path, header=None, index=False ,na_rep=None)

    def save_images(self):
        for image in self.images:
            abs_path = os.path.join(self._input_data_root, image.rel_path)
            cv2.imwrite(abs_path, image.image)
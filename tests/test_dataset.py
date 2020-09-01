
import os
import unittest

import numpy as np
import cv2

from pipedet.data.dataset import Dataset

_INPUT_DATA_ROOT = "/home/appuser/data"
_CVAT_XML_ROOT = "/home/appuser/data/for_rsm_detection/separated/annotations"

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset(_INPUT_DATA_ROOT)

    def tearDown(self):
        del self.dataset

    def test_load_cvat_xmls(self):
        xml_files = [os.path.join(_CVAT_XML_ROOT, xml_dir_name, "annotations.xml") for xml_dir_name in os.listdir(_CVAT_XML_ROOT)]
        self.dataset.load_cvat_xmls(xml_files)
        self.dataset.set_tag_from_rel_path()
        _root_of_rel_path_of_pathched_dataset = "/home/appuser/data/for_rsm_detection/cropped_1024_1024"
        random_patch_dataset = self.dataset.generate_random_patch_dataset(_root_of_rel_path_of_pathched_dataset)
        _output_csv_path = "/home/appuser/data/for_rsm_detection/integrated_cropped_1024_1024/integrated_rsm_labels_for_automl.csv"
        _location_in_gcp = "gs://mirror-images_20200729/img/"
        random_patch_dataset.write_csv_for_automl(_output_csv_path, _location_in_gcp)
        random_patch_dataset.save_images()
        
        _BREAKPOINT = []




import json
import os
import glob
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple

# TODO: locate
from .cropper import Images

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

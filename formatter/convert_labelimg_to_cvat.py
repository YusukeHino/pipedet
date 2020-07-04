
import json
import os
import glob
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple

"""
NEED to change TARGET variable to execute.
"""

def labelimg_to_cvat(list_tag_image: List[ET.Element], input_dir: str, rel_path: str, image_id: int) -> Tuple[List[ET.Element], int]:
    """
    """    
    for xml_file in glob.glob(input_dir + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename =  root.find('filename').text
        width = float(root.find('size')[0].text)
        height = float(root.find('size')[1].text)
        image_attrib = {
            'id': str(image_id),
            'name': os.path.join(rel_path, os.path.basename(input_dir), filename),
            'width': str(int(width)),
            'heigt': str(int(height)),
        }
        image_id += 1
        tag_image = ET.Element('image', attrib=image_attrib)
        for member in root.findall('object'):
            xmin = float(member.find('bndbox').find('xmin').text)
            ymin = float(member.find('bndbox').find('ymin').text)
            xmax = float(member.find('bndbox').find('xmax').text)
            ymax = float(member.find('bndbox').find('ymax').text)
            box_attrib = {
                'label': "mirror",
                'occluded': "0",
                'xtl': str(xmin),
                'ytl': str(ymin),
                'xbr': str(xmax),
                'ybr': str(ymax),
            }
            tag_box = ET.SubElement(tag_image, 'box', attrib=box_attrib)
        list_tag_image.append(tag_image)
    return list_tag_image, image_id


if __name__ == "__main__":
    PATH_ROOT = os.path.join(os.environ["HOME"], "data")
    REL_PATH = os.path.join("for_rsm_detection", "separated")
    PATH_OUTPUT = os.path.join(PATH_ROOT, REL_PATH, "annotations")
    TARGET = "4104_3006"
    out_file = "annotations.xml"
    PATH_INPUT = os.path.join(PATH_ROOT, REL_PATH)
    path_to_out_file = os.path.join(PATH_OUTPUT, TARGET, out_file)
    input_dir_list = [i for i in os.listdir(PATH_INPUT) if i.startswith(TARGET)]
    assert len(input_dir_list) == 3, input_dir_list
    
    tag_annotations = ET.Element('annotations')
    tag_versions = ET.SubElement(tag_annotations, 'versions')
    tag_versions.text = str(1.1)
    tag_meta = ET.SubElement(tag_annotations, 'meta')
    tag_task = ET.SubElement(tag_meta, 'task')
    tag_labels = ET.SubElement(tag_task, 'labels')
    tag_label = ET.SubElement(tag_labels, 'label')
    tag_name = ET.SubElement(tag_label, 'name')
    tag_name.text = 'mirror'
    tag_attributes = ET.SubElement(tag_label, 'attributes')
    list_tag_image = []
    image_id = 0
    for input_dir in input_dir_list: # for [train, validate, test]
        full_path_input = os.path.join(PATH_INPUT, input_dir)
        list_tag_image, image_id = labelimg_to_cvat(list_tag_image, full_path_input, REL_PATH, image_id)
    for tag_image in list_tag_image:
        tag_annotations.append(tag_image)
    out_tree = ET.ElementTree(tag_annotations)
    out_tree.write(path_to_out_file, encoding='UTF-8', xml_declaration=True)
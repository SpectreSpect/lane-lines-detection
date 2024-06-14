from .data_handler import DataHandler
import os
import xml.etree.ElementTree as ET
import numpy as np
import re
from ..data import Mask
from ..containers import ExplicitImageContainer
from ..data.annotation import Annotation
from typing import List


class CvatImageHandler(DataHandler):
    def __init__(self):
        pass
    
    def load(self, path: str) -> List[Annotation]:
        annotations: List[Annotation] = []
        
        annotation_file_path = os.path.join(path, "annotations.xml")
        images_path = os.path.join(path, "images")
        
        if not os.path.exists(annotation_file_path):
            raise Exception("Incorrect dataset structure: annotations.xml file is missing!")
        
        if not os.path.exists(images_path):
            raise Exception("Incorrect dataset structure: images folder is missing!")
        
        tree = ET.parse(annotation_file_path)
        root = tree.getroot()
        
        label_elements = root.find('.//labels').findall(".//label")
        label_names = [label_element.find('.//name').text for label_element in label_elements]
        label2id = {label_name: idx  for idx, label_name in enumerate(label_names)}
    
        image_elements = root.findall('.//image')

        for image_element in image_elements:
            width = int(image_element.attrib['width'])
            height = int(image_element.attrib['height'])
            
            image_shape = np.array([width, height])
            
            image_path = os.path.join(path, "images", image_element.attrib['name'])
            # image_path = os.path.join(path, image_element.attrib['name'])
            # print("YOU SHOULD OPEN cvat_image_handler.py AND CHANGE THE LINE ABOVE THIS PRINT!!!!!")
            
            image_container = ExplicitImageContainer(image_path)

            polygon_elements = image_element.findall('.//polygon')
            for polygon_element in polygon_elements:
                points_str = polygon_element.attrib['points']
                
                label = polygon_element.attrib['label']
                label = label2id[label]
                
                matches = re.findall(r'-?\d+\.\d+|-?\d+', points_str)
                points = np.array([float(match) for match in matches])
                points = points.reshape((-1, 2))
                
                points /= image_shape
                
                mask = Mask(points, label, image_container, False)
                annotations.append(mask)
        return annotations
    
    def save(self, path: str, validation_split: int):
        pass
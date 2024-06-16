from .data_handler import DataHandler
import os
import xml.etree.ElementTree as ET
import numpy as np
import re
from ..data import Mask
from ..containers import ExplicitImageContainer
import shutil
from typing import List
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle


class YoloImageHandler(DataHandler):
    def __init__(self):
        pass
    
    def load(self, path: str) -> list:
        pass
    
    def save(self, annotation_bundels: List[AnnotationBundle], path: str, validation_split: int):
        os.makedirs(path, exist_ok=True)
        
        train_image_dir = os.path.join(path, "images", "train")
        valid_image_dir = os.path.join(path, "images", "valid")
        train_label_dir = os.path.join(path, "labels", "train")
        valid_label_dir = os.path.join(path, "labels", "valid")
        
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(valid_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(valid_label_dir, exist_ok=True)
        
        
        num_train = int(len(annotation_bundels) * (1.0 - validation_split))
        for bundle_id, annotation_bundel in enumerate(annotation_bundels):
            output_str = ""
            for annotation in annotation_bundel.annotations: 
                output_str += str(annotation.label)
                for point in annotation.points:
                    output_str += f" {point[0]} {point[1]}"
                output_str += "\n"
                
            image_name = annotation_bundel.image_container.image_name
                
            if bundle_id < num_train:
                label_path = os.path.join(train_label_dir, image_name + ".txt")
            else:
                label_path = os.path.join(valid_label_dir, image_name + ".txt")
                
            with open(label_path, 'w') as file:
                file.write(output_str)
            
            if bundle_id < num_train:
                annotation_bundel.image_container.save_image(train_image_dir)
            else:
                annotation_bundel.image_container.save_image(valid_image_dir)
            
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


class YoloImageHandler(DataHandler):
    def __init__(self):
        pass
    
    def load(self, path: str) -> list:
        pass
    
    def save(self, annotations: List[Annotation], path: str, validation_split: int):
        os.makedirs(path, exist_ok=True)
        
        print("VALIDATION SPLIT DOESN'T WORK!!!! YOU NEED TO FIX IT SOMEHOW!!! DON'T FORGET!!")
        
        train_image_dir = os.path.join(path, "images", "train")
        valid_image_dir = os.path.join(path, "images", "valid")
        train_label_dir = os.path.join(path, "labels", "train")
        valid_label_dir = os.path.join(path, "labels", "valid")
        
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(valid_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(valid_label_dir, exist_ok=True)
        
        
        num_train = int(len(annotations) * (1.0 - validation_split))
        for idx, annotation in enumerate(annotations):
            output_str = ""
            
            output_str += str(annotation.label)
            for point in annotation.points:
                output_str += f" {point[0]} {point[1]}"
            output_str += "\n"
            
            image_name = annotation.image_container.image_name
            
            if idx < num_train:
                label_path = os.path.join(train_label_dir, image_name + ".txt")
            else:
                label_path = os.path.join(valid_label_dir, image_name + ".txt")
                
            if os.path.exists(label_path):
                with open(label_path, 'a') as file:
                    file.write(output_str)
            else:
                with open(label_path, 'w') as file:
                    file.write(output_str)
            
            if idx < num_train:
                annotation.image_container.save_image(train_image_dir)
            else:
                annotation.image_container.save_image(valid_image_dir)
            
            
            
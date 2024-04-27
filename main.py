from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *
from src.dataset_balancing import *
from src.from_xml_to_yolo import *
import re


def load_image(path: str):
    image = np.asarray(Image.open(path))
    return image


def load_images(path: str, max_images_count=-1) -> list:
    images = []
    image_names = []
    images_loaded = 0
    for image_name in os.listdir(path):
        if max_images_count >= 0:
            if images_loaded >= max_images_count:
                break
        image_path = os.path.join(path, image_name)
        if os.path.isfile(image_path):
            image = load_image(image_path)
            images.append(image)
            image_names.append(image_name)
            images_loaded += 1
    return images, image_names


if __name__ == "__main__":
    # from_cvat_to_yolo("data/new_yolo_data/images",
    #                   "data/new_yolo_data/labels",
    #                   get_label_names("config.yaml"),
    #                   input_videos_path="data/new_data/videos",
    #                   input_labels_path="data/new_data/labels")
    

    # from_cvat_to_yolo("data/road-to-adler-2_double-white-solid/road-to-adler-2(double-white-solid).mp4",
    #                   "data/road-to-adler-2_double-white-solid/annotations.xml",
    #                   "tmp/temp_yolo_labels/images", 
    #                   "tmp/temp_yolo_labels/labels",
    #                   get_label_names("config.yaml"))
    

    
    LaneMask.visualize_masks(masks_path="data/new_yolo_data/labels/1b7b711d-e92c-448b-b372-17dbeeea9de6.txt",
                             image_path="data/new_yolo_data/images/1b7b711d-e92c-448b-b372-17dbeeea9de6.jpg", mask_alpha=0.8)
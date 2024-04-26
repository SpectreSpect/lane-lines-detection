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

    from_cvat_to_yolo("tmp/test_input_cvat_data", 
                      "tmp/temp_yolo_labels/images", 
                      "tmp/temp_yolo_labels/labels",
                      get_label_names("config.yaml"))
    # lane_line_batches = get_lane_lines_from_xml("data/annotations.xml", get_label_names("config.yaml"))
    # lane_mask_batches = LaneMask.from_line_batches_to_mask_batches(lane_line_batches, (1920, 1080))

    # from_mask_batches_to_yolo(lane_mask_batches, "tmp/temp_yolo_labels")



    # print(lane_mask_batches)

    # print(lane_lines)
    # print('sdfsd')
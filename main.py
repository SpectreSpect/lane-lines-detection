from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *
from src.dataset_balancing import *
from src.reinforcement_data import *
from src.from_xml_to_yolo import *
import re
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader


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
    model = YOLO("models/robolife-detection-yolov8l-seg/model.pt")
    model.to("cuda")

    class_names = model.names
    for key in class_names:
        print(f"Class {key}: {class_names[key]}")
    
    image = cv2.imread("data/yolov8-new-data1-val015-fmasks/images/train/00f929d2-69da-45d3-ab8e-aa9c5b1d1a10.jpg")


    bounding_boxes = BoundingBox.from_yolo("tmp/00f929d2-69da-45d3-ab8e-aa9c5b1d1a10.txt")
    for box in bounding_boxes:
        box.draw_on_image(image, class_names)

    
    cv2.imshow('Loaded Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # data_generator = DataGenerator(model)
    # data_generator.generate("data/yolov8-new-data1-val015-fmasks/images/train", 
    #                         "data/new-data-lables/train", 10)
    
    # plastic_drum_images_path = ""
    # plastic_drum_labels_output_path = ""
    # data_generator.generate(plastic_drum_images_path, 
    #                         plastic_drum_labels_output_path, 10)
    # cvat_labels_path = ""
    # data_generator.merge_cvat_images_to_yolo(plastic_drum_labels_output_path, cvat_labels_path)

    
    


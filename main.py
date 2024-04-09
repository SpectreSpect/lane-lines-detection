from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math



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
    lane_model = LaneLineModel("models/lane_line_model/model.pt")
    # lane_model.train("E:/Freedom/Projects/ultralytics-dataset-visualisation/data/yolov8_medium-1000_2", 1)


    # model = lane_model.model

    # args = {''}

    # model.train()

    # image = load_image("data/yolov8_medium-1000_2/images/train/155320867831365400.jpg")  
    # lane_model.visualize_prediction(image)
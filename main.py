from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *


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

    image1 = load_image("data/yolov8_medium-1000_2/images/train/155320867831365400.jpg")
    image2 = load_image("data/yolov8_medium-1000_2/images/train/150776258831391200.jpg")
    image3 = load_image("data/yolov8_medium-1000_2/images/train/155727751413760500.jpg")
    image4 = load_image("data/yolov8_medium-1000_2/images/train/155727750023736100.jpg")

    images = [image1, image2, image3, image4]

    predictions = lane_model.model.predict(images)
    lines_batch = lane_model.predict(images) # Проверка get_lines()

    images_to_draw = np.copy(images)
    draw_segmentation(images_to_draw, predictions)
    show_images(images_to_draw)
    #lane_model.visualize_prediction(image)
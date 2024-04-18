from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *
import torch


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
    lane_model = LaneLineModel("models/sizefull-ep20/model.pt")
    # lane_model.train("data/yolov8-size1000-val02-fmasks", 2, output_directory="runs")
    #lane_model.predict()

    # image1 = load_image("data/yolov8_medium-1000_2/images/train/155727749493776200.jpg")
    # image2 = load_image("data/yolov8_medium-1000_2/images/train/150776258831391200.jpg")
    # image3 = load_image("data/yolov8_medium-1000_2/images/train/155320868381244300.jpg")
    # image4 = load_image("data/yolov8_medium-1000_2/images/train/155320867831365400.jpg")

    # images = [image1, image2, image3, image4]
    # predictions = lane_model.model.predict(images)
    # batch_lines = lane_model.get_lines(predictions)

    # images_to_draw = np.copy(images)
    # draw_segmentation(images_to_draw, predictions)
    # draw_curves(images_to_draw, batch_lines)
    # show_images(images_to_draw)

    view_prediction_video(lane_model, "data/videos/road-video-russia.mp4")

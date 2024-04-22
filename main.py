from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *
from src.dataset_balancing import *


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
    model = LaneLineModel("models/sizefull-ep20/model.pt")

    preview_prediction_video(model, "data/videos/road-video-russia.mp4", "config.yaml")



    # images, predictions = view_prediction_video(model, "data/videos/road-video-russia.mp4", True)

    # print(f"images len: {len(images)}   preds len: {len(predictions)}")
    # print(type(predictions[0][0][0]))

    # print_labels_distribution_stats("data/yolov8-sizefull-val02-fmasks/labels/train", "config.yaml")
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
import time


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
    model = LaneLineModel("models/data-openlane-vs02-e50/model.pt")

    x_data = []
    y_data = []

    n_frame = [0]
    start_time = time.time()

    def start_frame_callback():
        global start_frame_time
        start_frame_time = time.time()
    
    def end_frame_callback():
        end_frame_time = time.time()
        
        frame_time = end_frame_time - start_frame_time 
        
        x_data.append(n_frame[0])
        y_data.append(frame_time)
        n_frame[0] += 1

    view_prediction_video(model, 
                          "data/videos/road-video-yellow-solid.mp4", 
                          get_label_names("config.yaml"), 
                          1000, 
                          start_frame_callback=start_frame_callback, 
                          end_frame_callback=end_frame_callback,
                          max_frames=200)

    y_data = np.array(y_data)

    y_mean = y_data.mean()
    y_std = y_data.std()

    print(float(np.array(y_data).mean()) * 1000)
    plt.plot(x_data, y_data)
    plt.ylim(y_mean - y_std, y_mean + y_std)
    plt.xlabel("Номер кадра")
    plt.ylabel("Время, мс.")
    plt.title("График времени предсказания")
    plt.show()
from src.LaneLineModel import LaneLineModel
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *
from src.dataset_balancing import *
from src.reinforcement_data import *
from src.from_xml_to_yolo import *
from src.dataset import *
import re
import time

def load_image(path: str):
    image = np.asarray(Image.open(path))
    if image.shape[2] == 4:
        image = image[..., :3]
    return image


if __name__ == "__main__":
    model = LaneLineModel(r"models\LLD\model.pt")

    view_prediction_video(model, r"data\videos\alabuga.mp4", model.model.names, 1000)
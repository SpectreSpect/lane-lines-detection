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


if __name__ == "__main__":
   image_path = "data/rtsi/images/valid/autosave24_10_2013_11_21_12_0.jpg"
   label_path = "data/rtsi/labels/valid/autosave24_10_2013_11_21_12_0.txt"
   show_bbox_yolo_dataset(image_path, label_path)


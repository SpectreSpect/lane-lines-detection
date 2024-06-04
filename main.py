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
from src.dataset import YoloImageDataset
import re
import time


if __name__ == "__main__":
    test_dataset = YoloImageDataset.create_dataset("tmp/test-dataset", 
                                                   "tmp/test-images", "tmp/test-labels", 
                                                #    label_name_list=["double-dash", "some-label", "hello"],
                                                   config_path="config.yaml",
                                                   validation_split=0.2)

    # model = LaneLineModel("models/LLD-2.pt")

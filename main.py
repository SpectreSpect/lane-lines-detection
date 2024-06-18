# from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import math
# from src.utils import *
# from src.dataset_balancing import *
# from src.reinforcement_data import *
# from src.from_xml_to_yolo import *
# from src.dataset import *
# import re
# import time
from src.converter.containers import ExplicitImageContainer
from src.converter.data import Mask
from src.converter.core import Core
from src.converter.handlers.data_handler_factory import DataHandlerFactory
from src.utils import show_bbox_yolo_dataset
import os


if __name__ == "__main__":
    handler = DataHandlerFactory.create_handler("traffic-light-detection-dataset")
    handler.min_side_size = 100

    core = Core(r"data\traffic-light-detection-dataset\train_dataset", handler=handler)
    core.export("data/test-yolo-export", "yolo", 0.2)

    # show_bbox_yolo_dataset(r"data\test-yolo-export\images\train\00003.jpg",
    #                        r"data\test-yolo-export\labels\train\00003.txt")


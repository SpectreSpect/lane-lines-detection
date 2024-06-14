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


if __name__ == "__main__":
    image_container = ExplicitImageContainer("data/rm-dataset/images/images/Screenshot 2024-06-13 095752.png")
    mask = Mask([], 5, image_container, False) # why does the "points" argument has the "any" type?
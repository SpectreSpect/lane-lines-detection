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
from src.utils import *
import os
import yaml


if __name__ == "__main__":
    core = Core(r"data\datasets\sign-detection\rtsi\rtsi", "yolo")
    bundels = core._annotation_bundles

    core._annotation_bundles = list(filter(lambda bundle: any(map(lambda annotation: annotation.label == "3_18_1", bundle.annotations)), bundels))
    core.export(r"data\datasets\sign-detection\rtsi-3_18_1", "yolo", 0)

    # core._annotation_bundles = list(filter(lambda bundle: any(map(lambda annotation: annotation.label == "5_15_3", bundle.annotations)), bundels))
    # core.export(r"data\datasets\sign-detection\rtsi-5_15_3", "yolo", 0)

    # core._annotation_bundles = list(filter(lambda bundle: any(map(lambda annotation: annotation.label == "5_15_5", bundle.annotations)), bundels))
    # core.export(r"data\datasets\sign-detection\rtsi-5_15_5", "yolo", 0)

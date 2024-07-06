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
from src.converter.handlers.cvat_video_handler import CvatVideoHandler
from src.converter.containers.video_image_container import VideoImageContainer
from src.converter.data.box import Box
from src.utils import *

import os
from ultralytics import YOLO
from PIL import Image
from src.converter.models import YoloSegmentationModel
from src.converter.visualizer import Visualizer
from src.converter.visualizer import ColorKeypoint
from src.converter.visualizer import ColorMap
from src.converter.visualizer import Pallete

def load_image(path: str):
    image = np.asarray(Image.open(path))
    if image.shape[2] == 4:
        image = image[..., :3]
    return image


if __name__ == "__main__":    
    segmment1_core = Core("data/TrafficLight/segment1", "cvat-image")
    # segmment2_core = Core("data/Traffic Light/segment 2", "cvat-image")
    # segmment3_core = Core("data/Traffic Light/segment 3", "cvat-image")
    segmment4_core = Core("data/TrafficLight/segment4", "cvat-image")
    # segmment5_core = Core("data/TrafficLight/segment5", "cvat-image")
    segmment6_core = Core("data/TrafficLight/segment6", "cvat-image")
    tld_core = Core("data/TLD-dataset-3", "yolo")
    
    # segmment1_core.merge(segmment2_core)
    # segmment1_core.merge(segmment3_core)
    segmment1_core.merge(segmment4_core)
    # segmment1_core.merge(segmment5_core)
    segmment1_core.merge(segmment6_core)
    segmment1_core.merge(tld_core)

    segmment1_core.export("data/TLD-dataset-4", "yolo", 0.2)
    # tld_core = Core("data/TLD-dataset-4", "yolo")
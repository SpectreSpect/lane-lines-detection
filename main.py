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
    segment1_core = Core("data/Road segmentation/Segment-1  (done)", "cvat-video")
    segment2_core = Core("data/Road segmentation/Segment-2 (done)", "cvat-video")
    segment3_core = Core("data/Road segmentation/Segment-3 (done)", "cvat-video")
    zed_segment1_core = Core("data/zed-morning/segment-1 (done)", "cvat-image")

    segment1_core.merge(zed_segment1_core)
    segment1_core.merge(segment2_core)
    segment1_core.merge(segment3_core)
    

    segment1_core.export("data/RC-dataset-3", "yolo", 0.2)
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
from src.utils import *

import os
import yaml


if __name__ == "__main__":
    core = Core("data/segment-1-seg", "cvat-video")
    core.export("data/segmet-1-seg-yolo", "yolo", 0.2)

    # cvat_video_handler = CvatVideoHandler()
    # annotation_bundels, label_names = cvat_video_handler.load("data/segment-1-seg")

    # print(len(annotation_bundels))

    # print(label_names)

    # video_image_container = VideoImageContainer("data/segment-1-seg/video.mp4", 100)
    # image = video_image_container.get_image()

    # cv2.imshow("Image", annotation_bundels[0].image_container.get_image())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


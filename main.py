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
import yaml


if __name__ == "__main__":
    rc_dataset_core = Core("data/RC-dataset", "yolo")
    
    # for annotation_bundle in rc_dataset_core._annotation_bundles:
    #     for annotation in annotation_bundle._annotations:
    #         if isinstance(annotation, Box):
    #             print("BOOOOX")
    
    core_segment_2 = Core("data/segment-2-seg", "cvat-video")
    
    rc_dataset_core.merge(core_segment_2)
    
    rc_dataset_core.export("data/RC-dataset-2", "yolo", 0.2)


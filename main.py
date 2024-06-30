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
    keypoints = [ColorKeypoint([255, 0, 0], 0.3),
                 ColorKeypoint([0, 255, 0], 0.8),
                 ColorKeypoint([0, 0, 255], 0.9)]

    color_map = ColorMap.from_keypoints(keypoints)

    
    


    # print(f"Colors: {pallete.colors}\n")
    # print(f"Colors: {pallete.key_values}\n")
    

    # color = color_map.get_color(0.88)

    # print(color)

    model = YoloSegmentationModel("models/LLD/model.pt")

    core = Core("data/rm-dataset-yolo", "yolo")
    # core.annotate(model)
    model.annotate([core._annotation_bundles[1]])

    core._label_names = list(set(core._label_names + model.get_label_names()))

    pallete = Pallete.from_colormap(core._label_names, color_map)

    visualizer = Visualizer()

    visualizer.show_annotation_bundle(core._annotation_bundles[1], pallete)
    

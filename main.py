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

def load_image(path: str):
    image = np.asarray(Image.open(path))
    if image.shape[2] == 4:
        image = image[..., :3]
    return image


if __name__ == "__main__":
    # model = YoloSegmentationModel("models/LLD/model.pt")
    
    image_container = ExplicitImageContainer("data/segmet-1-seg-yolo/images/train/0c71095e-fbf9-4d60-8673-4a0315241266.jpg")
    
    # model.predict([image_container])
    
    images = [load_image("data/segmet-1-seg-yolo/images/train/0c71095e-fbf9-4d60-8673-4a0315241266.jpg")]
    
    yolo_model = YOLO("models/LLD/model.pt")
    
    yolo_model.predict(images)
    
    # core = Core(r"data\datasets\sign-detection\rtsi\rtsi", "yolo")
    # bundels = core._annotation_bundles

    # core._annotation_bundles = list(filter(lambda bundle: any(map(lambda annotation: annotation.label == "3_18_1", bundle.annotations)), bundels))
    # core.export(r"data\datasets\sign-detection\rtsi-3_18_1", "yolo", 0)

    # core._annotation_bundles = list(filter(lambda bundle: any(map(lambda annotation: annotation.label == "5_15_3", bundle.annotations)), bundels))
    # core.export(r"data\datasets\sign-detection\rtsi-5_15_3", "yolo", 0)

    # core._annotation_bundles = list(filter(lambda bundle: any(map(lambda annotation: annotation.label == "5_15_5", bundle.annotations)), bundels))
    # core.export(r"data\datasets\sign-detection\rtsi-5_15_5", "yolo", 0)

    # core = Core(r'data/RC-segmet-1-seg-yolo/segmet-1-seg-yolo', 'yolo')
    # core.annotate(YOLOModel('data/lld-pytorch-level-5-v1-v1/runs/segment/train/weights/best.pt', is_segmentation=True))
    # core.export(r'data/RC-segmet-1-seg-yolo/segmet-1-seg-yolo/test', 'yolo', 0)

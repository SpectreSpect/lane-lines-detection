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
    model = YoloSegmentationModel("models/LLD/model.pt")
    
    image_container = ExplicitImageContainer("data/segmet-1-seg-yolo/images/train/0d510753-92af-4fbc-ab4f-9fa3b88c4893.jpg")
    
    annotation_bundels = model.predict([image_container])
    print(annotation_bundels[0])
    

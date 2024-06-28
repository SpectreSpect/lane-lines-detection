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

def load_image(path: str):
    image = np.asarray(Image.open(path))
    if image.shape[2] == 4:
        image = image[..., :3]
    return image


if __name__ == "__main__":
    core = Core("data/RC-dataset-2", "yolo")

    annotation_bundle = core._annotation_bundles[0]
    image = annotation_bundle.image_container.get_image()
    mask = annotation_bundle.annotations[0]



    visualizer = Visualizer()
    # visualizer.draw_mask()

    visualizer.show_image(image)


    # model = YoloSegmentationModel("models/LLD/model.pt")
    
    # image_container = ExplicitImageContainer("data/RC-dataset-2/images/train/0a839bcc-95d3-4d65-ad64-897e9f5b82b1.jpg")
    
    # print(model.get_label_names())

    # annotation_bundels = model.predict([image_container])
    # print(annotation_bundels[0])
    

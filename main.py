import numpy as np
from PIL import Image
from src.converter.core.core import Core
from typing import List
from src.converter.models.yolo_detection_model import YoloDetectionModel
from src.converter.models.yolo_segmentation_model import YoloSegmentationModel
from src.converter.containers.video_image_container import VideoImageContainer
from src.converter.visualizer.palette.palette_register import PaletteRegister, palette_register
import cv2

def load_image(path: str):
    image = np.asarray(Image.open(path))
    if image.shape[2] == 4:
        image = image[..., :3]
    return image

if __name__ == "__main__":    
    rtsi_core = Core(r"data\datasets\sign-detection\rtsi-filtered-100", "yolo")
    rtsr_core = Core(r"data\datasets\sign-detection\rtsr", "yolo")
    magistral_core = Core(r"data\datasets\sign-detection\magistral-cvat-dataset", "cvat-image")

    sign_1_6_core = Core(r"data\datasets\sign-detection\signs\1_6", "cvat-image")
    sign_1_35_core = Core(r"data\datasets\sign-detection\signs\1_35", "cvat-image")
    sign_3_20_core = Core(r"data\datasets\sign-detection\signs\3_20", "cvat-image")
    sign_3_22_core = Core(r"data\datasets\sign-detection\signs\3_22", "cvat-image")
    sign_3_23_core = Core(r"data\datasets\sign-detection\signs\3_23", "cvat-image")
    sign_6_3_2_core = Core(r"data\datasets\sign-detection\signs\6_3_2", "cvat-image")

    rtsi_core.merge(rtsr_core)
    rtsi_core.merge(magistral_core)
    rtsi_core.merge(sign_1_6_core)
    rtsi_core.merge(sign_1_35_core)
    rtsi_core.merge(sign_3_20_core)
    rtsi_core.merge(sign_3_22_core)
    rtsi_core.merge(sign_3_23_core)
    rtsi_core.merge(sign_6_3_2_core)

    signs = [
        "1_1", 
        "1_6",
        "1_8",
        "1_22",
        "1_25",
        "1_31",
        "1_33",
        "1_35",
        "2_1",
        "2_2",
        "2_3_1",
        "2_4",
        "2_5",
        "2_6",
        "2_7",
        "3_1",
        "3_13",
        "3_18_1",
        "3_18_2",
        "3_20",
        "3_21",
        "3_22",
        "3_23",
        "3_24",
        "3_25",
        "3_27",
        "3_28",
        "3_31",
        "4_1_1",
        "4_3",
        "5_1",
        "5_5",
        "5_6",
        "5_14",
        "5_14_1",
        "5_16",
        "5_19_1",
        "5_19_2",
        "5_20",
        "6_3_2",
        "6_4",
        "7_3",
        "7_4"
    ]

    remaining_core = rtsi_core.filter_and_split(signs, 50)
    rtsi_core.count_annotations(verbose=1)

    remaining_core.export(r"data\datasets\sign-detection\remaining-rtsi-dataset", "yolo", 0.2)
    rtsi_core.export(r"data\datasets\sign-detection\tld-dataset-without-annotation", "yolo", 0.2)

    
import numpy as np
from PIL import Image
from src.converter.core.core import Core
from typing import List

def load_image(path: str):
    image = np.asarray(Image.open(path))
    if image.shape[2] == 4:
        image = image[..., :3]
    return image

if __name__ == "__main__":    
    rtsi_core = Core(r"data\datasets\sign-detection\rtsi-filtered-100", "yolo")
    rtsr_core = Core(r"data\datasets\sign-detection\rtsr", "yolo")
    magistral_core = Core(r"data\datasets\sign-detection\magistral-dataset", "yolo")
    sign_1_6_core = Core(r"data\datasets\sign-detection\signs\1_6", "cvat-image")
    sign_1_35_core = Core(r"data\datasets\sign-detection\signs\1_35", "cvat-image")
    sign_3_22_core = Core(r"data\datasets\sign-detection\signs\3_22", "cvat-image")
    sign_3_23_core = Core(r"data\datasets\sign-detection\signs\3_23", "cvat-image")
    sign_6_2_3_core = Core(r"data\datasets\sign-detection\signs\6_3_2", "cvat-image")

    rtsi_core.merge(rtsr_core)
    rtsi_core.merge(magistral_core)
    rtsi_core.merge(sign_1_6_core)
    rtsi_core.merge(sign_1_35_core)
    rtsi_core.merge(sign_3_22_core)
    rtsi_core.merge(sign_3_23_core)
    rtsi_core.merge(sign_6_2_3_core)

    bundles = rtsi_core._annotation_bundles
    
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

    rtsi_core.filter_bundles(signs, 200)
    rtsi_core.count_annotations(verbose=1)

    rtsi_core.export(r"data\datasets\sign-detection\rtsi-filtered-merged-100", "yolo", 0.2)
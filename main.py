import numpy as np
from PIL import Image
from src.converter.core.core import Core

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

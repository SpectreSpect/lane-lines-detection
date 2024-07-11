import numpy as np
from PIL import Image
from src.converter.core.core import Core
from typing import List
from src.converter.models.yolo_detection_model import YoloDetectionModel
from src.converter.visualizer.palette.palette_register import PaletteRegister, palette_register
import cv2

def load_image(path: str):
    image = np.asarray(Image.open(path))
    if image.shape[2] == 4:
        image = image[..., :3]
    return image

if __name__ == "__main__":    
    model = YoloDetectionModel(r"models\SDM\model.pt")
    level_5_core = Core(r"data\datasets\sign-detection\rtsr", "yolo")
    
    for bundle in level_5_core._annotation_bundles:
        predicted_bundle = model.predict([bundle.image_container])[0]
        image = predicted_bundle.draw_pp(palette=palette_register.palettes["rainbow"], target_width=1000)
        
        cv2.imshow("image", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
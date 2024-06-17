from PIL import Image
import numpy as np
import shutil
import os
from .image_container import ImageContainer

class ExplicitImageContainer(ImageContainer):
    def __init__(self, image_path: str):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        super().__init__(image_name)
        
        self._image_path = os.path.normpath(image_path)
    
    def get_image(self):
        image = np.array(Image.open(self._image_path))
        return image
    
    def save_image(self, dir_path: str):
        base = os.path.basename(self._image_path)
        shutil.copy(self._image_path, os.path.join(dir_path, base))
        

from PIL import Image
import numpy as np
import shutil
import os
from .image_container import ImageContainer

class ExplicitImageContainer(ImageContainer):
    def __init__(self, image_path: str):
        self.image_path = image_path
    
    def get_image(self):
        image = np.array(Image.open(self.image_path))
        return image
    
    def save_image(self, path: str):
        base, ext = os.path.splitext(path)
        
        if ext:
            shutil.copy(self.image_path, path)
        else:
            image_name = os.path.basename(self.image_path)
            shutil.copy(self.image_path, os.path.join(path, image_name))
        

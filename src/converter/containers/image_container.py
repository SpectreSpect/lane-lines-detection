from abc import ABC, abstractmethod
import numpy as np

class ImageContainer:
    @abstractmethod
    def get_image(self) -> np.array:
        pass
    
    @abstractmethod
    def save_image(self, path: str) -> np.array:
        pass
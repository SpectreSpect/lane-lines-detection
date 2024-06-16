from abc import ABC, abstractmethod
import numpy as np
from ..containers.explicit_image_container import ExplicitImageContainer

class Annotation(ABC):
    def __init__(self, 
                 points: np.ndarray, 
                 label: int, 
                 image_container: ExplicitImageContainer,
                 is_valid: bool):
        self._points = points
        self._label = label
        self._image_container = image_container
        self._is_valid = is_valid
    
    @property
    def points(self):
        return self._points
    
    @property
    def label(self):
        return self._label
    
    @property
    def image_container(self):
        return self._image_container
    
    @property
    def is_valid(self):
        return self._is_valid
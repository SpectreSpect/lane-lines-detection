import numpy as np
from ..containers.explicit_image_container import ExplicitImageContainer
from .annotation import Annotation

class Mask(Annotation):
    def __init__(self,
                 points: np.ndarray, 
                 label: int, 
                 image_container: ExplicitImageContainer,
                 is_valid: bool):
        super().__init__(points, label, image_container, is_valid)
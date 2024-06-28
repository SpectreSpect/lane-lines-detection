from typing import List
from .annotation import Annotation
from ..containers.image_container import ImageContainer

class AnnotationBundle():
    def __init__(self, annotations: List[Annotation], image_container: ImageContainer):
        self._annotations = annotations
        self._image_container = image_container
    
    @property
    def annotations(self):
        return self._annotations
    
    @annotations.setter
    def annotations(self, new_value):
        self._annotations = new_value
    
    @property
    def image_container(self):
        return self._image_container
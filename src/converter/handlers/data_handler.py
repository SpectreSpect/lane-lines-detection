from abc import ABC, abstractmethod
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from typing import List

class DataHandler(ABC):
    @abstractmethod
    def load(self, path: str) -> List[AnnotationBundle]:
        pass
    
    @abstractmethod
    def save(self, annotation_bundels: List[AnnotationBundle], path: str, validation_split: int):
        pass
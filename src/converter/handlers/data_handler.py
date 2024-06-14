from abc import ABC, abstractmethod
from ..data.annotation import Annotation
from typing import List

class DataHandler(ABC):
    @abstractmethod
    def load(self, path: str) -> List[Annotation]:
        pass
    
    @abstractmethod
    def save(self, annotations: list, path: str, validation_split: int):
        pass
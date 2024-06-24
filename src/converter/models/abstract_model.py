from abc import abstractmethod
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from ..containers.image_container import ImageContainer
from typing import List
import numpy as np


class AbstractModel:
    def __init__(self):
        pass
    
    # В зависимости от формата предсказания конкретной модели, формирует унифицированный список объектов класса Annotation
    @abstractmethod
    def predict(self, image_containers: List[ImageContainer]) -> List[AnnotationBundle]:
        pass

    # Возвращает унифицированный список AnnotationBundle в зависимости от результата работы метода predict конкретной модели
    @abstractmethod
    def annotate(self, annotation_bundles: List[AnnotationBundle]):
        pass

    # Возвращает список классов модели в зависимости от формата их хранения
    @abstractmethod
    def get_label_names(self) -> list:
        pass

from abc import abstractmethod
from ..converter.data.annotation import Annotation
from ..converter.data.annotation_bundle import AnnotationBundle
from typing import List
import numpy as np


class AbstractModel:
    def __init__(self, model_file_path: str):
        self._model_file_path = model_file_path

    # Возвращает унифицированный список AnnotationBundle в зависимости от результата работы метода predict конкретной модели
    @abstractmethod
    def annotate(self, annotation_bundles: List[AnnotationBundle]) -> List[AnnotationBundle]:
        pass

    # В зависимости от формата предсказания конкретной модели, формирует унифицированный список объектов класса Annotation
    @abstractmethod
    def predict(self, image: np.array) -> List[Annotation]:
        pass

    # Возвращает список классов модели в зависимости от формата их хранения
    @abstractmethod
    def get_label_names(self) -> List:
        pass

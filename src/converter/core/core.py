from ..models import AbstractModel
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from typing import Callable, List
from ..handlers import DataHandler
from ..handlers.data_handler_factory import DataHandlerFactory


class Core():
    def __init__(self, dataset_path: str, dataset_format: str="yolo", handler: DataHandler=None):
        self._label_names = []
        self._validation_split = 0.0
        self._dataset_format = dataset_format
        self._dataset_path = dataset_path

        handler = DataHandlerFactory.create_handler(dataset_format) if handler is None else handler
        self._annotation_bundles, self._label_names = handler.load(dataset_path)


    def export(self, output_path: str, dataset_format: str, validation_split: float):
        handler = DataHandlerFactory.create_handler(dataset_format)
        handler.save(self._annotation_bundles, self._label_names, output_path, validation_split)
    

    def merge(self, core):
        self._annotation_bundles += core._annotation_bundles
        self._label_names = list(set(self._label_names + core._label_names))

    def annotate(self, model: AbstractModel, verbose=True):
        model.annotate(self._annotation_bundles, verbose=verbose)
        self._label_names = list(set(self._label_names + model.get_label_names()))

    # @staticmethod
    # def filter_bundles(annotation_bundles: List[AnnotationBundle], key: Callable[[Annotation], bool]):
    #     filtered_annotation_bundel: List[AnnotationBundle] = []
    #     for annotation_bundel in annotation_bundles:
    #         for annotation in annotation_bundel.annotations:
    #             if key(annotation):
    #                 filtered_annotation_bundel.append(annotation_bundel)
    #     return filtered_annotation_bundel
        
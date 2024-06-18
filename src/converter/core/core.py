from ..handlers.data_handler_factory import DataHandlerFactory
from ..handlers import DataHandler
from typing import List

class Core():
    def __init__(self, dataset_path: str, dataset_format: str="yolo", handler: DataHandler=None):
        self._label_names = []
        self._validation_split = 0.0
        self._dataset_format = dataset_format
        
        handler = DataHandlerFactory.create_handler(dataset_format) if handler is None else handler
        self._annotation_bundles = handler.load(dataset_path)
    
    def export(self, output_path: str, dataset_format: str, validation_split: float):
        handler = DataHandlerFactory.create_handler(dataset_format)
        handler.save(self._annotation_bundles, output_path, validation_split)
        
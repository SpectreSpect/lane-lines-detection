from ..handlers.data_handler_factory import DataHandlerFactory
from ..handlers import DataHandler
from typing import List

class Core():
    def __init__(self, dataset_path: str, dataset_format: str):
        self._label_names = []
        self._validation_split = 0.0
        self._dataset_format = dataset_format
        
        handler = DataHandlerFactory.create_handler(dataset_format)
        self._annotations = handler.load(dataset_path)
    
    def export(self, output_path, dataset_format: str, validation_split: float):
        handler = DataHandlerFactory.create_handler(dataset_format)
        handler.save(self._annotations, output_path, validation_split)
        
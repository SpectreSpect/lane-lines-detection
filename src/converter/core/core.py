from ..handlers.data_handler_factory import DataHandlerFactory
from ..handlers import DataHandler

class Core():
    def __init__(self, dataset_path: str, dataset_format: str):
        self._label_names = []
        self._validation_split = 0.0
        self._dataset_format = dataset_format
        
        handler = DataHandlerFactory.create_handler(dataset_format)
        self.annotations = handler.load(dataset_path)
        
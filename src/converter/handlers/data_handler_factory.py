from .data_handler import DataHandler
from .cvat_image_handler import CvatImageHandler
from .yolo_image_handler import YoloImageHandler
from .traffic_light_detection_dataset_handler import TrafficLightDetectionDatasetHandler

class DataHandlerFactory:
    @staticmethod
    def create_handler(format_name: str) -> DataHandler:
        switch = {
            "cvat-image": CvatImageHandler,
            "yolo": YoloImageHandler,
            "traffic-light-detection-dataset": TrafficLightDetectionDatasetHandler
        }
        
        if format_name not in switch:
            raise Exception("Invalid data handler format.")
        
        handler = switch.get(format_name)
        
        return handler()

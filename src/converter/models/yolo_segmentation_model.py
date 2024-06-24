from .abstract_model import AbstractModel
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from ..data.box import Box
from ..containers.image_container import ImageContainer
from typing import List
import numpy as np
from ultralytics import YOLO
from src.utiles_for_test import *


class YoloSegmentationModel(AbstractModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.__model = YOLO(model_path)

        if torch.cuda.is_available():
            self.__model.to('cuda')

    def annotate(self, annotation_bundles: List[AnnotationBundle]):
        for annotation_bundle in annotation_bundles:
            annotation_bundle.annotations += self.predict([annotation_bundle.image_container])[0]
        return annotation_bundles

    
    def predict(self, image_containers: List[ImageContainer]) -> List[AnnotationBundle]:
        '''
        This method is not debuged and may not work correctly.
        
        
        
        THIS IS IMPORTANT. DON'T FORGET TO DEBUG IT!
        ^-^
        '''
        
        annotation_bundles: List[AnnotationBundle] = []
        
        for image_container in image_containers:
            results = self.__model.predict([image_container.get_image()], verbose=False)
            
            image_shape = image_container.get_image_shape()   
            for result in results:
                annotations: List[Annotation] = []
                
                for idx, result_mask in enumerate(result.masks):
                    points_n = result_mask.xyn[0] # Might be an error
                    points = points_n * image_shape
                    label = self.__model.names[int(result.boxes[idx].cls)]
                    
                    mask = Mask(points, points_n, label, image_container, False)
                    
                    annotations.append(mask)
                
                annotation_bundle = AnnotationBundle(annotations, image_container)
                annotation_bundles.append(annotation_bundle)
        
        return annotation_bundles

        # results = self.__model.predict(, verbose=False)

        # for result in results:
        #     if self._is_segmentation and result.masks is not None:
        #         for mask in result.masks:
        #             annotation = Mask(
        #                 points=np.array(mask.xy), 
        #                 points_n=np.array(mask.xyn), 
        #                 label=self._model.names[int(result.boxes[0].cls)], 
        #                 image_container=image, is_valid=False
        #             )

        #             annotations.append(annotation)
        #     elif result.boxes is not None:
        #         for box in result.boxes:
        #             annotation = Box(
        #                 points=np.array(box.xyxy), 
        #                 points_n=np.array(box.xyxyn), 
        #                 label=self._model.names[int(result.boxes[0].cls)], 
        #                 image_container=image, 
        #                 is_valid=False
        #             )

        #             annotations.append(annotation)

        # return annotations
        return None

    def get_label_names(self) -> List:
        label_names = []

        for _, name in self._model.names.items():
            label_names.append(name)

        return label_names

from .abstract_model import AbstractModel
from ..data.annotation import Annotation
from ..data.annotation_bundle import AnnotationBundle
from ..data.box import Box
from ..containers.image_container import ImageContainer
from typing import List
import numpy as np
from ultralytics import YOLO
from src.utiles_for_test import *
import timeit


class YoloSegmentationModel(AbstractModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.__model = YOLO(model_path)

        if torch.cuda.is_available():
            self.__model.to('cuda')

    def annotate(self, annotation_bundles: List[AnnotationBundle], verbose=True):
        bundles_count = len(annotation_bundles)
        annotated_bundles_count = 0
        if verbose:
            print(f"Annotation started. Total number of annotation bundles: {bundles_count}")
        start = timeit.default_timer()
        for annotation_bundle in annotation_bundles:
            annotation_bundle.annotations += self.predict([annotation_bundle.image_container])[0].annotations
            
            annotated_bundles_count += 1

            end = timeit.default_timer()

            elapsed_time = end - start
            eta = (elapsed_time / annotated_bundles_count) * bundles_count - annotated_bundles_count

            if verbose:
                print(f"{annotated_bundles_count}/{bundles_count}   elapsed time: {elapsed_time:.3f}    eta: {eta:.3f}")
            
    
    def predict(self, image_containers: List[ImageContainer]) -> List[AnnotationBundle]:
        '''
        This method is not debuged and may not work correctly.
        
        
        
        THIS IS IMPORTANT. DON'T FORGET TO DEBUG IT!
        ^-^
        '''
        
        annotation_bundles: List[AnnotationBundle] = []
        
        for image_container in image_containers:
            image = image_container.get_image()[:, :, :3]
            results = self.__model.predict([image], verbose=False)
            
            image_shape = image_container.get_image_shape()   
            for result in results:
                annotations: List[Annotation] = []
                
                if result.masks is not None:
                    for idx, result_mask in enumerate(result.masks):
                        points_n = result_mask.xyn[0] # Might be an error
                        points = points_n * image_shape
                        label = self.__model.names[int(result.boxes[idx].cls)]
                        
                        mask = Mask(points, points_n, label, image_container, False)
                        
                        annotations.append(mask)
                
                annotation_bundle = AnnotationBundle(annotations, image_container)
                annotation_bundles.append(annotation_bundle)
        
        return annotation_bundles

    def get_label_names(self) -> List:
        label_names = []

        for _, name in self.__model.names.items():
            label_names.append(name)

        return label_names

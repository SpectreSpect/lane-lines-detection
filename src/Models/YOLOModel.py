from .AbstractModel import AbstractModel
from ..converter.data.annotation import Annotation
from ..converter.data.annotation_bundle import AnnotationBundle
from ..converter.data.box import Box
from ..converter.containers.image_container import ImageContainer
from typing import List
import numpy as np
from ultralytics import YOLO
from src.utiles_for_test import *


class YOLOModel(AbstractModel):
    def __init__(self, model_file_path: str, is_segmentation=False):
        super().__init__(model_file_path)

        self._model = YOLO(self._model_file_path)
        self._is_segmentation = is_segmentation

        if torch.cuda.is_available():
            self._model.to('cuda')

    def annotate(self, annotation_bundles: List[AnnotationBundle]) -> List[AnnotationBundle]:
        for annotation_bundle in annotation_bundles:
            annotation_bundle._annotations += self.predict(annotation_bundle.image_container)
        return annotation_bundles

    def predict(self, image: ImageContainer) -> List[Annotation]:
        annotations: List[Annotation] = []

        results = self._model.predict(image.get_image(), verbose=False)

        for result in results:
            if self._is_segmentation and result.masks is not None:
                for mask in result.masks:
                    annotation = Mask(
                        points=np.array(mask.xy), 
                        points_n=np.array(mask.xyn), 
                        label=self._model.names[int(result.boxes[0].cls)], 
                        image_container=image, is_valid=False
                    )

                    annotations.append(annotation)
            elif result.boxes is not None:
                for box in result.boxes:
                    annotation = Box(
                        points=np.array(box.xyxy), 
                        points_n=np.array(box.xyxyn), 
                        label=self._model.names[int(result.boxes[0].cls)], 
                        image_container=image, 
                        is_valid=False
                    )

                    annotations.append(annotation)

        return annotations

    def get_label_names(self) -> List:
        label_names = []

        for _, name in self._model.names.items():
            label_names.append(name)

        return label_names

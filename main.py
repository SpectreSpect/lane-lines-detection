from src.LaneLineModel import LaneLineModel
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *
from src.dataset_balancing import *
from src.reinforcement_data import *
from src.from_xml_to_yolo import *
from src.dataset import *
import re
import time
from src.predictions.Predictor import Predictor
from src.IPMTransformator import IPMTransformator

if __name__ == "__main__":
    model = LaneLineModel(r"models\LLD-level-5-v2\model.pt")
    ipm_transformator = IPMTransformator("ipm_config.yaml")

    predictor = Predictor(model, ipm_transformator)
    predictor.start(r"data\videos\level-5-video.mp4", 1280)

    # from_cvat_to_yolo(output_images_folder=r"data\level-5-dataset\Segment 3\images\train", 
    #                   output_labels_folder=r"data\level-5-dataset\Segment 3\labels\train",
    #                   label_names=get_label_names(r"config.yaml"),
    #                   input_video_path=r"data\videos\segments\Segment 3\segment-3.mp4",
    #                   xml_path=r"data\videos\segments\Segment 3\annotations.xml")
    # save_plotted_video(model,
    #                    r"data\videos\level-5-video-6x - Made with Clipchamp.mp4",
    #                    r"data\videos\plotted-level-5-video.mp4",
    #                    get_label_names("config.yaml"))
    #view_prediction_video(model, r"data\videos\road-video-yellow-solid.mp4", get_label_names("config.yaml"), 1280, 1)

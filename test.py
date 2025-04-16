from src.dataset_balancing import preview_prediction_video, save_plotted_video, get_label_names_dict, get_label_names_dict_inversed
from src.LaneLineModel import LaneLineModel
from PIL import Image
import numpy as np

if __name__ == "__main__":
    model_path = "models/sizefull-ep20/model.pt"
    lane_model = LaneLineModel(model_path)

    label_names_dict = get_label_names_dict("config.yaml")
    label_names_dict_inversed = get_label_names_dict_inversed("config.yaml")
    
    # preview_prediction_video(lane_model, "data/video1.mp4", "config.yaml")
    save_plotted_video(lane_model, "data/road-video-russia.mp4", "video4.mp4", label_names_dict_inversed)
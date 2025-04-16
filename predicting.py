from src.LaneLineModel import LaneLineModel
from PIL import Image
import numpy as np


if __name__ == "__main__":
    # sample model: https://www.kaggle.com/models/spectrespect/sizefull-ep20
    model_path = "models/sizefull-ep20/model.pt"

    # sample image: none
    image_path = "data/20240702-105702.png"

    lane_model = LaneLineModel(model_path)
    image = np.asarray(Image.open(image_path))
    # lines = lane_model.predict([image])

    lane_model.visualize_prediction([image])
    
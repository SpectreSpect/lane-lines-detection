from src.LaneLineModel import LaneLineModel
from PIL import Image
import numpy as np


if __name__ == "__main__":
    # sample model: https://www.kaggle.com/models/spectrespect/sizefull-ep20
    model_path = "models/sizefull-ep20/model.pt"

    # sample image: none
    image_path = "data/yolov8-size1000-val02-fmasks/images/train/150897398774839600.jpg"

    lane_model = LaneLineModel(model_path)
    image = np.asarray(Image.open(image_path))
    lines = lane_model.predict([image])
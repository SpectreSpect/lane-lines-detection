from src.LaneLineModel import LaneLineModel


if __name__ == "__main__":
    # sample model: https://www.kaggle.com/models/spectrespect/sizefull-ep20
    model_path = "models/sizefull-ep20/model.pt"

    # sample dataset: https://www.kaggle.com/datasets/spectrespect/yolov8-size1000-val02-fmasks
    dataset_path = "data/yolov8-size1000-val02-fmasks" 

    output_directory_path = "runs"

    lane_model = LaneLineModel(model_path)
    lane_model.train(dataset_path, 2, output_directory=output_directory_path)
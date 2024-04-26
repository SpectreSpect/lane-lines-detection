from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *
from src.dataset_balancing import *
import re


def load_image(path: str):
    image = np.asarray(Image.open(path))
    return image


def load_images(path: str, max_images_count=-1) -> list:
    images = []
    image_names = []
    images_loaded = 0
    for image_name in os.listdir(path):
        if max_images_count >= 0:
            if images_loaded >= max_images_count:
                break
        image_path = os.path.join(path, image_name)
        if os.path.isfile(image_path):
            image = load_image(image_path)
            images.append(image)
            image_names.append(image_name)
            images_loaded += 1
    return images, image_names


if __name__ == "__main__":
    model = LaneLineModel("models/sizefull-ep20/model.pt")   
    # view_prediction_video(model, "data/videos/road-video-russia.mp4", get_label_names("config.yaml"))
    

    ############################################################################################################################################

    # video_segments = read_video_segments("video-segments.txt")
    # cap = cv2.VideoCapture("road-video-russia-PLOTTED.mp4")

    # cap = cv2.VideoCapture("data/videos/road-video-russia.mp4")
    # if not cap.isOpened():
    #     print("Can't open the video.")
    #     cap.release()

    # video_segment_to_train_data(model, cap, video_segments[0], 
    #                             "test/images", 
    #                             "test/labels", 
    #                             label_names=get_label_names("config.yaml"),
    #                             output_video_path="test_video.mp4")
    



    # video_segments_to_train_data(model, 
    #                              "data/videos/road-video-russia.mp4", 
    #                              video_segments, 
    #                              "test/images",
    #                              "test/labels",
    #                              "tmp/test_videos",
    #                              get_label_names("config.yaml"))
    
    # videos_to_train_data(model, 
    #                      "tmp/test_inputs_data/videos", 
    #                      "tmp/test_inputs_data/segments", 
    #                      "tmp/test_storage3",
    #                      get_label_names("config.yaml"))

    generate_plotted_videos(model, 
                            "tmp/test_inputs_data/videos",
                            "tmp/test_ouptut_plotted",
                            get_label_names("config.yaml"),
                            fps=5)

    ###############################################################################################################################################
    # video_segment_to_train_data(model, cap, video_segments[0], "test/images", "test/labels")



    # video_segments = read_video_segments("video-segments.txt")
    # for video_segment in video_segments:
    #     print(f"{video_segment.start_time}s {video_segment.end_time}s {video_segment.substitutions}")


    # model = LaneLineModel("models/sizefull-ep20/model.pt")

    # save_plotted_video(model, "data/videos/road-video-russia.mp4", "road-video-russia-PLOTTED.mp4")

    # (1:20, 1:30, [(5, 2), (6, 7), (9, 11)])

    # "20" = 20 sec
    # "1:20" = 1 minute 20 seconds
    # "3:01:20" = 3 hours 1 minute 20 seconds

    # preview_prediction_video(model, "data/videos/road-video-russia.mp4", "config.yaml")



    # images, predictions = view_prediction_video(model, "data/videos/road-video-russia.mp4", True)

    # print(f"images len: {len(images)}   preds len: {len(predictions)}")
    # print(type(predictions[0][0][0]))

    # print_labels_distribution_stats("data/yolov8-sizefull-val02-fmasks/labels/train", "config.yaml")
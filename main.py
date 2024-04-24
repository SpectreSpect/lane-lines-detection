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
    # model = LaneLineModel("models/sizefull-ep20/model.pt")   
    # # masks = LaneMask.from_file("test/labels/f7fdc14f-af4d-4daa-b358-a2f091c98120.txt", image_path="test/images/f7fdc14f-af4d-4daa-b358-a2f091c98120.jpg")
    # masks = LaneMask.from_file("data/yolov8-size1000-val02-fmasks/labels/train/150897399236705800.txt", 
    #                            image_path="data/yolov8-size1000-val02-fmasks/images/train/150897399236705800.jpg")
    
    # image = cv2.imread("data/yolov8-size1000-val02-fmasks/images/train/150897399236705800.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # mask_batches = model.predict_masks([image])
    # plot_image = LaneMask.generate_plot(masks, image_path="data/yolov8-size1000-val02-fmasks/images/train/150897399236705800.jpg")
    # show_images([plot_image])

# test\images\a51de2f9-8951-42ef-8637-71f07bcef62f.jpg

    # LaneMask.visualize_masks(masks_path="data/yolov8-size1000-val02-fmasks/labels/train/150912151242604600.txt",
    #                          image_path="data/yolov8-size1000-val02-fmasks/images/train/150912151242604600.jpg",
    #                          mask_alpha=0.7,
    #                          draw_lines=False)

    LaneMask.visualize_masks(masks_path="test/labels/e8326e0a-4336-432f-b20b-3bcdae696455.txt",
                             image_path="test/images/e8326e0a-4336-432f-b20b-3bcdae696455.jpg",
                             mask_alpha=0.7,
                             draw_lines=False)
    exit()
    

    # save_plotted_video(model, "data/videos/road-video-russia.mp4", 
    #                    "lets_look_at_what_it_looks_like.mp4", 
    #                    label_names=get_label_names("config.yaml"))

    # video_segments = read_video_segments("video-segments.txt")

    view_prediction_video(model, "data/videos/road-video-russia.mp4", get_label_names("config.yaml"))
    

    ############################################################################################################################################

    video_segments = read_video_segments("video-segments.txt")
    # cap = cv2.VideoCapture("road-video-russia-PLOTTED.mp4")

    cap = cv2.VideoCapture("data/videos/road-video-russia.mp4")
    if not cap.isOpened():
        print("Can't open the video.")
        cap.release()

    video_segment_to_train_data(model, cap, video_segments[0], 
                                "test/images", 
                                "test/labels", 
                                label_names=get_label_names("config.yaml"),
                                output_video_path="test_video.mp4")

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
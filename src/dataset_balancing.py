import os
import pandas as pd
import yaml
from src.utils import *
from src.LaneLineModel import *
import uuid
import re
# show how we should balance current data.

# show a video with predictions
# specify what labels should be replaced
# replace labels
# save labels files


def get_file_names(path: str):
    file_names = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            file_names.append(name)
    return file_names


def get_labels_df(path: str):
    file_names = get_file_names(path)
    labels = []
    for filename in file_names:
        full_path = os.path.join(path, filename)
        with open(full_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                label = line.split(' ', 1)[0]
                labels.append(label)
    df = pd.DataFrame({'labels': labels})
    return df


def get_label_names(config_path: str) -> list:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        label_names = config["names"]
    return label_names


def get_label_names_dict(config_path: str) -> list:
    label_names = get_label_names(config_path)
    names_dict = {str(i): name for i, name in enumerate(label_names)}
    return names_dict


def get_label_names_dict_inversed(config_path: str) -> list:
    label_names_dict = get_label_names_dict(config_path)
    inv_map = {v: k for k, v in label_names_dict.items()}
    return inv_map


def print_labels_distribution_stats(labels_path: str, config_path: str):
    labels_df = get_labels_df(labels_path)
    label_names = get_label_names(config_path)

    names_dict = {str(i): name for i, name in enumerate(label_names)}
    
    value_counts = labels_df['labels'].value_counts()
    value_counts.index = value_counts.index.map(names_dict)

    value_counts_df = pd.DataFrame({'abs': value_counts})

    value_counts_df['add-to-balance'] = value_counts_df['abs'].max() - value_counts_df['abs']

    print(value_counts_df)


def print_dict(dictionary: dict, keys_to_note: list = [], palette=[]):
    idx = 0
    for k, v in dictionary.items():
        found = True if k in keys_to_note else False
        if len(palette) > 0: 
            print(paint_str(f'{k}: {v}', palette[idx]))
        else:
            if found:
                print("\033[32m", end="")
            print(f"{k}: {v}", end="") 
            if found: 
                print("\033[0m", end="")
            print("\033[0m")
        idx += 1
    # \033[32mGreen text\033[0m


def save_predictions(images_path: str, preds_path: str, images, predictions, label_names_dict_inversed, label_names_dict, tolerance=0.0015):
    for idx, (image, prediction) in enumerate(zip(images, predictions)):
        random_filename = str(uuid.uuid4())

        image_path = os.path.join(images_path, random_filename + ".jpg")
        prediction_path = os.path.join(preds_path, random_filename + ".txt")

        cv2.imwrite(image_path, image)

        file_string = ""
        if prediction.masks is not None:
            for idy, (xyn, cls) in enumerate(zip(prediction.masks.xyn, prediction.boxes.cls)):
                simplified_polygon = Polygon(xyn).simplify(tolerance, preserve_topology=True)
                points = np.array(simplified_polygon.exterior.coords)

                label = int(label_names_dict_inversed[label_names_dict[str(int(cls))]])
                file_string += str(int(label)) + " "
                for idx, point in enumerate(points):
                    file_string += str(float(point[0])) + " " + str(float(point[1]))
                    if idy < len(prediction.masks.xyn) - 1:
                        if idx < len(points) - 1:
                            file_string += " "
                        else:
                            file_string += "\n"
        
        with open(prediction_path, "w") as file:
            file.write(file_string)
            
  

def preview_prediction_video(model: LaneLineModel, video_path: str, config_path: str):

    label_names_dict = get_label_names_dict(config_path)
    label_names_dict_inversed = get_label_names_dict_inversed(config_path)
    # print(label_names_dict)

    print_dict(label_names_dict_inversed, palette=default_palette)
    images, predictions = view_prediction_video(model, video_path, True, verbose=0)
    
    last_label_name = ""
    while True:
        print("-" * 20)
        if last_label_name != "":
            print_dict(label_names_dict_inversed, [last_label_name])
        print("-" * 20)

        input_text = input()

        if input_text == "q":
            break

        input_text_split = input_text.split(' ', 1)
        label_name = input_text_split[0][:-1]
        replacement = int(input_text_split[1])

        label_names_dict_inversed[label_name] = replacement

        last_label_name = label_name


    # for prediction in predictions:
    #     for idx, cls in enumerate(prediction.boxes.cls):
    #         prediction.boxes.cls[idx] = int(label_names_dict_inversed[label_names_dict[str(int(cls))]])
    
    save_predictions("test/images", "test/labels", images, predictions, label_names_dict_inversed, label_names_dict)
    

def save_plotted_video(model: LaneLineModel, src_video_path: str, output_path: str, label_names: list = [], fps_=-1, verbose=True):
    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        print("Can't open the video.")
        cap.release()
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps_ <= 0:
        fps_ = fps

    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_, (width, height))

    fps_n = fps // fps_

    frame = 0
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        if frame % fps_n == 0:
            # predictions = model.model.predict([image], verbose=False)
            mask_batches = model.predict_masks([image])
            batch_lines = model.get_lines(mask_batches)
            

            # mask_batches = LaneMask.from_predictions(predictions, tolerance=0)

            draw_segmentation([image], mask_batches)
            draw_lines([image], batch_lines)
            if (len(label_names) > 0):
                draw_labels([image], mask_batches, label_names)
            out.write(image)
            
        frame += 1
        if verbose:
            print(f"frame {frame}/{frames_count}")
    out.release()


class VideoSegment:
    def __init__(self, start_time, end_time, substitutions: dict):
        self.start_time = start_time
        self.end_time = end_time
        self.substitutions = substitutions


# def read_video_segments(path: str) -> list:
#     video_segments = []
#     with open(path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             data = line.split(", ")
#             start_time = data[0]
#             end_time = data[1]
#             substitutions = []
            
#             for str_value in data[2:]:
#                 numbers = re.findall(r'\d+', str_value)
#                 substitutions.append([numbers[0], numbers[1]])
            
#             video_segment = VideoSegment(start_time, end_time, substitutions)
#             video_segments.append(video_segment)
#     return video_segments


def read_video_segments(path: str) -> list:
    video_segments = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.split(", ", 2)
            start_time = str_to_seconds(data[0])
            end_time = str_to_seconds(data[1])
            substitutions = dict()

            numbers = re.findall(r'\d+', data[2])
            for i in range(0, len(numbers), 2):
                substitutions[numbers[i]] = numbers[i + 1]

            video_segment = VideoSegment(start_time, end_time, substitutions)
            video_segments.append(video_segment)
    return video_segments


# def save_prediction(image, prediction, video_segment):


def save_prediction(prediction, video_segment, output_path, tolerance=0.0015):
    file_string = ""
    if prediction.masks is not None:
        for idy, (xyn, cls) in enumerate(zip(prediction.masks.xyn, prediction.boxes.cls)):
            
            simplified_polygon = Polygon(xyn).simplify(tolerance, preserve_topology=True)
            points = np.array(simplified_polygon.exterior.coords)

            if str(int(cls)) in video_segment.substitutions:
                label = int(video_segment.substitutions[str(int(cls))])
            else:
                label = int(cls)
            
            file_string += str(label) + " "
            for idx, point in enumerate(points):
                file_string += str(float(point[0])) + " " + str(float(point[1]))
                if idy < len(prediction.masks.xyn) - 1:
                    if idx < len(points) - 1:
                        file_string += " "
                    else:
                        file_string += "\n"
    
    with open(output_path, "w") as file:
            file.write(file_string)


def save_prediction_2(lane_masks: list, output_path: str, tolerance: float = 0.0015):
    file_string = ""
    if lane_masks is not None:
        for idy, lane_mask in enumerate(lane_masks):
            
            if lane_mask.points_n.shape[0] > 4:
                simplified_polygon = Polygon(lane_mask.points_n).simplify(tolerance, preserve_topology=True)
                points = np.array(simplified_polygon.exterior.coords)
            else:
                points = lane_mask.points_n
            
            file_string += str(lane_mask.label) + " "
            for idx, point in enumerate(points):
                file_string += str(float(point[0])) + " " + str(float(point[1]))
                if idy < len(points) - 1:
                    if idx < len(points) - 1:
                        file_string += " "
                    else:
                        file_string += "\n"
    
    with open(output_path, "w") as file:
            file.write(file_string)


def video_segment_to_train_data(model: LaneLineModel, cap, video_segment: VideoSegment, 
                                ouput_images_path: str, output_labels_path: str, output_video_path: str = None, 
                                label_names: list = [], fps_=-1, verbose=3):
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    if fps_ <= 0:
        fps_ = frame_rate
    
    fps_n = frame_rate // fps_

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = video_segment.start_time * frame_rate
    end_frame = video_segment.end_time * frame_rate

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if output_video_path != None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps_, (width, height))

    for idx in range(end_frame - start_frame):
        ret, image = cap.read()
        if not ret:
            break

        if idx % fps_n == 0:
            random_filename = str(uuid.uuid4())

            image_path = os.path.join(ouput_images_path, random_filename + ".jpg")
            labels_path = os.path.join(output_labels_path, random_filename + ".txt")

            cv2.imwrite(image_path, image)

            # prediction = model.model.predict([image], verbose=False)[0]


            # mask_batches = LaneMask.from_predictions(prediction, tolerance=0)
            # LaneMask.substitute_labels(mask_batches[0], video_segment.substitutions)

            # save_prediction(prediction, video_segment, labels_path, 0.0015)

            mask_batches = model.predict_masks([image])

            for masks in mask_batches:
                LaneMask.substitute_labels(masks, video_segment.substitutions)
            save_prediction_2(mask_batches[0], labels_path, 0.0015)

            if output_video_path != None:
                batch_lines = model.get_lines(mask_batches)
                draw_segmentation([image], mask_batches)
                draw_lines([image], batch_lines)
                if len(label_names) > 0:
                    draw_labels([image], mask_batches, label_names)
                out.write(image)
            if verbose == 3:
                print(f"        frame {idx}/{end_frame - start_frame}")
    if output_video_path != None:
        out.release()


def video_segments_to_train_data(model: LaneLineModel, video_path: str, video_segments: VideoSegment, 
                                ouput_images_path: str, output_labels_path: str, output_videos_path: str = None, 
                                label_names: list = [], fps=-1, verbose=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Can't open the video.")
        cap.release()

    input_video_name = os.path.basename(video_path).split(".")[0]

    for idx, video_segment in enumerate(video_segments):
        if verbose == 3:
            print(f"    Segments loaded: {idx+1}")
        output_video_name = input_video_name + "_" + str(idx)
        output_video_path = os.path.join(output_videos_path, output_video_name + ".mp4")
        video_segment_to_train_data(model, cap, video_segment, ouput_images_path, 
                                    output_labels_path, output_video_path, 
                                    label_names, fps_=fps, verbose=verbose)


def videos_to_train_data(model: LaneLineModel, videos_path: str, video_segments_path: str, output_path: str,
                         label_names: list = [], fps=-1):
    video_paths = []
    for filename in os.listdir(videos_path):
        file_path = os.path.join(videos_path, filename)
        if os.path.isfile(file_path):
            video_paths.append(file_path)
    
    video_segment_paths = []
    for filename in os.listdir(video_segments_path):
        file_path = os.path.join(video_segments_path, filename)
        if os.path.isfile(file_path):
            video_segment_paths.append(file_path)
    
    for idx, (video_path, video_segment_path) in enumerate(zip(video_paths, video_segment_paths)):
        video_basename = os.path.basename(video_path)
        video_name = os.path.splitext(video_basename)[0]
        
        segment_output_path = os.path.join(output_path, video_name)

        output_images_path = os.path.join(segment_output_path, "images")
        output_labels_path = os.path.join(segment_output_path, "labels")
        output_videos_path = os.path.join(segment_output_path, "videos")

        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_labels_path, exist_ok=True)
        os.makedirs(output_videos_path, exist_ok=True)

        print(f"{idx}: {video_name}")

        video_segments = read_video_segments(video_segment_path)
        video_segments_to_train_data(model, 
                                    video_path, 
                                    video_segments, 
                                    output_images_path,
                                    output_labels_path,
                                    output_videos_path,
                                    label_names, fps=fps, verbose=3)


def generate_plotted_videos(model: LaneLineModel, src_videos_path: str, output_path: str, label_names: list = [], fps=-1, verbose=True):
    for idx, filename in enumerate(os.listdir(src_videos_path)):
        filepath = os.path.join(src_videos_path, filename)
        if os.path.isfile(filepath):
            output_path = os.path.join(output_path, filename)
            save_plotted_video(model, filepath, output_path, label_names, fps_=fps)
            if verbose:
                print(f"{idx}: {filename}")


# def video_to_train_data(model: LaneLineModel, video_path: str, video_segments_path: str, ouput_images_path: str, output_labels_path: str, tolerance=0.0015):
#     video_segments = read_video_segments(video_segments_path)

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Can't open the video.")
#         cap.release()
#         return
    

#     video_segment = ?
    
#     start_frame = 100
#     end_frame = 200

#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#     for _ in range(end_frame - start_frame):
#         ret, image = cap.read()
#         if not ret:
#             break

#         random_filename = str(uuid.uuid4())

#         image_path = os.path.join(ouput_images_path, random_filename + ".jpg")
#         labels_path = os.path.join(output_labels_path, random_filename + ".txt")

#         cv2.imwrite(image_path, image)

#         prediction = model.model.predict([image], verbose=False)[0]
#         save_prediction(prediction, video_segment, labels_path, 0.0015)





        
        
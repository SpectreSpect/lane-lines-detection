import os
import pandas as pd
import yaml
from src.utils import *
from src.LaneLineModel import *
import uuid
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
    

    
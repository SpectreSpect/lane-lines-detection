from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from src.utils import *
from src.dataset_balancing import *
from src.reinforcement_data import *
from src.from_xml_to_yolo import *
import re
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader



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
    model = YOLO("models/robolife-detection-yolov8l-seg/model.pt")
    model.to("cuda")
    batch_size = 100
    generator = image_batch_generator("data/yolov8-new-data1-val015-fmasks/images/train", batch_size)
    data_generator = DataGenerator(model)

    done_images_count = 0
    images_count = len(os.listdir("data/yolov8-new-data1-val015-fmasks/images/train"))

    for images, names in generator:
        results = model.predict(images, verbose=False)
        data_generator.generate_from_results(results, names, "data/new-data-lables/train")

        done_images_count += len(images)
        print(f"{done_images_count}/{images_count}")


    # data_transforms = transforms.Compose([
    # transforms.ToTensor()  # Convert images to PyTorch tensors
    # ])
    # image_dataset = datasets.ImageFolder(root="data/yolov8-new-data1-val015-fmasks/images/train", 
    #                                      transform=data_transforms)
    
    # data_loader = DataLoader(image_dataset, batch_size=256, shuffle=True, num_workers=4)

    # for inputs, labels in data_loader:
    #     print(inputs.size(), labels.size())
    #     break


    # bounding_box_batches = get_boxes_from_image_cvat("data/annotations.xml")
    # BoundingBox.batches_to_yolo(bounding_box_batches, "tmp/output_folder")



    # model = YOLO("models/robolife-detection-yolov8l-seg/model.pt")
    # model.to("cuda")
    # data_generator = DataGenerator(model)
    # data_generator.generate("data/yolov8-new-data1-val015-fmasks/images/train", "data/new-data-lables/train")
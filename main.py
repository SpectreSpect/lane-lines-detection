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


if __name__ == "__main__":
   #  test_dataset = YoloImageDataset.create_dataset("tmp/test-dataset", 
   #                                                 "tmp/test-images", "tmp/test-labels", 
   #                                              #    label_name_list=["double-dash", "some-label", "hello"],
   #                                                 config_path="config.yaml",
   #                                                 validation_split=0.2)

   # images_folder_dataset = ImagesFolderDataset("tmp/test-images", batch_size=2)
   # print(len(images_folder_dataset))
   # for images, image_names in images_folder_dataset.flow():
   #    print(image_names)


   labels_folder_dataset = YoloSegLabelsFolderDataset("tmp/test-labels", batch_size=2)
   print(len(labels_folder_dataset))
   for labels in labels_folder_dataset.flow():
      print(labels[0][0].image_name)
   
   masks = LaneMask.from_file("tmp/test-labels/A.txt")
   print("sdf")


    # model = LaneLineModel("models/LLD-2.pt")



   # image_dataset = ImageDataset(path_to_images, path_to_labels)
   # dataset_size = len(image_dataset)
   # images, labels = image_dataset.next(128)
   # image_dataset.save(save_path, format=format)


   # image_dataset = ImageDataset(images_path)
   # label_dataset = LabelDataset(labels_path)

   # yolo_dataset = YoloDataset(image_dataset, label_dataset)

   # dataset_size = len(image_dataset)
   # images, labels = image_dataset.next(128)
   # image_dataset.save(save_path, format=format)


   # 


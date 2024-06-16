import os
import yaml
import shutil
from src.utils import get_shared_names
import numpy as np
from PIL import Image
from src.utils import *


class Dataset():
    def __init__(self):
        pass

    def get_size(self) -> int:
        pass

    def get(self, idx: int) -> int:
        pass

    def next(self, batch_size: int):
        pass

    def flow(self):
        pass


class ImageDataset(Dataset):
    def __init__(self):
        pass


class ImagesFolderDataset(Dataset):
    def __init__(self, folder_path: str, batch_size: int = 128):
        self.folder_path = folder_path
        self.image_names = os.listdir(self.folder_path)
        self.size = len(self.image_names)
        self.batch_size = batch_size
    
    def __len__(self):
        return self.size
    
    def flow(self):    
        start_index = 0
        while start_index < len(self.image_names):
            end_index = min(start_index + self.batch_size - 1, len(self.image_names) - 1)
            
            batch = []
            for i in range(start_index, end_index + 1):
                image_path = os.path.join(self.folder_path, self.image_names[i])
                image = np.asarray(Image.open(image_path))
                batch.append(image)
            
            yield [batch, self.image_names[start_index:(end_index + 1)]]

            start_index = end_index + 1


class LabelDataset(Dataset):
    def __init__(self):
        pass


class YoloSegLabelsFolderDataset(LabelDataset):
    def __init__(self, folder_path: str, batch_size: int = 128):
        self.folder_path = folder_path
        self.label_names = os.listdir(self.folder_path)
        self.size = len(self.label_names)
        self.batch_size = batch_size
    
    def __len__(self):
        return self.size
    
    def flow(self):    
        start_index = 0
        while start_index < len(self.label_names):
            end_index = min(start_index + self.batch_size - 1, len(self.label_names) - 1)
            
            batch = []
            for i in range(start_index, end_index + 1):
                label_path = os.path.join(self.folder_path, self.label_names[i])
                masks = LaneMask.from_file(label_path)
                batch.append(masks)
            
            yield batch

            start_index = end_index + 1


class YoloImageDataset(Dataset):
    def __init__(self, dataset_path: str,
                 save_dataset: bool = False,
                 input_images_path: str = None, input_labels_path: str = None,
                 config_path: str = None, label_name_list: list = None):
        self.dataset_path = dataset_path
        if config_path is None:
            config_path = os.path.join(self.dataset_path, "config.yaml")
        
        # if create_dataset:
        #     YoloImageDataset.create_dataset(dataset_path, input_labels_path, input_images_path, config_path)


    @staticmethod
    def create_dataset(output_datset_path: str, 
                       input_images_path: str, input_labels_path: str, 
                       config_path: str = None, label_name_list: list = None, 
                       validation_split: float = 0.0):

        if config_path is None and label_name_list is None:
            raise ValueError('You should specify either "config_path" or "label_name_list"')
            
        output_labels_train_path = os.path.join(output_datset_path, "labels", "train")
        output_labels_valid_path = os.path.join(output_datset_path, "labels", "valid")
        output_images_train_path = os.path.join(output_datset_path, "images", "train")
        output_images_valid_path = os.path.join(output_datset_path, "images", "valid")

        if not os.path.exists(output_datset_path):
            os.makedirs(output_datset_path)
        
        if not os.path.exists(output_labels_train_path):
            os.makedirs(output_labels_train_path)
        
        if not os.path.exists(output_labels_valid_path):
            os.makedirs(output_labels_valid_path)
        
        if not os.path.exists(output_images_train_path):
            os.makedirs(output_images_train_path)
        
        if not os.path.exists(output_images_valid_path):
            os.makedirs(output_images_valid_path)

        output_dataset = None
        if config_path is None:
            config_path = os.path.join(output_datset_path, "config.yaml")
            YoloImageDataset.create_config(config_path, label_name_list, 
                                           output_datset_path, output_images_train_path, 
                                           output_labels_train_path)
        else:
            output_config_path = os.path.normpath(os.path.join(output_datset_path, os.path.basename(config_path)))
            shutil.copy(config_path, output_config_path)
            
        
        output_dataset = YoloImageDataset(output_datset_path, config_path)

        shared_names = get_shared_names(input_images_path, input_labels_path)

        idx = 0
        train_count = int((1.0 - validation_split) * len(shared_names[0]))
        for image_name, label_name in zip(shared_names[0], shared_names[1]):
            if idx < train_count:
                output_image_path = output_images_train_path
                output_label_path = output_labels_train_path
            else:
                output_image_path = output_images_valid_path
                output_label_path = output_labels_valid_path
            
            image_path = os.path.join(input_images_path, image_name)
            label_path = os.path.join(input_labels_path, label_name)
            
            shutil.copy(image_path, output_image_path)
            shutil.copy(label_path, output_label_path)

            idx += 1

        return output_dataset


    @staticmethod
    def create_config(output_config_path: str, label_name_list: list,
                      output_datset_path: str = "", 
                      output_images_train: str = "", output_labels_train: str = ""):
        with open(output_config_path, 'w') as file:
                yaml_dict = {
                    "path": os.path.normpath(output_datset_path),
                    "train": os.path.normpath(output_images_train),
                    "val": os.path.normpath(output_labels_train),
                    "nc": len(label_name_list),
                    "names": label_name_list
                }
                yaml.dump(yaml_dict, file)


class LabelDataset(Dataset):
    def __init__(self):
        pass

    def get_size(self) -> int:
        pass

    def get(self, idx: int) -> int:
        pass

    def next(self, batch_size: int):
        pass


class CvatImageLabelDataset(LabelDataset):
    def __init__(self, file_path):

        pass

    def get_size(self) -> int:
        pass

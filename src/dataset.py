import os
import yaml
import shutil
from src.utils import get_shared_names


class Dataset():
    def __init__(self):
        pass


class ImageDataset(Dataset):
    def __init__(self):
        pass


class YoloImageDataset(ImageDataset):
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


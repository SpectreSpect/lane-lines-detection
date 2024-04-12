import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml
import numpy as np
import cv2
from src.utils import *


class LaneLineModel:
    def __init__(self, path: str):
        self.model = YOLO(path)
    
    def get_lines(self, results):
        #batch_lines = get_straight_lines(results)
        batch_lines = get_lines_contours(results)
        return batch_lines

    def train(self, dataset_path, epochs, train_path="images/train", val_path="images/val"):

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

            config['path'] = dataset_path
            config['train'] = train_path
            config['val'] = val_path

        with open("tmp/tmp_config.yaml", 'w') as file:
            yaml.dump(config, file)
        
        results = self.model.train(data="tmp/tmp_config.yaml", epochs=epochs)
        return results

    def predict(self, images):
        results = self.model.predict(images)
        lines = self.get_lines(results)
        return lines
    
    def generate_prediction_plots(self, images):
        results = self.model.predict(images)
        plot_images = [result.plot() for result in results]
        return plot_images
    
    def visualize_prediction(self, image):
        result = self.model.predict([image])[0]
        plot_image = result.plot()
        plt.imshow(plot_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return plot_image
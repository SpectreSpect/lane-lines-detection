import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml
import numpy as np
import cv2
from src.utils import *
import os


class LaneLineModel:
    def __init__(self, path: str, use_curve_line=True):
        self.model = YOLO(path)
        self.use_curve_line = use_curve_line
    
    def get_lines(self, results):
        if self.use_curve_line:
            batch_lines = get_lines_contours(results)
        else:
            batch_lines = get_straight_lines(results)
        return batch_lines

    def train(self, dataset_path, epochs, output_directory="runs", train_path="images/train", val_path="images/valid"):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

            absolute_path = os.path.abspath(dataset_path)

            config['path'] = absolute_path
            config['train'] = train_path
            config['val'] = val_path

        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        
        with open("tmp/tmp_config.yaml", 'w') as file:
            yaml.dump(config, file)
        
        results = self.model.train(data="tmp/tmp_config.yaml", epochs=epochs, project=output_directory)
        return results

    def predict(self, images):
        results = self.model.predict(images)
        lines = self.get_lines(results)
        return lines
    
    def generate_prediction_plots_yolo(self, images):
        results = self.model.predict(images)
        plot_images = [result.plot() for result in results]
        return plot_images
    
    def generate_prediction_plots(self, images):
        results = self.model.predict(images)
        batch_lines = self.get_lines(results)

        images_to_draw = np.copy(images)
        draw_segmentation(images_to_draw, results)
        draw_lines(images_to_draw, batch_lines)

        return images_to_draw
    
    def visualize_prediction(self, image):
        result = self.model.predict([image])[0]
        plot_image = result.plot()
        plt.imshow(plot_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return plot_image
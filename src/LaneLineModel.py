import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml
import numpy as np
import cv2



class LaneLineModel:
    def __init__(self, path: str):
        self.model = YOLO(path)
    
    def get_lines(self, results):
        batch_lines = []
        for result in results:
            masks = result.masks
            if masks is None:
                return []

            mask_image = np.zeros(masks.orig_shape + (1,), dtype=np.uint8)
            
            mask_lines = []
            for xy in masks.xy:
                cv2.drawContours(mask_image, [np.expand_dims(xy, 1).astype(np.int32)], contourIdx=-1, color=(255), thickness=-1)
                lines = cv2.HoughLinesP(mask_image, 1, np.pi / 180, threshold=300, minLineLength=25, maxLineGap=30)
            
                if lines is not None:
                    best_line = None
                    max_lenght = 0

                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        lenght = np.linalg.norm([x2-x1, y2-y1])
                        if best_line is None or lenght > max_lenght:
                            max_lenght = lenght
                            best_line = line
                    mask_lines.append(best_line)
                
                mask_image[:] = 0
            batch_lines.append(mask_lines)

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
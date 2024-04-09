import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml



class LaneLineModel:
    def __init__(self, path: str):
        self.model = YOLO(path)
    
    def get_lines(self, results):
#         for result in results:
#             result.masks
#         start = [0, 0]
#         end = [1, 1]
#         label = 0
#         line = [[start, end, label]]
        lines = []
        return lines

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
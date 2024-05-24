from ultralytics import YOLO
import os
from PIL import Image
import numpy as np


class DataGenerator():
    def __init__(self, model: YOLO):
        self.model = model
    

    def load_images(self, path: str, max_images_count=-1):
        images = []
        image_names = []
        images_loaded = 0
        for image_name in os.listdir(path):
            if max_images_count >= 0:
                if images_loaded >= max_images_count:
                    break
            image_path = os.path.join(path, image_name)
            if os.path.isfile(image_path):
                image = np.asarray(Image.open(image_path))
                images.append(image)
                image_names.append(os.path.splitext(image_name)[0]) # A name without an extention
                images_loaded += 1
        return {"images": images,
                "names": image_names}


    def generate_from_result(self, result, output_path: str):
        output_string = ""
        for box in result.boxes:
            label = int(box.cls)
            x = float(box.xywhn[0][0])
            y = float(box.xywhn[0][1])
            width = float(box.xywhn[0][2])
            height = float(box.xywhn[0][3])

            output_string += f"{label} {x} {y} {width} {height}\n"
        
        with open(output_path, 'w') as file:
            file.write(output_string)
    

    def generate_from_results(self, results: list, names: list, output_folder_path: str):
        for idx, result in enumerate(results):
            name = names[idx] + ".txt"
            output_path = os.path.join(output_folder_path, name)
            self.generate_from_result(result, output_path)


    def generate(self, input_images_path: str, output_labels_path: str):
        image_dict = self.load_images(input_images_path, max_images_count=2) # DON'T FORGET TO REMOVE THIS 2
        images = image_dict["images"]
        image_names = image_dict["names"]

        results = self.model.predict(images, verbose=False)
        self.generate_from_results(results, image_names, output_labels_path)
        # self.generate_from_result(results[0], os.path.join(output_labels_path, "test.txt"))


    # def save_yolo_labels(model: YOLO, images, image_names):
    #     results = model.predict(images)
        
    #     for result in results:
    #         result.boxes
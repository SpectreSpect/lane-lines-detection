from src.LaneLineModel import LaneLineModel
from src.IPMTransformator import IPMTransformator
from src.predictions.Accelerometer import Accelerometer
import cv2
import numpy as np
from src.utils import *
from src.Map import Map
import threading
import time
from src.predictions.LinesAnalizator import LinesAnalizator

class Predictor():

    def __init__(self, model: LaneLineModel, ipm_transformator: IPMTransformator):
        self.model = model
        self.ipm_transformator = ipm_transformator
        self.is_start = False
        self.map = None
        self.drawing_images = {}
        self.drawing_images_lock = threading.Lock()
        self.predicator_resource = []
        
        self.mouse_inf_queue_dict = {}
        self.mouse_inf_queue_lenght = 10
        self.mouse_inf_lock = threading.Lock()

        self.key_queue_locker = threading.Lock()
        self.key_queue = []
        self.key_queue_lenght = 10

        self.predicted_lines_queue = []
        self.predicted_lines_queue_lenght = 30
        self.predicted_lines_lock = threading.Lock()
        self.lines_analizator = LinesAnalizator(self)

        self.accelerometer = Accelerometer(self)
        self.car_speed = 0
        self.car_speed_lock = threading.Lock()

        self.__src = 0
        self.__resized_width = 1280
        self.__skip_seconds = 10
        
    def mouse_callback(self, event, x, y, flags, param):
        predictor, window_name = param
        if window_name not in predictor.mouse_inf_queue_dict:
            with self.mouse_inf_lock:
                predictor.mouse_inf_queue_dict[window_name] = []
        
        with self.mouse_inf_lock:
            predictor.mouse_inf_queue_dict[window_name].append(dict(event=event, x=x, y=y, flags=flags))
            if len(predictor.mouse_inf_queue_dict[window_name]) > predictor.mouse_inf_queue_lenght:
                predictor.mouse_inf_queue_dict[window_name].pop(0)

    def find_key(self, key_code):
        for idx, saved_key_code in enumerate(self.key_queue):
            if key_code == saved_key_code:
                self.key_queue.pop(idx)
                return True
        return False


    def predict_images(self):
        cap = cv2.VideoCapture(self.__src)
        if not cap.isOpened():
            print("Не удалось открыть файл.")
            cap.release()
            return
        
        self.predicator_resource = [None, threading.Lock()]

        with self.drawing_images_lock:
            self.drawing_images["prediction video"] = self.predicator_resource

        self.map = Map(self)
        # map_thread = threading.Thread(target=self.map.start)
        # map_thread.start()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        skip_frames = int(fps * self.__skip_seconds)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(r"data\videos\level-5-cout-roads-test-plotted.mp4", fourcc, fps, (width, height))
        
        self.accelerometer.to_reset_acceleration()
        frame = 0
        while cap.isOpened() and self.is_start:
            ret, image = cap.read()
            if not ret:
                break
            
            with self.car_speed_lock:
                self.car_speed = self.accelerometer.get_acceleration(image)

            # Обработка изображения
            batch_lines, mask_batches, predictions = self.model.predict([image])
            label_names = list(map(lambda x: x[1], predictions[0].names.items()))

            count_roads, car_line_id = self.lines_analizator.analize_roads_with_accamulator(image, mask_batches, 300, 600, image.shape[1], 20)
            print(f"count_roads: {count_roads}  car_line_id: {car_line_id}")

            with self.predicted_lines_lock:
                if len(self.predicted_lines_queue) == 0:
                    self.predicted_lines_queue.append(batch_lines[0])
                if len(self.predicted_lines_queue) > self.predicted_lines_queue_lenght:
                    self.predicted_lines_queue.pop(0)

            draw_segmentation([image], mask_batches)
            draw_lines([image], batch_lines)
            draw_labels([image], mask_batches, label_names)
            self.lines_analizator.draw_labels(image, 
                                              label_names=[f"Count of lanes: {count_roads}", f"Car on lane: {car_line_id}"],
                                              colors=[(27, 198, 250), (25, 247, 184)])

            out.write(image)

            if self.__resized_width > 0:
                resized_height = int(image.shape[0] / image.shape[1] * self.__resized_width)
                image = cv2.resize(image, (self.__resized_width, resized_height))

            self.ipm_transformator.draw_points(image)

            with self.predicator_resource[1]:
                self.predicator_resource[0] = image

            i += 1
            frame += 1

            with self.key_queue_locker:
                with self.mouse_inf_lock:
                    if "prediction video" in self.mouse_inf_queue_dict:
                        self.ipm_transformator.listen_key_queue(self.key_queue, self.mouse_inf_queue_dict["prediction video"])

                if self.find_key(ord('q')):
                    self.is_start = False
                    break

                if self.find_key(ord('b')):
                    frame = max(0, frame - skip_frames)
                    i = max(0, i - skip_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

                if self.find_key(ord('n')):
                    frame = min(frame + skip_frames, total_frames - 1)
                    i = min(i + skip_frames, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        
        out.release()
        cap.release()
        self.is_start = False
        self.map.is_start = False
        #map_thread.join()
        self.map = None

        with self.drawing_images_lock:
            del self.drawing_images["prediction video"]



    def start(self, src, resized_width, skip_seconds=5):
        self.is_start = True

        self.__src = src
        self.__resized_width = resized_width
        self.__skip_seconds = skip_seconds
        predicator_thread = threading.Thread(target=self.predict_images)
        predicator_thread.start()

        while self.is_start:
            with self.key_queue_locker:
                key_code = cv2.waitKey(5) & 0xFF
                if key_code != 255:
                    self.key_queue.append(key_code)
                if len(self.key_queue) > self.key_queue_lenght:
                    self.key_queue.pop(0)

            with self.drawing_images_lock:
                for name, (image, lock) in self.drawing_images.items():
                    if image is not None:
                        with lock:
                            cv2.imshow(name, image) 
                            if name not in self.mouse_inf_queue_dict:
                                cv2.setMouseCallback(name, self.mouse_callback, [self, name])
        
        cv2.destroyAllWindows()
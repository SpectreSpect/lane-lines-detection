import numpy as np
import cv2
import threading
import time
from src.utils import default_palette

class Map():
    def __init__(self, predicator):
        self.predicator = predicator
        self.is_start = False
        
        self.cam_position = np.array([0, 0], dtype=np.float32)
        self.car_position = np.array([0, 0], dtype=np.float32)
        
        self.image_size = (600, 600)

        self.grid_gap = 50
        self.background_color = (0, 0, 0)
        self.grid_color = (33, 33, 33)
        self.grid_thickness = 2
        
        self.line_thickness = 2
        self.point_radius = 2

        self.pov_point = np.array([50, 20])

        self.map_resource = []

        self.lines = []


    def draw_grid(self, image):
        count_grid = (np.array(self.image_size) / self.grid_gap + 1).astype(int)
        
        # Вертикальные линии
        left_pos = self.cam_position[0] - self.image_size[0] / 2
        first_vert_line_x = left_pos - left_pos % self.grid_gap + self.grid_gap
        for idx in range(count_grid[0]):
            line_x = first_vert_line_x + idx * self.grid_gap
            cam_line_x = line_x - self.cam_position[0] + self.image_size[0] / 2
            if cam_line_x < self.image_size[0]:
                cv2.line(image, (int(cam_line_x), 0), (int(cam_line_x), int(self.image_size[1])), self.grid_color, thickness=self.grid_thickness)
        
        # Горизонтальные линии
        top_pos = self.cam_position[1] - self.image_size[1] / 2
        first_horiz_line_y = top_pos - top_pos % self.grid_gap + self.grid_gap
        for idy in range(count_grid[1]):
            line_y = first_horiz_line_y + idy * self.grid_gap
            cam_line_y = line_y - self.cam_position[1] + self.image_size[1] / 2
            if cam_line_y < self.image_size[1]:
                cv2.line(image, (0, int(cam_line_y)), (int(self.image_size[1]), int(cam_line_y)), self.grid_color, thickness=self.grid_thickness)
        
        return image


    def draw_lines(self, image):
        for line in self.lines:
            if len(line[0]) > 1:
                color = (default_palette[line[1] % len(default_palette)])[::-1]
                cv2.circle(image, line[0][0].astype(int), self.point_radius, color, -1)
                for idx in range(len(line[0]) - 1):
                    cv2.line(image, line[0][idx].astype(int), line[0][idx + 1].astype(int), color, thickness=self.line_thickness)
                    cv2.circle(image, line[0][idx + 1].astype(int), self.point_radius, color, -1)

    
    def handle_front_lines(self):
        with self.predicator.predicted_lines_lock:
            if len(self.predicator.predicted_lines_queue) > 0:
                self.lines.clear()
            for lines in self.predicator.predicted_lines_queue:
                for line in lines:
                    bev_points = []
                    for point in line.points_n:
                        point = np.copy(point) * np.array([self.predicator.ipm_transformator.img_width, self.predicator.ipm_transformator.img_height])
                        point[1] += 70
                        point = self.predicator.ipm_transformator.calc_bev_point(point)
                        point[0] += (self.image_size[0] - self.predicator.ipm_transformator.img_width) / 2
                        point[1] += (self.image_size[1] - self.predicator.ipm_transformator.img_height) / 2
                        point -= self.pov_point
                        point += self.car_position
                        bev_points.append(point)
                    self.lines.append([np.array(bev_points), line.label])
            self.predicator.predicted_lines_queue.clear()


    def update_car_position(self):
        with self.predicator.car_speed_lock:
            self.car_position += self.predicator.car_speed * 0.00


    def start(self):
        self.is_start = True

        self.map_resource = [None, threading.Lock()]
        self.predicator.drawing_images_lock.acquire()
        self.predicator.drawing_images["map"] = self.map_resource
        self.predicator.drawing_images_lock.release()
        
        while self.is_start:

            self.handle_front_lines()
            self.update_car_position()

            # Отрисовка
            image = np.zeros(shape=tuple(list(self.image_size)[::-1] + [3]), dtype=np.uint8)
            image[:] = np.array(self.background_color, dtype=np.uint8)
            
            self.draw_grid(image)
            self.draw_lines(image)

            cv2.circle(image, (np.array(self.image_size) / 2).astype(int), 5, (0, 0, 255), -1)

            self.map_resource[1].acquire()
            self.map_resource[0] = image
            self.map_resource[1].release()
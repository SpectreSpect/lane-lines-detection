import os
import yaml
import numpy as np
import cv2

class IPMTransformator():
    def __init__(self, config_path):
        self.load_ipm_config(config_path)
        self.src_points = np.zeros(shape=(4, 2), dtype=np.int32)
        self.dst_points = np.zeros(shape=(4, 2), dtype=np.int32)
        
        self.color_src_points = (255, 0, 0)
        self.thickness_src_points_line = 2
        self.radius_src_points = 3
        self.src_keys = "asdf"
        
        self.color_dst_points = (0, 0, 255)
        self.thickness_dst_points_line = 2
        self.radius_dst_points = 3
        self.dst_keys = "zxcv"
    

    def load_ipm_config(self, config_path):
        if not os.path.exists(config_path):
            print('Config file not found. Use default values')
            return
        with open(config_path) as file:
            config = yaml.full_load(file)
        self.homograpthy_matrix = np.array(config['homography'])
        self.horizont_line_height = config['horizont']
        self.img_height = config['height']
        self.img_width = config['width']
    

    def calc_bev_point(self, p, to_sub_horizont_line=True):
        if to_sub_horizont_line:
            p[1] -= self.horizont_line_height
        
        m = self.homograpthy_matrix
        px = ((m[0][0] * p[0] + m[0][1] * p[1] + m[0][2]) / ((m[2][0] * p[0] + m[2][1] * p[1] + m[2][2])))
        py = ((m[1][0] * p[0] + m[1][1] * p[1] + m[1][2]) / ((m[2][0] * p[0] + m[2][1] * p[1] + m[2][2])))
        return np.array([px, py])
    

    def get_ipm(self, image):
        return cv2.warpPerspective(image, self.homograpthy_matrix, (image.shape[1], image.shape[0]))
    

    def find_key(self, key_code, key_queue):
        for idx, saved_key_code in enumerate(key_queue):
            if key_code == saved_key_code:
                key_queue.pop(idx)
                return True
        return False


    def listen_key_queue(self, key_queue, mouse_inf_queue):
        mouse_pos = np.array([0, 0])
        if len(mouse_inf_queue) > 0:
            mouse_pos = np.array([mouse_inf_queue[-1]['x'], mouse_inf_queue[-1]['y']])
        for idx, key in enumerate(self.src_keys):
            if self.find_key(ord(key), key_queue):
                self.src_points[idx] = mouse_pos
        
        for idx, key in enumerate(self.dst_keys):
            if self.find_key(ord(key), key_queue):
                self.dst_points[idx] = mouse_pos
        
        if self.find_key(ord('u'), key_queue):
            matrix, status = cv2.findHomography(self.src_points, self.dst_points)
            if (status != 0).any():
                self.homograpthy_matrix = matrix

    def draw_points(self, image):
        for idx in range(len(self.src_points)):
            cv2.line(image, self.src_points[idx], self.src_points[(idx + 1) % len(self.src_points)], self.color_src_points, thickness=self.thickness_src_points_line)
        
        for point in self.src_points:
            cv2.circle(image, point, self.radius_src_points, self.color_src_points, thickness=-1)
        
        for idx in range(len(self.dst_points)):
            cv2.line(image, self.dst_points[idx], self.dst_points[(idx + 1) % len(self.dst_points)], self.color_dst_points, thickness=self.thickness_dst_points_line)
        
        for point in self.dst_points:
            cv2.circle(image, point, self.radius_dst_points, self.color_dst_points, thickness=-1)
        
        
        
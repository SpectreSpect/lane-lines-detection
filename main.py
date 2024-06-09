# from src.LaneLineModel import LaneLineModel
# from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import math
# from src.utils import *
# from src.dataset_balancing import *
# from src.reinforcement_data import *
# from src.from_xml_to_yolo import *
# from src.dataset import *
import re
import time
import cv2


def draw_line(image, line: np.array, color=(255, 255, 255)):
   if len(line) < 2:
      raise Exception("Line should consist of at least 2 points")
   
   image_shape = image.shape[:2]

   for i2 in range(1, len(line)):
      i1 = i2 - 1
      pt1 = (np.array(line[i1]) * image_shape).astype(int)
      pt2 = (np.array(line[i2]) * image_shape).astype(int)
      cv2.line(image, pt1, pt2, color, 2)
      cv2.circle(image, pt1, 5, (0, 0, 255), -1)
      cv2.circle(image, pt2, 5, (0, 0, 255), -1)


def draw_lines(image, lines: np.array):
   for line in lines:
      draw_line(image, line)


# def merge_lines(lines):

def trace_line(line):
   acceleration = np.zeros(2)
   velocity1 = np.zeros(2)
   velocity2 = np.zeros(2)

   traced_points = []
   if len(line) > 2:
      pt1 = np.array(line[-3])
      pt2 = np.array(line[-2])
      pt3 = np.array(line[-1])

      velocity1 = pt2 - pt1
      velocity2 = pt3 - pt2
      acceleration = velocity2 - velocity1

      last_point = pt3
      velocity = velocity2
      for i in range(10):
         velocity += acceleration
         new_point = last_point + velocity
         traced_points.append(new_point)
         last_point = new_point
   
   return traced_points


def get_score(point1, line, max_radius, min_cos, inc_dir, dir_importance):
   diff0 = line[0] - point1
   dist0 = np.linalg.norm(diff0)

   if dist0 > max_radius:
      return -1
   dist_score0 = (1.0 - (dist0 / max_radius)) * (1 - dir_importance)
   
   dir0 = diff0 / dist0
   cos0 = np.dot(inc_dir, dir0)

   if cos0 < min_cos:
      return -1

   cos_score0 = abs(cos0) * dir_importance

   return (dist_score0 + cos_score0)


def find_appropriate_endpoint(pt0, pt1, lines, dir_importance):
   point0 = np.array(pt0)
   point1 = np.array(pt1)
   inc_diff = point1 - point0
   inc_dist = np.linalg.norm(inc_diff)
   inc_dir = (inc_diff - inc_dist) / inc_dist

   max_radius = 0.7

   best_score = 999
   best_line_id = -1
   best_point_id = -1
   for line_id, line in enumerate(lines):
      score0 = get_score(point1, line[0], max_radius, inc_dir, dir_importance)
      score1 = get_score(point1, line[1], max_radius, inc_dir, dir_importance)
      
      point_id = (len(line[1]) - 1) if score1 < score0 else 0
      score = score1 if score1 < score0 else score0
      if score < best_score:
         best_score = score
         best_line_id = line_id
         best_point_id = point_id

   return best_line_id, best_point_id
      

def get_score_2(dir, point1, point2, max_radius, min_cos, dir_importance):
   diff0 = np.array(point2) - np.array(point1)
   dist0 = np.linalg.norm(diff0)

   if dist0 > max_radius:
      return -1     
   dist_score0 = (1.0 - (dist0 / max_radius)) * (1 - dir_importance)

   dir0 = diff0 / dist0
   cos0 = np.dot(inc_dir, dir0)

   if cos0 < min_cos:
      return -1
   cos_score0 = abs(cos0) * dir_importance

   return (dist_score0 + cos_score0)


def merge_lines(lines):
   pass
   # for line1 in lines:
   #    for line2 in lines:

      

if __name__ == "__main__":
   # model = LaneLineModel("models/LLD-2.pt")

   lines = [
      [[0.1, 0.5], [0.2, 0.45], [0.3, 0.425]],
      [[0.35, 0.5], [0.4, 0.52], [0.6, 0.5]],
      [[0.5, 0.4], [0.6, 0.42], [0.7, 0.44], [0.8, 0.5]]
   ]
   

   image = np.zeros((800, 800, 3), dtype=np.uint8)

   # traced_line = trace_line(lines[0])

   # best_line_id, best_point_id = find_appropriate_endpoint(lines[0][-2], lines[0][-1], lines[1:], 1)



   draw_lines(image, lines)

   inc_diff = (np.array(lines[0][-1]) - np.array(lines[0][-2]))
   inc_dir = inc_diff / np.linalg.norm(inc_diff)
   inc_point = lines[0][-1]

   best_score = -1
   best_point = [0, 0]
   for line in lines[1:]:
      for point in line:
         score = get_score_2(inc_dir, inc_point, point, 0.7, 0.7, 0.7)
         if score >= 0:
            pt = (np.array(point) * image.shape[:2]).astype(int)
            # color_intensity = int((255 * (score ** 2)))
            # cv2.circle(image, pt, 20, (color_intensity, 0, 0), -1)

         if score > best_score:
            best_score = score
            best_point = pt
   
   cv2.circle(image, best_point, 20, (255, 0, 0), -1)

   # pt = (np.array(lines[best_line_id][best_point_id]) * image.shape[:2]).astype(int)
   # cv2.circle(image, pt, 20, (255, 0, 0), -1)


   # draw_line(image, traced_line, (255, 100, 100))

   
   
   

   cv2.imshow("Image", image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()




from .abstract_intersection_line import AbstractIntersectionLine
from .line_points.line_point import LinePoint
from .line_points.lane_point import LanePoint
from typing import List
import cv2
import numpy as np

class HorizontalIntersectionLine(AbstractIntersectionLine):
    def __init__(self, y_n):
        origin = np.array([0, y_n])
        direction = np.array([1, 0])
        super().__init__(origin, direction, [])
    
    
    def set_lane_points(self, lines, min_distance=0.015, distance_to_error=0.015):
        for line in lines:
            if line.label == 14:
                pass
            for point in line.points_n:
                if abs(point[1] - self.origin[1]) < min_distance:
                    lane_point = LanePoint(point[0], line.label)
                    self.insert_point(lane_point)
                    break
                
        self.sort_points()
        
        idx = 0
        prev_line_point_idx = -1
        while idx < len(self.line_points) - 1:
            if isinstance(self.line_points[idx], LanePoint):
                if prev_line_point_idx < 0:
                    prev_line_point_idx = idx
                    idx += 1
                    continue
                
                point = self.line_points[idx]
                prev_point = self.line_points[prev_line_point_idx] 
                if point.label == prev_point.label and abs(prev_point.t - point.t) <= distance_to_error:
                   self.line_points.pop(idx)
                   continue

                prev_line_point_idx = idx
            idx += 1
        
        return self
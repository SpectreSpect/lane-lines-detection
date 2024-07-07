from .line_points.line_point import LinePoint
from typing import List
import numpy as np

class AbstractIntersectionLine():
    def __init__(self, origin: np.ndarray, direction: np.ndarray, line_points: List[LinePoint] = []):
        self.origin = origin
        self.direction = direction
        self.line_points = line_points


    
    def insert_point(self, point: LinePoint):
        self.line_points.append(point)


    def sort_points(self):
        self.line_points = sorted(self.line_points, key=lambda point: -point.t)
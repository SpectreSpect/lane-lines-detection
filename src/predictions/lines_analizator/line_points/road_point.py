from .line_point import LinePoint

class RoadPoint(LinePoint):
    def __init__(self, t, road_id):
        super().__init__(t)
        self.road_id = road_id
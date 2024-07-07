from .line_point import LinePoint

class LanePoint(LinePoint):
    def __init__(self, t, label):
        super().__init__(t)
        self.label = label
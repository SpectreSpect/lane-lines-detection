import cv2
import numpy as np

class Accelerometer():
    def __init__(self, predictor):
        self.predictor = predictor
        self.prev_frame = None
        self.prev_points = None
        # self.accumulated_speeds = []

        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    
    def to_reset_acceleration(self):
        self.prev_frame = None
        self.prev_points = None


    def get_acceleration(self, frame, reset_acceleration=False):
        if reset_acceleration:
           self.to_reset_acceleration()

        frame = cv2.resize(frame, (self.predictor.ipm_transformator.img_width, self.predictor.ipm_transformator.img_height))
        frame = self.predictor.ipm_transformator.get_ipm(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_points = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
            self.prev_frame = frame
            return np.array([0, 0], dtype=np.float32)
        
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame, self.prev_points, None, **self.lk_params)

        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        speeds = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            speeds.append(np.array([b - d, a - c]))
        
        return np.mean(np.array(speeds), axis=0)

        # if self.accumulated_speeds:
        #     avg_speed = np.mean(speeds)
        #     prev_avg_speed = np.mean(self.accumulated_speeds[-1])
        #     acceleration = (avg_speed - prev_avg_speed) * fps
        #     self.accumulated_speeds.append(speeds)
        # else:
        #     self.accumulated_speeds.append(speeds)
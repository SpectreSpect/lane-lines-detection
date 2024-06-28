import numpy as np
from ..data.mask import Mask
import cv2
# from ..containers.image_container import ImageContainer


class Visualizer():
    def __init__(self):
        pass

    def show_image(self, image: np.ndarray):
        rgb_image = image.copy()
        
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        cv2.imshow("Image", bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def draw_mask(image: np.ndarray, mask: Mask, color: list):
    #     # cv2.drawContours(image, )
    #     print("Drawing a mask!!!")
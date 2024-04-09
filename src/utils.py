import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

default_palette = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255), 
    (255, 0, 255), 
    (255, 255, 0), 
    (150, 255, 255)
]

def draw_segmentation_cv2(image, predict, alpha=0.4, palette=default_palette):
    if predict.masks is None:
        return []

    mask_image = np.zeros(image.shape[:-1], dtype=np.uint8)
    for idx, xy in enumerate(predict.masks.xy):
        color = palette[idx % len(palette)]
        cv2.drawContours(mask_image, [np.expand_dims(xy, 1).astype(int)], contourIdx=-1, color=(255), thickness=-1)
        
        indices = mask_image != 0 
        image[indices] = image[indices] * (1 - alpha) + np.array(color) * alpha
        mask_image[:] = 0

    return image

    # rows = np.ceil(len(images) / float(columns))
    # fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(10, 5))

    # axes = axes.flatten()
    # for (idx, (ax, image)) in enumerate(zip(axes, images)):
    #     if idx >= len(axes):
    #         break

    #     ax.imshow(image)
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.show()

def show_images(images, figsize=(15, 5), count_images_for_ineration=2, columns=2):
    for slice_id in range(len(images) // count_images_for_ineration):
        rows = math.ceil(count_images_for_ineration / float(columns))
        fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)

        slice_min = slice_id * count_images_for_ineration
        slice_max = (slice_id + 1) * count_images_for_ineration

        axes = axes.flatten()
        for (idx, (ax, image)) in enumerate(zip(axes, images[slice_min:slice_max])):
            if idx >= len(axes):
                break

            ax.imshow(image)
            ax.axis('off')

        plt.tight_layout()
        plt.show()
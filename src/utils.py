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

def draw_segmentation_(image, predict, alpha=0.4, palette=default_palette):
    if predict.masks is None:
        return []

    mask_image = np.zeros(image.shape[:-1], dtype=np.uint8)
    for idx, xy in enumerate(predict.masks.xy):
        if xy.shape[0] == 0:
            break
        color = palette[int(predict.boxes.cls[idx]) % len(palette)]
        cv2.drawContours(mask_image, [np.expand_dims(xy, 1).astype(int)], contourIdx=-1, color=(255), thickness=-1)
        
        indices = mask_image != 0 
        image[indices] = image[indices] * (1 - alpha) + np.array(color) * alpha
        mask_image[:] = 0

    return image

def draw_segmentation(images, predictions, alpha=0.4, palette=default_palette):
    for (image, predict) in zip(images, predictions):
        draw_segmentation_(image, predict, alpha, palette)
    
    if len(images) == 1:
        return images[0]
    
    return images


def draw_lines(images, batch_lines, palette=default_palette, thickness=4):
    for (image, mask_lines) in zip(images, batch_lines):
        for idx, (cls, line) in enumerate(mask_lines):
            color = palette[cls % len(palette)]
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)


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
            
def view_prediction_video(model, src):
    cap = cv2.VideoCapture(src)
    # if not cap.isOpened():
    #     print("Не удалось открыть видеофайл.")
    #     exit()
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        # Обработка изображения

        predictions = model.model.predict([image])
        batch_lines = model.get_lines(predictions) # Проверка get_lines()

        draw_segmentation([image], predictions)
        draw_lines([image], batch_lines)
        cv2.imshow('prediction video', image)

        key_code = cv2.waitKey(5) & 0xFF
        if key_code == ord('q'):
            break
        
    
    cap.release()

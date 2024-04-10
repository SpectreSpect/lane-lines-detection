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
    if not cap.isOpened():
        print("Не удалось открыть файл.")
        cap.release()
        return
    
    while cap.isOpened():
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


def get_lines_graph(image, predict, chunk_size=200, offset_size=0):
    if predict.masks is None:
        return []
    
    masks = predict.masks

    mask_image = np.zeros(masks.orig_shape + (1,), dtype=np.uint8)

    for xy in masks.xy:
        cv2.drawContours(mask_image, [np.expand_dims(xy, 1).astype(np.int32)], contourIdx=-1, color=(255), thickness=-1)

    count_chunks_x = math.ceil(masks.orig_shape[1] / (chunk_size + offset_size))
    count_chunks_y = math.ceil(masks.orig_shape[0] / (chunk_size + offset_size))

    lines = [[]] * count_chunks_x
    for x in range(count_chunks_x):
        lines[x] = [[]] * count_chunks_y
        for y in range(count_chunks_y):
            chunk_x_min = (chunk_size + offset_size) * x
            chunk_x_max = (chunk_size + offset_size) * x + chunk_size

            chunk_y_min = (chunk_size + offset_size) * y
            chunk_y_max = (chunk_size + offset_size) * y + chunk_size

            cv2.rectangle(image, (chunk_x_min, chunk_y_min), (chunk_x_max, chunk_y_max), (y / count_chunks_y * 255, 0, x / count_chunks_x * 255), 3)

            chunk = mask_image[chunk_y_min:chunk_y_max, chunk_x_min:chunk_x_max]
            
            chunk_lines = cv2.HoughLinesP(chunk, 1, np.pi / 180, threshold=100, minLineLength=25, maxLineGap=30)
            if chunk_lines is not None:
                lines[x][y] = chunk_lines

                for line in chunk_lines:
                    x1, y1, x2, y2 = line[0]
                    x1 += chunk_x_min
                    y1 += chunk_y_min
                    x2 += chunk_x_min
                    y2 += chunk_y_min
                    cv2.line(image, (x1, y1), (x2, y2), default_palette[0], thickness=3)

            else:
                lines[x][y] = []
    
    return lines
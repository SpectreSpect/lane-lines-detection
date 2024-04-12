import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

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

def draw_segmentation(images, predictions, alpha=0.2, palette=default_palette):
    for (image, predict) in zip(images, predictions):
        draw_segmentation_(image, predict, alpha, palette)
    
    if len(images) == 1:
        return images[0]
    
    return images


def draw_lines(images, batch_lines, palette=default_palette, thickness=8):
    for (image, mask_lines) in zip(images, batch_lines):
        for idx, (cls, line) in enumerate(mask_lines):
            color = palette[cls % len(palette)]
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)


def draw_curves(images, batch_curves, palette=default_palette, thickness=4):
    for (image, mask_curves) in zip(images, batch_curves):
        for idx, (cls, curve) in enumerate(mask_curves):
            color = palette[cls % len(palette)]
            for id in range(1, len(curve)):
                cv2.line(image, curve[id - 1].astype(int), curve[id].astype(int), color, thickness=thickness)


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
    
    i = 0
    
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        
        # Обработка изображения

        predictions = model.model.predict([image])
        batch_lines = model.get_lines(predictions)

        draw_segmentation([image], predictions)
        #draw_lines([image], batch_lines)
        draw_curves([image], batch_lines)
        cv2.imshow('prediction video', image)

        key_code = cv2.waitKey(5) & 0xFF
        if key_code == ord('q'):
            break
        
        i += 1
    
    cap.release()


def get_straight_lines(results):
    batch_lines = []
    for result in results:
        masks = result.masks
        if masks is None:
            return []

        mask_image = np.zeros(masks.orig_shape + (1,), dtype=np.uint8)
        
        mask_lines = []
        for xy, cls in zip(masks.xy, result.boxes.cls):
            if xy.shape[0] == 0:
                break
            cv2.drawContours(mask_image, [np.expand_dims(xy, 1).astype(np.int32)], contourIdx=-1, color=(255), thickness=-1)
            lines = cv2.HoughLinesP(mask_image, 1, np.pi / 180, threshold=100, minLineLength=25, maxLineGap=30)
        
            if lines is not None:
                best_line = None
                max_lenght = 0

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    lenght = np.linalg.norm([x2-x1, y2-y1])
                    if best_line is None or lenght > max_lenght:
                        max_lenght = lenght
                        best_line = line
                mask_lines.append([int(cls), best_line])
            
            mask_image[:] = 0
        batch_lines.append(mask_lines)
    return batch_lines

    
def correct_point(xy, const_point_id, moving_point_id, max_distance, dist_accum_factor, n_accum, direction=[1, -1], decrease_distance=True):
    temp_point2_id = moving_point_id
    best_dis = np.linalg.norm(xy[const_point_id] - xy[temp_point2_id])
    best_point_id = temp_point2_id
    
    for dir in direction:
        i = 0
        prev_dist = best_dis
        accum_dist_change = 0
        while True:
            temp_point2_id = (temp_point2_id + dir) % xy.shape[0]

            if temp_point2_id == const_point_id:
                return -1
            
            distance = np.linalg.norm(xy[const_point_id] - xy[temp_point2_id])
            dist_change = distance - prev_dist
            prev_dist = distance
            accum_dist_change = accum_dist_change * dist_accum_factor + dist_change * (1 - dist_accum_factor)

            if decrease_distance:
                if distance < best_dis:
                    best_dis = distance
                    best_point_id = temp_point2_id
                
                if (i >= n_accum and accum_dist_change > 0):
                    break
            else:
                if distance > best_dis:
                    best_dis = distance
                    best_point_id = temp_point2_id
                
                if (i >= n_accum and accum_dist_change < 0):
                    break
            
            i += 1
    
    return best_point_id


def get_line_contour(predict, max_distance=100, min_id_dis_ratio=0.5, edge_point_dis=20, dist_accum_factor=0.8, n_accum=5, tolerance=0.0001):
    masks = predict.masks
    if masks is None:
        return []
    
    mask_lines = []
    for xyn, cls in zip(masks.xyn, predict.boxes.cls):
        simplified_polygon = Polygon(xyn).simplify(tolerance, preserve_topology=True)
        points = np.array(simplified_polygon.exterior.coords)
        points *= np.array([masks.orig_shape[1], masks.orig_shape[0]])
        
        if points.shape[0] == 0:
            break
        min_id_dis = int(points.shape[0] * min_id_dis_ratio)

        start_point1_id = np.random.randint(0, points.shape[0])
        start_point2_id = start_point1_id

        # best_distance = 0
        # best_point_id = -1
        # monotone = False
        # count_pass = 0
        # while True:

        #     if count_pass > 6:
        #         break

        #     start_point2_id = correct_point(points, start_point1_id, start_point2_id, max_distance, dist_accum_factor, n_accum, [1], monotone)
        #     monotone = not monotone

        #     if start_point2_id == -1:
        #         if best_distance > max_distance:
        #             start_point1_id = np.random.randint(0, points.shape[0])
        #             start_point2_id = start_point1_id

        #             best_distance = 0
        #             best_point_id = -1
        #             monotone = False
        #             count_pass += 1
        #             continue

        #     if start_point2_id == -1:
        #         break

        #     distance = np.linalg.norm(points[start_point2_id] - points[start_point1_id])
        #     if best_point_id < 0 or (distance < best_distance):
        #         best_distance = distance
        #         best_point_id = start_point2_id
        
        # if count_pass > 5:
        #     mask_lines.append([int(cls), []])
        #     break

        # start_point2_id = best_point_id

        




        diff_start_point_1 = (points[(start_point1_id + 1) % points.shape[0]] - points[start_point1_id])
        dir_start_point_1 = diff_start_point_1 / np.linalg.norm(diff_start_point_1)

        best_dis = max_distance + 1
        best_start_point2_id = start_point2_id
        i = 1 + min_id_dis
        count_pass = 0
        while i < points.shape[0] + 1:
            start_point2_id = (start_point1_id - i) % points.shape[0]
            diff_start_point_2 = (points[(start_point2_id - 1) % points.shape[0]] - points[start_point2_id])
            dir_start_point_2 = diff_start_point_2 / np.linalg.norm(diff_start_point_2)

            distance = np.linalg.norm(points[start_point2_id] - points[start_point1_id])

            if distance <= max_distance:
                if distance < best_dis:
                    best_dis = distance
                    best_start_point2_id = start_point2_id
                    if best_dis <= edge_point_dis:
                        count_pass += 1
                        if count_pass >= min_id_dis:
                            break
                        i = (1 + min_id_dis - count_pass)
                        best_dis = max_distance + 1
                        start_point1_id = np.random.randint(0, points.shape[0])
                        start_point2_id = start_point1_id
                else:
                    start_point2_id = best_start_point2_id
                    break
            
            i += 1
        if count_pass >= min_id_dis:
            mask_lines.append([int(cls), []])
            break
            
        start_point2_id = correct_point(points,start_point1_id, start_point2_id, max_distance, dist_accum_factor, n_accum)

        moving_point1_id = start_point1_id
        moving_point2_id = start_point2_id


        line = [(points[moving_point1_id] + points[moving_point2_id]) / 2]
        back_line = []
        for dir in [1, -1]:
            moving_point1_id = start_point1_id
            moving_point2_id = start_point2_id

            while True:
                moving_point1_id = (moving_point1_id + dir) % points.shape[0]
                moving_point2_id = correct_point(points, moving_point1_id, moving_point2_id, max_distance, dist_accum_factor, n_accum, [-dir])

                if moving_point2_id == -1:
                    break
                else:
                    if dir == 1:
                        line = line + [(points[moving_point1_id] + points[moving_point2_id]) / 2]
                    else:
                        line = [(points[moving_point1_id] + points[moving_point2_id]) / 2] + line
        
        mask_lines.append([int(cls), line])

    return mask_lines
        

def get_lines_contours(predicts, max_distance=100, min_id_dis_ratio=0.5, edge_point_dis=20, dist_accum_factor=0.99, n_accum=10):
    lines = []
    for predict in predicts:
        lines.append(get_line_contour(predict, max_distance, min_id_dis_ratio, edge_point_dis, dist_accum_factor, n_accum))
    return lines
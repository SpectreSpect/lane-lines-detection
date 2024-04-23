import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from timeit import default_timer as timer
import time
import os

# default_palette = [
#     (255, 0, 0),
#     (0, 255, 0),
#     (0, 0, 255),
#     (0, 255, 255), 
#     (255, 0, 255), 
#     (255, 255, 0), 
#     (150, 255, 255)
# ]

default_palette = [
    (0, 0, 0), # unkown
    (85, 172, 238), # white-dash
    (120, 177, 89), # white-solid
    (0, 0, 0), # double-white-dash
    (0, 0, 0), # double-white-solid
    (0, 0, 0), # white-ldash-rsolid
    (0, 0, 0), # white-lsolid-rdash
    (0, 0, 0), # yellow-dash
    (253, 203, 88), # yellow-solid
    (0, 0, 0), # double-yellow-dash
    (244, 144, 12), # double-yellow-solid
    (221, 46, 68), # yellow-ldash-rsolid
    (0, 0, 0), # yellow-lsolid-rdash
    (0, 0, 0), # left-curbside
    (0, 0, 0), # right-curbside
]


class LaneMask():
    def __init__(self, points: np.ndarray = [], label: int = []):
        self.points = points
        self.label = label
            
    
    @staticmethod
    def from_predictions(predictions, tolerance=0.0015) -> list:
        lane_masks = []
        if predictions.masks is not None:
            for (xyn, cls) in zip(predictions.masks.xyn, predictions.boxes.cls):          
                simplified_polygon = Polygon(xyn).simplify(tolerance, preserve_topology=True)
                
                points = np.array(simplified_polygon.exterior.coords)             
                label = int(cls)

                lane_mask = LaneMask(points, label)
                lane_masks.append(lane_mask)
        return lane_masks


class LaneLine():
    def __init__(self, points: np.ndarray, label: int, elapsed_time=0, mask_count_points=0):
        self.points = points
        self.label = label
        self.elapsed_time = elapsed_time
        self.mask_count_points=mask_count_points



def draw_segmentation_(image, predict, alpha=0.4, palette=default_palette):
    if predict.masks is None:
        return []

    mask_image = np.zeros(image.shape[:-1], dtype=np.uint8)
    for idx, xy in enumerate(predict.masks.xy):
        if xy.shape[0] == 0:
            break
        color = palette[int(predict.boxes.cls[idx]) % len(palette)]
        color = (color[2], color[1], color[0])
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


def draw_lines(images, batch_curves, palette=default_palette, thickness=4):
    for (image, mask_curves) in zip(images, batch_curves):
        for idx, lane_line in enumerate(mask_curves):
            color = palette[lane_line.label % len(palette)]
            color = (color[2], color[1], color[0])
            for id in range(1, len(lane_line.points)):
                cv2.line(image, lane_line.points[id - 1], lane_line.points[id], color, thickness=thickness)


def show_images(images, figsize=(15, 5), count_images_for_ineration=2, columns=2):
    columns = min(len(images), columns)
    count_images_for_ineration = min(len(images), count_images_for_ineration)
    for slice_id in range(len(images) // count_images_for_ineration):
        rows = math.ceil(count_images_for_ineration / float(columns))
        fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)

        slice_min = slice_id * count_images_for_ineration
        slice_max = (slice_id + 1) * count_images_for_ineration

        if rows * columns > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for (idx, (ax, image)) in enumerate(zip(axes, images[slice_min:slice_max])):
            if idx >= len(axes):
                break

            ax.imshow(image)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def get_mean_elapsed_time(batch_lines: list):
    mean_elapsed_time = 0
    count = 0
    for masks in batch_lines:
        for line in masks:
            mean_elapsed_time += line.elapsed_time
            count += 1
    
    if count == 0:
        return 0

    return mean_elapsed_time / count

def paint_str(string: str, color):
    start = "\033[38;2;{};{};{}m"
    end = "\033[0m"
    text = start + string + end
    return text.format(color[0], color[1], color[2])
    # print(text.format(color[0], color[1], color[2]))


def view_prediction_video(model, src, save_predictions=False, verbose=0):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Не удалось открыть файл.")
        cap.release()
        return
    

    mean_hertz = 0
    mean_elapsed_time = 0
    i = 0

    count_tests = 500
    mean_elapsed_time = 0

    max_count_points = 5000
    measurements = np.zeros((max_count_points,), dtype=np.float32)
    hits = np.zeros((max_count_points,), dtype=np.int32)
    
    ret_images = []
    ret_predictions = []

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        
        if verbose != 0:
            print(f"id: {i}")

        # Обработка изображения
        start = timer()
        predictions = model.model.predict([image], verbose=False)
        end = timer()

        if save_predictions:
            ret_images.append(image.copy())
            # cv2.imwrite(os.path.join("test", "sdfsdfsdf" + ".jpg"), ret_images[0])
            ret_predictions.append(predictions[0])

        elapsed_time = end - start
        hertz = 1.0 / elapsed_time
        
        mean_hertz += hertz 
        mean_elapsed_time += elapsed_time * 1000.0 # ms

        if verbose != 0:
            print(f"frame: {i + 1}     elapsed time: {round(elapsed_time * 1000.0, 2)} ms")
            print(f"hertz: {round(hertz, 2)}     mean hertz: {round(mean_hertz / float(i + 1), 2)}    mean elapsed time: {round(mean_elapsed_time / float(i + 1), 2)} ms")
            print()

        batch_lines = model.get_lines(predictions)

        for masks in batch_lines:
            for line in masks:
                measurements[line.mask_count_points] += line.elapsed_time
                hits[line.mask_count_points] += 1

        mean_elapsed_time += get_mean_elapsed_time(batch_lines)


        if i > 0 and i % count_tests == 0:
            if verbose != 0:
                print(f"Test {i // count_tests}. Elapsed time = {mean_elapsed_time / count_tests * 1000}")

            # measurements /= hits
            # x = np.linspace(0, max_count_points, max_count_points)
            # plt.plot(x, measurements)
            # plt.title("Сложность алгоритма")
            # plt.xlabel("Количество точек контура")
            # plt.ylabel("Время")
            # plt.show()
            # measurements *= hits

            mean_elapsed_time = 0

        draw_segmentation([image], predictions)
        #draw_lines([image], batch_lines)
        draw_lines([image], batch_lines)
        cv2.imshow('prediction video', image)

        key_code = cv2.waitKey(5) & 0xFF
        if key_code == ord('q'):
            break
        
        i += 1
    
    cap.release()
    return ret_images, ret_predictions


def get_straight_lines(results):
    batch_lines = []
    for result in results:
        masks = result.masks
        if masks is None:
            batch_lines.append([])
            continue

        mask_image = np.zeros(masks.orig_shape + (1,), dtype=np.uint8)
        
        mask_lines = []
        for xy, cls in zip(masks.xy, result.boxes.cls):
            if xy.shape[0] == 0:
                break

            t1 = time.time()
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

                t2 = time.time()
                mask_lines.append(LaneLine(np.reshape(np.array(list(best_line), dtype=np.int32), (2, 2)), int(cls), t2-t1, int(xy.shape[0])))
            
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


def get_line_contour(
        predict, 
        max_distance=100, 
        min_id_dis_ratio=0.5, 
        edge_point_dis=20, 
        dist_accum_factor=0.8, 
        n_accum=5, 
        tolerance=0.0001) -> list:
    masks = predict.masks
    if masks is None:
        return []
    
    mask_lines = []
    for xyn, cls in zip(masks.xyn, predict.boxes.cls):
        t1 = time.time()
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
            mask_lines.append(LaneLine(np.array([], dtype=np.int32), int(cls)))
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
        
        t2 = time.time()
        mask_lines.append(LaneLine(np.array(line, dtype=np.int32), int(cls), t2-t1, len(line)))

    return mask_lines
        

def get_lines_contours(predicts, max_distance=100, min_id_dis_ratio=0.5, edge_point_dis=20, dist_accum_factor=0.99, n_accum=10):
    lines = []
    for predict in predicts:
        lines.append(get_line_contour(predict, max_distance, min_id_dis_ratio, edge_point_dis, dist_accum_factor, n_accum))
    return lines


def str_to_seconds(string: str):
    splits = string.split(':')
    
    seconds = 0
    minutes = 0
    hours = 0
    if len(splits) == 1:
        seconds = int(splits[0])
    elif len(splits) == 2:
        minutes = int(splits[0])
        seconds = int(splits[1])
    elif len(splits) == 3:
        hours = int(splits[0])
        minutes = int(splits[1])
        seconds = int(splits[2])
    
    output = seconds + minutes * 60 + hours * 60 * 60
    return output
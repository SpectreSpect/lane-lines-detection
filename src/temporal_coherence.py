import numpy as np
import math


def intersection_area(rect1, rect2):
    left_x_inteval = min(max(rect2[0][0] - rect1[0][0], 0), rect1[1][0] - rect1[0][0])
    right_x_inverval = min(max(rect1[1][0] - rect2[1][0], 0), rect1[1][0] - rect1[0][0])
    x_intersection = (rect1[1][0] - rect1[0][0]) - left_x_inteval - right_x_inverval

    left_y_inteval = min(max(rect2[0][1] - rect1[0][1], 0), rect1[1][1] - rect1[0][1])
    right_y_inverval = min(max(rect1[1][1] - rect2[1][1], 0), rect1[1][1] - rect1[0][1])
    y_intersection = (rect1[1][1] - rect1[0][1]) - left_y_inteval - right_y_inverval

    return x_intersection * y_intersection


def temporal_update_frame(frame_lines_queue, current_frame_lines, frame_lines_queue_length=2, precent_to_save=0.4):
    for prev_line in frame_lines_queue[-1]:
        max_intersection_area = 0
        for current_line in current_frame_lines:
            area = intersection_area(prev_line.bounding_box, current_line.bounding_box)
            if area > max_intersection_area:
                max_intersection_area = area
        if max_intersection_area / (prev_line.bounding_box[0] * prev_line.bounding_box[1]) < precent_to_save:
            current_frame_lines.append(prev_line)
    
    frame_lines_queue.append(current_frame_lines)
    if len(frame_lines_queue) > frame_lines_queue_length:
        frame_lines_queue.pop(0)
    
    return frame_lines_queue

    
    
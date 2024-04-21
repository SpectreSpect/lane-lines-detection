import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
import time
from src.utils import *

def find_bc(point1, point2, a):
    b = (point2[1] - point1[1]) / (point2[0] - point1[0]) - a * (point1[0] + point2[0])
    c = point1[1] - a * point1[0]*point1[0] - point1[0]*b
    return b, c


def fit_polynom(x, y, count_epochs=5, a=0.0, learning_rate=0.01):
    min_id = np.argmin(x)
    max_id = np.argmax(x)
    
    point1 = [x[min_id], y[min_id]]
    point2 = [x[max_id], y[max_id]]

    b, c = find_bc(point1, point2, a)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    params = torch.tensor([a, b, c], requires_grad=True)

    optimizer = torch.optim.Adam([params], lr=learning_rate, betas=(0.9, 0.999))

    for epoch in range(count_epochs):
        y_pred = params[0]*x*x + params[1]*x + params[2]
        loss = torch.sum(torch.pow(y_pred - y, 2))

        #print(f'{epoch}: {loss.item()}')

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    
    return params.detach().numpy().tolist(), point1, point2


def get_polynom_line(predict, count_epochs=0, a=0.0, learning_rate=0.1, offset=10):
    masks = predict.masks
    if masks is None:
        return []
    
    mask_lines = []
    for xy, cls in zip(masks.xy, predict.boxes.cls):
        t1 = time.time()
        (local_a, local_b, local_c), point1, point2 = fit_polynom(xy[:, 1], xy[:, 0], count_epochs, a, learning_rate)
        
        count_points = int(point2[0] - point1[0]) // offset
        x_pred = np.linspace(point1[0], point2[0], count_points)
        y_pred = local_a * x_pred * x_pred + local_b * x_pred + local_c
        
        points = np.zeros((count_points, 2), dtype=np.int32)
        points[:, 0] = y_pred # y координата (не перепутал, так и есть)
        points[:, 1] = x_pred # x координата
        
        t2 = time.time()
        
        mask_lines.append(LaneLine(points, int(cls), t2-t1, int(xy.shape[0])))
    
    return mask_lines
    

def get_polynom_lines(predicts, count_epochs=100, a=0.0, learning_rate=0.1):
    lines = []
    for predict in predicts:
        lines.append(get_polynom_line(predict, count_epochs, a, learning_rate))
    return lines
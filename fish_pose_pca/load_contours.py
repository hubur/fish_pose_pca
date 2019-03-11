"""utility for loading contour points from the biotracker export."""
import json
from typing import List
import numpy as np
import cv2


class Contour:
    def __init__(self, data: List):
        self.xs = np.array([d["x"] for d in data])
        self.ys = np.array([d["y"] for d in data])
        contour = cv2.UMat(np.array([np.array([np.array([x, y])]) for x, y in zip(self.xs, self.ys)]))
        moments = cv2.moments(contour)
        self.c_x = int(moments["m10"] / moments["m00"])
        self.c_y = int(moments["m01"] / moments["m00"])


def _get_contours(line: str):
    return [Contour(c) for c in json.loads(line)]


def load_contours(path: str):
    with open(path) as data:
        lines = data.readlines()
    contours = []
    for line in lines:
        for data in json.loads(line):
            contours.append(Contour(data))
    return contours

"""utility for loading contour points from the biotracker export."""
import json
from typing import List
import numpy as np
import cv2
import config


class Contour:
    def __init__(self, data: List):
        self.xs = np.array([d["x"] for d in data])
        self.ys = np.array([d["y"] for d in data])
        contour = cv2.UMat(np.array([np.array([np.array([x, y])])
                                     for x, y in zip(self.xs, self.ys)]))
        moments = cv2.moments(contour)
        self.c_x = int(moments["m10"] / moments["m00"])
        self.c_y = int(moments["m01"] / moments["m00"])


def _get_contours(line: str):
    return [Contour(c) for c in json.loads(line)]


def load_contours(path: str = None, mask_path: str = None, threshold: float = None):
    """load contours from biotracker export.
    Args:
        path:      path of exported jsonl file
        mask_path: path to an image that has the dimensions of the video,
                   and has all pixels outside of the arena black, all pixels
                   inside the arena white.
        threshold: which amount of contour points need to be inside the arena.
                   whith threshold = 1, all points need to be in the arena,
                   lower values allow some contour points to be outside.
    """
    if not path:
        path = config.get_contour_data_path()
    if not threshold:
        threshold = config.get_mask_threshold()
    if not mask_path:
        mask_path = config.get_mask_path()
    if mask_path:
        arena_mask = cv2.imread(mask_path)
        arena_mask = cv2.cvtColor(arena_mask, cv2.COLOR_BGR2GRAY)
    with open(path) as data:
        lines = data.readlines()
    contours = []
    min_points = config.get_min_contour_points()
    for line in lines:
        for data in json.loads(line):
            contour = Contour(data)
            outside_arena = mask_path and (np.mean(arena_mask[contour.ys, contour.xs]) < threshold)
            if outside_arena or contour.xs.shape[0] < min_points:
                continue
            contours.append(contour)
    return contours

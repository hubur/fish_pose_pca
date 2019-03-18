"""utility for loading contour points from the biotracker export."""
import json
import numpy as np
import cv2
import config


def poly_area(x, y):
    """calculate polygon area using shoelace formula"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def load_biotracker_export(path: str = None, mask_path: str = None, threshold: float = None, max_lines = None):
    """load contours, centroids and frame metadata from biotracker export.
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
    centroids = []
    # The metadata list will include tuples (frame_id, fish_id)
    # The Position in the list maps to the position in the contours list
    metadata = []
    min_points = config.get_min_contour_points()

    areas = []
    too_small = 0
    if max_lines:
        lines = lines[:max_lines]
    for line in lines:
        data = json.loads(line)

        X = np.array(data["xs"])
        Y = np.array(data["ys"])
        areas.append(poly_area(X, Y))

        x_centroid = data["center_x"]
        y_centroid = data["center_y"]

        outside_arena = mask_path and arena_mask[int(y_centroid), int(x_centroid)] == 0

        if outside_arena or poly_area(X, Y) < threshold:
            too_small += 1
            continue

        contours.append((X, Y))
        centroids.append(np.array([x_centroid, y_centroid]))
        metadata.append((data["frame"], data["track"]))

    print(f"mean area {sum(areas) / len(areas)}")
    print(f"extreme   {min(areas), max(areas)}")
    print(f"too_small {too_small /(len(contours) + too_small)}")
    #import matplotlib.pyplot as plt
    #plt.hist(areas, bins=40)
    #plt.show()
    assert (len(contours) == len(centroids) == len(metadata))
    return contours, centroids, metadata

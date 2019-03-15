"""utility for loading contour points from the biotracker export."""
import json
import numpy as np
import cv2
import config


def load_biotracker_export(path: str = None, mask_path: str = None, threshold: float = None):
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

    for frame_id, line in enumerate(lines):
        for data in json.loads(line):

            X = np.array([d["x"] for d in data])
            Y = np.array([d["y"] for d in data])

            x_centroid = np.mean(X)
            y_centroid = np.mean(Y)

            outside_arena = mask_path and arena_mask[int(y_centroid), int(x_centroid)] == 0
            if outside_arena or len(X) < min_points:
                continue

            contours.append((X, Y))
            centroids.append(np.array([x_centroid, y_centroid]))
            # TODO: Get real track_index
            track_index = 0
            metadata.append((frame_id, track_index))

    assert (len(contours) == len(centroids) == len(metadata))
    return contours, centroids, metadata

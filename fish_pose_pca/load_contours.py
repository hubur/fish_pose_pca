"""utility for loading contour points from the biotracker export."""
import json

import numpy as np

import config


def load_biotracker_export(path: str = None, max_lines=None):
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
    with open(path) as data:
        lines = data.readlines()
    xy_arrays = []
    # The metadata list will include tuples (frame_id, fish_id)
    # The Position in the list maps to the position in the contours list
    metadata = []

    if max_lines:
        lines = lines[:max_lines]
    for line in lines:
        data = json.loads(line)

        X = np.array(data["xs"])
        Y = np.array(data["ys"])

        xy_arrays.append((X, Y))
        metadata.append((data["frame"], data["track"]))

    assert (len(xy_arrays) == len(metadata))
    return xy_arrays, metadata

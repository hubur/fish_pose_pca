"""utility for loading contour points from the biotracker export."""
import json
import numpy as np
import cv2
import config
import matplotlib.pyplot as plt
import scipy.signal as sig


def poly_area(x, y):
    """calculate polygon area using shoelace formula"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _get_middle_chunk(areas, threshold=0.2):
    bin_heights, bin_borders = np.histogram(areas, bins=30)
    peak_bins = sig.argrelmax(bin_heights, mode="wrap")[0]
    valley_bins = sig.argrelmin(bin_heights, mode="wrap")[0]

    peak_valley_map = np.zeros((len(bin_heights),))
    peak_valley_map[peak_bins] += 1
    peak_valley_map[valley_bins] -= 1
    valley_comes_first = peak_bins[0] < valley_bins[0]
    if valley_comes_first:
        peak_valley_map[0] = 0

    # List of triples: (left border bin index, peak bin index  right border bin index)
    big_bin_borders = np.zeros((len(peak_bins) + 1, 3), dtype=int)

    i = 0
    for j, val in enumerate(peak_valley_map):
        if val == -1:
            big_bin_borders[i, 2] = j
            i += 1
            big_bin_borders[i, 0] = j
        elif val == 1:
            big_bin_borders[i, 1] = j

    high_enough = threshold * bin_heights[np.argmax(bin_heights)]
    # print(high_enough_big_bin_borders[0])
    high_enough_big_bin_borders = np.array(
        [triple for triple in big_bin_borders if bin_heights[triple[1]] > high_enough])

    # This does not use the full potential of the Idea of this function
    # You could also eliminate noise between two extreme peaks, but then this does not give the "middle chunk" anymore
    low_thresh = bin_borders[high_enough_big_bin_borders[0, 0]]
    high_thresh = bin_borders[high_enough_big_bin_borders[-1, 2] + 1]
    return low_thresh, high_thresh


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


        x_centroid = data["center_x"]
        y_centroid = data["center_y"]

        outside_arena = mask_path and arena_mask[int(y_centroid), int(x_centroid)] == 0
        if outside_arena or poly_area(X, Y) < threshold:
            too_small += 1
            continue
        else:
            areas.append(poly_area(X, Y))

        contours.append((X, Y))
        centroids.append(np.array([x_centroid, y_centroid]))
        metadata.append((data["frame"], data["track"]))

    #print(f"mean area {sum(areas) / len(areas)}")
    #print(f"extrem    {min(areas), max(areas)}")
    #print(f"too_small {too_small /(len(contours) + too_small)}")
    low, high = _get_middle_chunk(areas)
    selector = [i for i, area in enumerate(areas) if low <= area <= high]

    contours, centroids, metadata = np.array(contours)[selector], np.array(centroids)[selector], np.array(metadata)[selector]
    assert (len(contours) == len(centroids) == len(metadata))
    return contours, centroids, metadata

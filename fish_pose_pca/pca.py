"""core stuff for performing the pca"""
import random
import math

from sklearn.decomposition import PCA
import numpy as np
import shapely.affinity
from shapely.geometry import Polygon
from scipy import interpolate

import load_contours
import config


class FishBlob:
    def __init__(self, xy_array):
        self.original_xy_array = xy_array
        self.polygon = Polygon(xy_array.T)
        self.index_of_point_furthest_from_centroid = None
        self.angle_of_point_furthest_from_centroid = None

    def rotate(self, angle_in_degrees):
        self.polygon = shapely.affinity.rotate(self.polygon, angle=angle_in_degrees, origin='center')

    def translate(self, x_offset, y_offset):
        self.polygon = shapely.affinity.translate(self.polygon, xoff=x_offset, yoff=y_offset)

    def area_too_small(self, area_threshold):
        return self.polygon.area < area_threshold

    def reduce_to_subfish(self, new_size):

        subfish_method = config.get_subfish_method()
        if subfish_method == config.SubFishMethod.RANDOM_SUBSET:
            contour = np.array(self.polygon.exterior.coords)
            selector = sorted(random.sample(range(0, self.polygon.length), new_size))
            self.polygon = Polygon(contour[selector])
        elif subfish_method == config.SubFishMethod.QUADRATIC_INTERPOLATE:
            contour = np.array(self.polygon.exterior.coords)

            # Linear length along the line:
            distance = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            # Interpolate
            interpolator = interpolate.interp1d(distance, contour, kind="quadratic", axis=0)
            interpolated_points = interpolator(np.linspace(0, 1, new_size))

            self.polygon = Polygon(interpolated_points)
        else:
            raise NotImplementedError(subfish_method)

    def normalize(self):
        # Normalize Contour Start
        contour_start = config.get_contour_start()
        if contour_start == config.ContourStart.MOST_DISTANT_FROM_CENTER_OF_MASS:
            contour = np.array(self.polygon.exterior.coords)[:-1]
            self.index_of_point_furthest_from_centroid = np.argmax(np.linalg.norm(contour, axis=1))
            self.polygon = Polygon(np.roll(contour, axis=0, shift=-self.index_of_point_furthest_from_centroid))
        else:
            raise NotImplementedError(contour_start)

        # Normalize Position
        translation_method = config.get_translation_method()
        if translation_method == config.TranslationMethod.CENTER_OF_MASS_ON_ZERO:
            self.translate(x_offset=-1 * self.polygon.centroid.coords[0][0],
                           y_offset=-1 * self.polygon.centroid.coords[0][1])
        elif translation_method == config.TranslationMethod.CENTER_OF_MASS_ON_X100:
            self.translate(x_offset=-1 * self.polygon.centroid.coords[0][0],
                           y_offset=-1 * self.polygon.centroid.coords[0][1])
            self.translate(x_offset=100, y_offset=0)
        else:
            raise NotImplementedError(translation_method)

        # Normalize Rotation
        rotation_method = config.get_rotation_method()
        if rotation_method == config.RotationMethod.MOST_DISTANT_POINT_AND_CENTER_ON_LINE:
            # if not rolling array around you need int(self.index_of_point_furthest_from_centroid) instead in 0
            x, y = self.polygon.exterior.coords[0]
            x, y = x - self.polygon.centroid.coords[0][0], y - self.polygon.centroid.coords[0][1]
            self.angle_of_point_furthest_from_centroid = math.degrees(math.atan2(y, x))
            self.rotate(angle_in_degrees=-self.angle_of_point_furthest_from_centroid)
        else:
            raise NotImplementedError(rotation_method)


def _get_contours(input_data):
    all_contours = []
    for line in input_data:
        contour = np.array([line[0], line[1]])
        all_contours.append(contour)
    return all_contours


def _get_fish_blobs(contours):
    all_blobs = []
    for contour in contours:
        fish = FishBlob(contour)
        all_blobs.append(fish)
    return np.array(all_blobs)


def _get_filtered_blobs(fish_blobs, metadata):
    area_too_small = config.get_area_too_small()
    subfish_size = config.get_subfish_size()
    filtered_blobs = []
    filtered_metadata = []
    for fish, meta in zip(fish_blobs, metadata):
        if fish.area_too_small(area_threshold=area_too_small):
            pass
        else:
            fish.reduce_to_subfish(new_size=subfish_size)
            fish.normalize()
            filtered_blobs.append(fish)
            filtered_metadata.append(meta)
    return np.array(filtered_blobs), np.array(filtered_metadata)


def _get_pca_input(fish_blobs):
    pca_input = np.array([np.concatenate(np.array(fish.polygon.exterior).T, axis=0) for fish in fish_blobs])
    return pca_input

def _from_pca_input_to_usable(multi_array_pca_input_style):
    half = multi_array_pca_input_style.shape[1] // 2
    A, B = multi_array_pca_input_style[:, :half], multi_array_pca_input_style[:, half:]
    A, B = A.reshape((A.shape[0], 1, A.shape[1])), B.reshape((B.shape[0], 1, B.shape[1]))
    return np.concatenate((A, B), axis=1)


def _from_usable_to_pca_input(multi_array_usable_style):
    return np.concatenate((multi_array_usable_style[:, 0, :], multi_array_usable_style[:, 1, :]), axis=1)


def _from_xy_usable_to_polar_pca_input(xy_multi_array_usable_style):
    lens = np.linalg.norm((xy_multi_array_usable_style), axis=1)
    angles = np.arctan2(xy_multi_array_usable_style[:, 1, :], xy_multi_array_usable_style[:, 0, :])
    polar_pca_input_style = np.concatenate((angles, lens), axis=1)
    return polar_pca_input_style


def _from_polar_usable_to_xy_pca_input(polar_multi_array_usable_style):
    angles, lens = polar_multi_array_usable_style[:, 0, :], polar_multi_array_usable_style[:, 1, :]
    X = np.multiply(np.cos(angles), lens)
    Y = np.multiply(np.sin(angles), lens)
    xy_pca_input_style = np.concatenate((X, Y), axis=1)
    return xy_pca_input_style

def do_pca(n_components: int):
    # Load Data
    data_path = config.get_contour_data_path()
    raw_data, raw_metadata = load_contours.load_biotracker_export(path=data_path)

    # Process raw data into [[x,y][x,y]...] format
    contours = _get_contours(raw_data)
    # Create FishBlob objects
    raw_fish_blobs = _get_fish_blobs(contours)
    # Try to filter out noise
    filtered_blobs, filtered_metadata = _get_filtered_blobs(raw_fish_blobs, raw_metadata)
    # Get pca_input format [x,x,x,x... y,y,y,y....y]
    pca_input = _get_pca_input(filtered_blobs)

    coordinate_system = config.get_coordinate_system()
    if coordinate_system == config.CoordinateSystem.POLAR_SYSTEM:
        pca_input = _from_xy_usable_to_polar_pca_input(_from_pca_input_to_usable(pca_input))

    # Do the PCA
    pca_object = PCA(n_components=n_components)
    transformed_fishes = pca_object.fit_transform(X=pca_input)

    return contours, filtered_blobs, filtered_metadata, pca_object, pca_input, transformed_fishes

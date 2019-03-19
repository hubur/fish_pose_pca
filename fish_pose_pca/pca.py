from sklearn.decomposition import PCA
import load_contours
import numpy as np
import fishy
import config


def _get_normalized_fish_blobs_from_contours_and_centroids(contours, centroids):
    all_blobs = []
    for contour, centroid in zip(contours, centroids):
        fish = fishy.FishBlob(contour[0], contour[1], centroid)
        fish.normalize()
        all_blobs.append(fish)
    return np.array(all_blobs)

def _get_pca_input_from_normalized_fish_blobs(normal_fish_blobs):
    # Grab subfish_size Points at random out of the fishes (in-place)
    subfish_size = config.get_subfish_size()
    [fish_blob.reduce_to_subfish(subfish_size) for fish_blob in normal_fish_blobs]

    pca_input = [[*fish_blob.X, *fish_blob.Y] for fish_blob in normal_fish_blobs]
    pca_input = np.array(pca_input)

    return np.array(pca_input)


def _from_pca_input_to_usable(multi_array_pca_input_style):
    half = multi_array_pca_input_style.shape[1] // 2
    A, B = multi_array_pca_input_style[:, :half], multi_array_pca_input_style[:, half:]
    A, B = A.reshape((A.shape[0], 1, A.shape[1])), B.reshape((B.shape[0], 1, B.shape[1]))
    return np.concatenate((A, B), axis=1)


def _from_usable_to_pca_input(multi_array_usable_style):
    print(f"shape {multi_array_usable_style.shape}")
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
    mask_path = config.get_mask_path()
    contours, centroids, metadata = load_contours.load_biotracker_export(path=data_path, mask_path=mask_path)

    # Create FishBlob objects and Normalize Data
    fish_blobs = _get_normalized_fish_blobs_from_contours_and_centroids(contours=contours, centroids=centroids)
    pca_input = _get_pca_input_from_normalized_fish_blobs(fish_blobs)

    coordinate_system = config.get_coordinate_system()
    if coordinate_system == config.CoordinateSystem.POLAR_SYSTEM:
        pca_input = _from_xy_usable_to_polar_pca_input(_from_pca_input_to_usable(pca_input))

    # Do the PCA
    fish_pca = PCA(n_components=n_components)
    transformed_fishes = fish_pca.fit_transform(X=pca_input)

    return fish_pca, transformed_fishes, pca_input, metadata

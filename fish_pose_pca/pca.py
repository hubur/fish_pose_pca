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
    return all_blobs

def _get_pca_input_from_normalized_fish_blobs(normal_fish_blobs):
    # Grab subfish_size Points at random out of the fishes (in-place)
    subfish_size = config.get_subfish_size()
    [fish_blob.reduce_to_subfish(subfish_size) for fish_blob in normal_fish_blobs]

    pca_input = [[*fish_blob.X, *fish_blob.Y] for fish_blob in normal_fish_blobs]
    pca_input = np.array(pca_input)

    return pca_input


def do_pca(n_components: int):
    # Load Data
    data_path = config.get_contour_data_path()
    mask_path = config.get_mask_path()
    contours, centroids, metadata = load_contours.load_biotracker_export(path=data_path, mask_path=mask_path)

    # Create FishBlob objects and Normalize Data
    fish_blobs = _get_normalized_fish_blobs_from_contours_and_centroids(contours=contours, centroids=centroids)
    pca_input = _get_pca_input_from_normalized_fish_blobs(fish_blobs)

    # Do the PCA
    fish_pca = PCA(n_components=n_components)
    transformed_fishes = fish_pca.fit_transform(X=pca_input)

    return fish_pca, transformed_fishes, pca_input, metadata

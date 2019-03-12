from sklearn.decomposition import PCA
import numpy as np


def get_pca_input_from_normalized_fish_blobs(normal_fish_blobs, minimum_blob_size, subfish_size):
    # Grab only fishes that have an outline with at least minimum_blob_size points
    only_big_blobs = np.array([fish_blob for fish_blob in normal_fish_blobs if len(fish_blob.X) >= minimum_blob_size])

    # Grab subfish_size Points at random out of the fishes (in-place)
    [fish_blob.reduce_to_subfish(subfish_size) for fish_blob in only_big_blobs]

    pca_input = [[*fish_blob.X, *fish_blob.Y] for fish_blob in only_big_blobs]
    pca_input = np.array(pca_input)

    return pca_input

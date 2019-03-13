from sklearn.decomposition import PCA
import numpy as np


def get_pca_input_from_normalized_fish_blobs(normal_fish_blobs, subfish_size):
    # Grab subfish_size Points at random out of the fishes (in-place)
    [fish_blob.reduce_to_subfish(subfish_size) for fish_blob in normal_fish_blobs]

    pca_input = [[*fish_blob.X, *fish_blob.Y] for fish_blob in normal_fish_blobs]
    pca_input = np.array(pca_input)

    return pca_input

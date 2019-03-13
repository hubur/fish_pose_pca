from sklearn.decomposition import PCA
import numpy as np
import fishy
import config


def _get_pca_input_from_normalized_fish_blobs(normal_fish_blobs):
    # Grab subfish_size Points at random out of the fishes (in-place)
    subfish_size = config.get_subfish_size()
    [fish_blob.reduce_to_subfish(subfish_size) for fish_blob in normal_fish_blobs]

    pca_input = [[*fish_blob.X, *fish_blob.Y] for fish_blob in normal_fish_blobs]
    pca_input = np.array(pca_input)

    return pca_input


def get_pca_input():
    data_path = config.get_contour_data_path()
    mask_path = config.get_mask_path()
    blobs = fishy.get_normalized_fish_blobs_from_data(data_path=data_path, mask_path=mask_path)
    return _get_pca_input_from_normalized_fish_blobs(normal_fish_blobs=blobs)


def transform_fishes(n_components: int):
    fish_pca = PCA(n_components=n_components)
    return fish_pca.fit_transform(X=get_pca_input()), fish_pca

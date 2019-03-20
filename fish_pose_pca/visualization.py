"""utilities for visualizing data"""
import copy
import statistics

from pathlib import Path
from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np

import config

def save_video_frames(parameters: List, fun: Callable, out_folder: str):
    """calls fun for each item in parameters and saves the
    return value of fun in a png."""
    out_path = Path(out_folder)
    if not out_path.exists():
        out_path.mkdir()
    for i, parameter in enumerate(parameters):
        img = fun(parameter)
        plt.imshow(img, cmap="gray")
        frame_name = f"{i}.png"
        plt.savefig(str(out_path / frame_name))


def _plot_fish(axis, xy_array):
    half = xy_array.shape[0] // 2
    X, Y = xy_array[:half], xy_array[half:]
    # Adding the first point again at the back to close the loop
    X, Y = np.concatenate((X, X[0:1])), np.concatenate((Y, Y[0:1]))
    # "k-" means solid black line
    axis.plot([-40, 40], [0, 0], 'k--', alpha=0.5)
    axis.plot([0, 0], [-40, 40], 'k--', alpha=0.5)
    axis.plot(X, Y, 'k-')
    axis.plot(X[1:-1], Y[1:-1], 'rd')
    axis.plot(X[0], Y[0], 'gD')


def _print_reconstruction_compare(pca_object, pca_input, transformed_fishes, fish_index):
    n_components = pca_object.n_components
    figsize = [9, 18]
    coordinate_system = config.get_coordinate_system()
    if coordinate_system == config.CoordinateSystem.CARTESIAN_SYSTEM:
        dct = None
    elif coordinate_system == config.CoordinateSystem.POLAR_SYSTEM:
        dct = dict(polar=True)
    else:
        raise NotImplementedError(coordinate_system)

    fig, axes = plt.subplots(ncols=2, nrows=1, sharey=True, subplot_kw=dct)

    fig.set_figheight(val=figsize[0])
    fig.set_figwidth(val=figsize[1])
    if coordinate_system == config.CoordinateSystem.CARTESIAN_SYSTEM:
        axes[0].set_xlim([-75, 75])
        axes[0].set_ylim([-75, 75])
        axes[1].set_xlim([-75, 75])
        axes[1].set_ylim([-75, 75])

    old, new = pca_input[fish_index], pca_object.inverse_transform(transformed_fishes[fish_index])
    _plot_fish(axes[0], old)
    _plot_fish(axes[1], new)

    axes[0].set_title(f"original {transformed_fishes[fish_index]}")
    axes[1].set_title(f"reconstructed from {n_components} Principal Components")

    plt.show()


def print_fishes_by_index(pca_object, pca_input, transformed_fishes, index_list):
    for i in index_list:
        _print_reconstruction_compare(pca_object, pca_input, transformed_fishes, i)

def print_min_max_fishes(pca_object, pca_input, transformed_fishes, n_fishes, principal_component_index):
    sorted_fishes = transformed_fishes[:, principal_component_index].argsort()
    print(f"Data distributution along PC #{principal_component_index}")
    print(len(transformed_fishes))
    plt.hist(transformed_fishes[:,principal_component_index],bins=len(transformed_fishes)//30)
    plt.show()
    min_fishes = sorted_fishes[:n_fishes]
    max_fishes = sorted_fishes[-n_fishes:][::-1]

    print(f"The {n_fishes} Fishes with the MINIMUM Values in Principal Component #{principal_component_index}")
    for n in min_fishes:
        print(f"PC #{principal_component_index}:{transformed_fishes[n,principal_component_index]}")
        _print_reconstruction_compare(pca_object, pca_input, transformed_fishes, n)

    print(f"The {n_fishes} Fishes with the MAXIMUM Values in Principal Component #{principal_component_index}")
    for n in max_fishes:
        print(f"PC #{principal_component_index}:{transformed_fishes[n,principal_component_index]}")
        _print_reconstruction_compare(pca_object, pca_input, transformed_fishes, n)


def print_variance_plot(pca_object):
    n_components = pca_object.n_components
    plt.bar(range(1, n_components + 1), pca_object.explained_variance_ratio_)
    plt.title(f"Explained Variance by the {n_components} Principal Components")
    text = [f"PC #{i}" for i in range(1, n_components + 1)]
    plt.xticks(np.arange(n_components) + 1, text)
    plt.show()


def print_pc_wiggle_effects(pca_object, transformed_fishes):
    n_components = pca_object.n_components
    figsize = [3.7 * n_components, 18]
    mean_fish = np.zeros((n_components,))

    coordinate_system = config.get_coordinate_system()
    if coordinate_system == config.CoordinateSystem.CARTESIAN_SYSTEM:
        dct = None
    elif coordinate_system == config.CoordinateSystem.POLAR_SYSTEM:
        dct = dict(polar=True)
    else:
        raise NotImplementedError(coordinate_system)

    fig, axes = plt.subplots(ncols=5, nrows=n_components, subplot_kw=dct)
    fig.set_figheight(val=figsize[0])
    fig.set_figwidth(val=figsize[1])
    fig.subplots_adjust(wspace=0, hspace=0)
    topText = ["-2 Std.Dev.", "-1 Std.Dev.", "Mean Fish", "+1 Std.Dev.", "+2 Std.Dev."]
    for i, _ in enumerate(axes):
        stdev_for_this_pc = statistics.stdev(transformed_fishes[:, i])
        factors = np.array([-2,-1,0,1,2])*stdev_for_this_pc
        for j, _ in enumerate(axes[i]):

            modified_mean_fish = copy.copy(mean_fish)
            modified_mean_fish[i] = factors[j]
            mod = pca_object.inverse_transform(modified_mean_fish)

            if coordinate_system == config.CoordinateSystem.CARTESIAN_SYSTEM:
                axes[i, j].set_xlim([-50, 50])
                axes[i, j].set_ylim([-50, 50])

            # "k-" means solid black line
            _plot_fish(axes[i, j], mod)
            if i == 0:
                axes[0, j].set_title(topText[j])

            if j == 0:
                axes[i, 0].set_ylabel(f"PC #{i+1}")
                axes[i, 0].get_yaxis().set_ticks([])
            else:
                pass
                axes[i, j].get_yaxis().set_visible(False)

            axes[i, j].get_xaxis().set_visible(False)

    plt.show()

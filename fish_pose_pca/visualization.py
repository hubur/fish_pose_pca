"""utilities for visualizing data"""
from pathlib import Path
from typing import Callable, List
import matplotlib.pyplot as plt


def save_video_frames(parameters: List, fun: Callable, out_folder: str):
    """calls fun for each item in parameters and saves the
    return value of fun in a png."""
    out_path = Path(out_folder)
    if not out_path.exists():
        out_path.mkdir()
    for i, parameter in enumerate(parameters):
        img = fun(parameter)
        plt.imshow(img)
        frame_name = f"{i}.png"
        plt.savefig(str(out_path / frame_name))

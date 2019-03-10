"""utility for loading contour points from the biotracker export."""
import json
from typing import Dict, List


class ContourPoint:
    def __init__(self, data: Dict):
        self.x = data["x"]
        self.y = data["y"]


class Contour:
    def __init__(self, data: List):
        self.points = [ContourPoint(d) for d in data]

    def min_x(self):
        return min([p.x for p in self.points])

    def min_y(self):
        return min([p.y for p in self.points])

    def max_x(self):
        return max([p.x for p in self.points])

    def max_y(self):
        return max([p.y for p in self.points])


class Frame:
    def __init__(self, line: str):
        self.contours = [Contour(c) for c in json.loads(line)]


def load_contours(path: str) -> [Frame]:
    with open(path) as data:
        lines = data.readlines()
    return [Frame(line) for line in lines]

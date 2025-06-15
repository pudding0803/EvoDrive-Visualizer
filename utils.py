from __future__ import annotations

import math
import random

from shapely.affinity import scale
from shapely.geometry import LineString, Polygon

from config import Config
from track import Track


class Utils:
    @staticmethod
    def generate_simple_track(track_range: float = 200) -> Track:
        scale_factor = (0.75 / math.sqrt(20)) * math.sqrt(track_range / (Config.car_radius * 2))
        points = [(random.uniform(0, track_range), random.uniform(0, track_range)) for _ in range(50)]
        outer_poly = Polygon(points).convex_hull
        inner_poly = scale(outer_poly, xfact=scale_factor, yfact=scale_factor, origin='center')

        while True:
            lines = []
            for i in range(len(outer_poly.exterior.coords) - 1):
                o1, o2 = outer_poly.exterior.coords[i], outer_poly.exterior.coords[i + 1]
                i1, i2 = inner_poly.exterior.coords[i], inner_poly.exterior.coords[i + 1]
                o_mid = ((o1[0] + o2[0]) / 2, (o1[1] + o2[1]) / 2)
                i_mid = ((i1[0] + i2[0]) / 2, (i1[1] + i2[1]) / 2)
                lines.append(LineString([o_mid, i_mid]))
            if all(line.length > 3 * Config.car_radius for line in lines):
                break

        return Track(
            polygon=Polygon(shell=outer_poly.exterior.coords, holes=[inner_poly.exterior.coords]),
            finish_line=random.choice(lines),
            start_from_right=True
        )

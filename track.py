import random

import networkx as nx
import numpy as np
from attr import dataclass
from scipy.spatial import KDTree
from shapely.geometry.base import BaseGeometry
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import triangulate

from config import Config


@dataclass
class Position:
    point: Point
    distance: float


class Track:
    def __init__(self, polygon: Polygon, finish_line: LineString, start_from_right: bool):
        self.polygon = polygon
        self.buffered_polygon = polygon.buffer(-Config.car_radius)
        self.finish_line = finish_line
        (x1, y1), (x2, y2) = finish_line.coords[:2]
        self.start_vector = (y1 - y2, x2 - x1)
        if (self.start_vector[0] * start_from_right < 0
                or self.start_vector[0] == 0 and self.start_vector[1] * start_from_right < 0
        ):
            self.start_vector = tuple(-v for v in self.start_vector)
        self.min_x, self.min_y, self.max_x, self.max_y = polygon.bounds
        self.width, self.height = self.max_x - self.min_x, self.max_y - self.min_y
        self.diag_len = np.sqrt(self.width ** 2 + self.height ** 2)
        self.approx_path: LineString
        self.checkpoints: list[Position] = []
        self._generate_approx_path()

    def on_the_edge(self, point: Point) -> bool:
        return self.polygon.boundary.distance(point) < 1e-6

    def intersect_point(self, point: Point, theta: float) -> Position:
        ray_point = (
            point.x + np.cos(theta) * self.diag_len,
            point.y + np.sin(theta) * self.diag_len
        )
        ray = LineString([point, ray_point])
        intersection = self.polygon.intersection(ray)
        nearest, min_dist2 = None, np.inf
        points = list(filter(self.on_the_edge, Track._extract_geometry_to_points(intersection)))
        for pt in points:
            d2 = (pt.x - point.x) ** 2 + (pt.y - point.y) ** 2
            if d2 < min_dist2:
                nearest, min_dist2 = pt, d2
        if nearest is None:
            raise ValueError(f'Cannot find the intersection point from point ({point.x}, {point.y}), θ = {theta}')
        return Position(point=nearest, distance=np.sqrt(min_dist2))

    def _generate_approx_path(self):
        print('Generating an approximate path of the track. It may take a while.')
        while True:
            try:
                triangle: Polygon = random.choice(triangulate(self.buffered_polygon))
                points = self._poisson_disk_sampling(tuple(triangle.centroid.coords[0]), 10)
                graph = nx.Graph()
                finish = self.finish_line.centroid.coords[0]
                v = np.array(self.start_vector) / np.linalg.norm(self.start_vector)
                start = (float(finish[0] + v[0]), float(finish[1] + v[1]))
                end = (float(finish[0] - v[0]), float(finish[1] - v[1]))
                vertices = [start] + list(self.buffered_polygon.exterior.coords) + points + [end]
                for i, v1 in enumerate(vertices):
                    for j, v2 in enumerate(vertices):
                        if i >= j:
                            continue
                        line = LineString([v1, v2])
                        if v1 == v2:
                            continue
                        elif v1 == start or v2 == end:
                            u, v = np.array([v2[0] - v1[0], v2[1] - v1[1]]), np.array(self.start_vector)
                            norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
                            if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
                                continue
                            if np.dot(u, v) / (norm_u * norm_v) <= 0:
                                continue
                        elif line.crosses(self.finish_line):
                            continue
                        if self.buffered_polygon.contains(line):
                            graph.add_edge(v1, v2, weight=line.length)
                path = nx.shortest_path(graph, source=start, target=end, weight='weight')
                break
            except nx.exception.NetworkXNoPath:
                print('Failed to generate an approximate path. Retrying...')
        print('The approximate path was generated successfully.\n')
        self.approx_path = LineString(path)
        path_length = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        checkpoint_distances = [
            path_length * ratio
            for ratio in np.linspace(0, 1, Config.checkpoint_number + 2)[1:-1]
        ]
        current_length = 0
        checkpoint = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_length = graph[u][v]['weight']
            while (checkpoint < Config.checkpoint_number
                   and current_length + edge_length >= checkpoint_distances[checkpoint]
            ):
                u, v = np.array(u), np.array(v)
                prop = (checkpoint_distances[checkpoint] - current_length) / edge_length
                self.checkpoints.append(
                    Position(point=Point(u + prop * (v - u)), distance=checkpoint_distances[checkpoint])
                )
                checkpoint += 1
            current_length += edge_length
            if checkpoint == Config.checkpoint_number:
                break

    # def approx_distance_to_path(self, point: Point) -> float:
    #     dist_to_start = self.approx_path.project(point)
    #     proj = self.approx_path.interpolate(dist_to_start)
    #     return point.distance(proj)

    def approx_distance_from_start(self, point: Point) -> float:
        return self.approx_path.project(point)

    def _poisson_disk_sampling(self, init_point: tuple[float, float], radius: float) -> list[tuple[float, float]]:
        active_points = [init_point]
        points = [init_point]
        tree = KDTree(points)
        while len(active_points) > 0:
            point = random.choice(active_points)
            found = False
            for _ in range(30):
                r = np.random.uniform(radius, 2 * radius)
                new_point = (
                    point[0] + r * np.cos(r),
                    point[1] + r * np.sin(r)
                )
                if self.buffered_polygon.contains(Point(new_point)) and not tree.query_ball_point(new_point, radius):
                    points.append(new_point)
                    active_points.append(new_point)
                    tree = KDTree(points)
                    found = True
                    break
            if not found:
                active_points.remove(point)
        return points

    @staticmethod
    def _extract_geometry_to_points(geometry: BaseGeometry) -> list[Point]:
        if geometry.is_empty:
            return []
        if isinstance(geometry, Point):
            return [geometry]
        elif isinstance(geometry, LineString):
            return [Point(coord) for coord in geometry.coords]
        elif isinstance(geometry, (MultiPoint, MultiLineString, GeometryCollection)):
            return [Point(coord) for geom in geometry.geoms for coord in Track._extract_geometry_to_points(geom)]
        else:
            raise Exception(f'Undefined intersection type: {geometry}')

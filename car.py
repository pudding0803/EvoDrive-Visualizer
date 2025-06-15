from __future__ import annotations

from collections import deque
from enum import auto, Enum
from typing import TYPE_CHECKING

import h5py
import numpy as np
from shapely.geometry.point import Point

from config import Config

if TYPE_CHECKING:
    from mlp import MLP
    from track import Position, Track


class CarStatus(Enum):
    RUNNING = auto()
    CRASHED = auto()
    LOOPING = auto()
    COMPLETED = auto()

    def get_color(self):
        match self:
            case CarStatus.RUNNING:
                return 'steelblue'
            case CarStatus.CRASHED:
                return 'red'
            case CarStatus.LOOPING:
                return 'gray'
            case CarStatus.COMPLETED:
                return 'lime'
            case _:
                raise ValueError('Undefined CarStatus')


class Car:
    def __init__(self, track: Track, network: MLP):
        self.track = track
        self.network = network
        self.point: Point = self.track.finish_line.centroid
        self.theta: float = np.atan2(*reversed(track.start_vector))
        self.recent_theta = deque(maxlen=5)
        self.sensor_data: list[Position] = []
        self.fitness: float = 0
        self.distance: float = 0
        self.checkpoint: int = 0
        self.status: CarStatus = CarStatus.RUNNING

    def move(self):
        if self.status != CarStatus.RUNNING:
            return
        self.distance = self.track.approx_distance_from_start(self.point)
        if self.point.distance(self.track.polygon.boundary) <= Config.car_radius:
            self.status = CarStatus.CRASHED
            self.fitness = self.distance / self.track.approx_path.length * 100
        else:
            self.recent_theta.append(self.theta)
            if len(self.recent_theta) == 5 and np.abs(np.mean(self.recent_theta)) > 30:
                self.status = CarStatus.LOOPING
                self.fitness = 0
            elif (self.checkpoint == Config.checkpoint_number
                  and self.point.distance(self.track.finish_line) <= Config.car_radius
            ):
                self.status = CarStatus.COMPLETED
                self.fitness = 100
                if Config.weights_export_prefix != '':
                    self._save_new_completed_weights()
            else:
                if self.checkpoint < Config.checkpoint_number and self.distance > self.track.checkpoints[self.checkpoint].distance:
                    self.checkpoint += 1
                elif self.checkpoint > 0 and self.distance < self.track.checkpoints[self.checkpoint - 1].distance:
                    self.checkpoint -= 1

                self.sensor_data = [
                    self.track.intersect_point(self.point, self.theta + np.pi / 2),
                    self.track.intersect_point(self.point, self.theta + np.pi / 4),
                    self.track.intersect_point(self.point, self.theta),
                    self.track.intersect_point(self.point, self.theta - np.pi / 4),
                    self.track.intersect_point(self.point, self.theta + np.pi / 2)
                ]
                output = self.network.forward(self.features())
                self.theta += (output[0] * 2 - 1) * Config.max_steering_angle
                self.point = Point(
                    self.point.x + Config.car_velocity * np.cos(self.theta),
                    self.point.y + Config.car_velocity * np.sin(self.theta)
                )

    def features(self) -> np.ndarray:
        return np.array([data.distance for data in self.sensor_data])

    def _save_new_completed_weights(self):
        with h5py.File(f'{Config.weights_export_prefix}.h5', 'a') as file:
            group_name = f'weights_{len(file.keys())}'
            weight_group = file.create_group(group_name)
            for i, array in enumerate(self.network.weights):
                weight_group.create_dataset(f'layer_{i}', data=array)

import numpy as np
from selection import SelectionStrategy


class Config:
    _instance = None

    hidden_layer_sizes: tuple[int, ...] = (12, 8)
    detail_visualization: bool = False
    checkpoint_number: int = 3
    population: int = 300
    car_radius: float = 5
    car_velocity: float = 3
    car_heading_line_len: float = 8
    max_steering_angle: float = np.pi / 18
    selection = SelectionStrategy('linear_ranking')
    eliminate_percentile: float = 0
    track_change_interval: int = 10
    completion_criteria: float = 0.5
    max_frames: int = 100000
    frame_interval: int = 50
    weights_export_prefix: str = ''

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            for key, value in kwargs.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
        return cls._instance

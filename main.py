import matplotlib
from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import Polygon

from model import Model
from track import Track


def main():
    matplotlib.use('TkAgg')

    # Predefined track
    track = Track(
        Polygon(
            shell=[
                (110, 170), (180, 170), (210, 140), (210, 40), (150, 10), (120, 20), (80, 10), (60, 30),
                (60, 70), (30, 70), (30, 0), (-50, 0), (-65, 30), (-50, 60), (-65, 90), (-50, 120), (-65, 150),
                (-50, 180), (45, 160)
            ],
            holes=[
                [(0, 30), (0, 100), (90, 100), (90, 40), (180, 60), (180, 130), (170, 140),
                 (115, 145), (45, 130), (-30, 150), (-15, 120), (-30, 90), (-15, 60), (-30, 30)]
            ]
        ),
        finish_line=LineString([(0, 40), (30, 40)]),
        start_from_right=True
    )
    Model(track, weights_exports_prefix='elaborate', eliminate_percentile=50)

    # Simple Track Generator by default
    # Model()


if __name__ == '__main__':
    main()

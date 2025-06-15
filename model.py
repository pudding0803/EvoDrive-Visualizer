from __future__ import annotations

import concurrent.futures
from itertools import chain
from statistics import mean
from typing import TYPE_CHECKING

import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrow
from matplotlib.ticker import MaxNLocator

from car import Car, CarStatus
from config import Config
from mlp import MLP
from utils import Utils

if TYPE_CHECKING:
    from track import Track


class Model:
    def __init__(self, track: Track = None, pretrained_file: str = None, **kwargs):
        Config(**kwargs)
        self.track = track if track is not None else Utils.generate_simple_track()

        self.cars: list[Car] = []
        if pretrained_file:
            Config.weights_export_prefix = ''
            self._load_all_completed_weights(pretrained_file)
        else:
            self.cars = [Car(self.track, MLP()) for _ in range(Config.population)]

        self.generation = 1
        self.completed_cars = 0
        self.stop = False
        self.selection = Config.selection

        # Declaration of plot
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=1)

        self.ax_animation = fig.add_subplot(gs[:4, 0])
        self.ax_animation.set_title('Track Visualization')
        self.ax_animation.set_aspect('equal')

        self.ax_fitness = fig.add_subplot(gs[4:, 0])
        self.ax_fitness.set_title('Average Fitness')
        self.ax_fitness.set_xlabel('Generation')
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.set_xticks([1])
        self.ax_fitness.set_xlim(0.8, 2.5)
        self.ax_fitness.set_ylim(0, 1)

        self.ax_data = fig.add_subplot(gs[:2, 1])
        self.ax_data.axis('off')

        self.ax_completion = fig.add_subplot(gs[2:, 1])
        self.ax_completion.set_title('Completion Rates of the Last Generation')
        self.ax_completion.set_xlabel('Completion Rate (%)')
        self.ax_completion.set_ylabel('Number of Cars')
        self.ax_completion.set_ylim(0, Config.population * 1.2)

        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        fig.canvas.manager.set_window_title('EvoDrive Visualizer')

        self.track_artists = []
        self.track_future: concurrent.futures.Future
        self.track_executor = concurrent.futures.ThreadPoolExecutor()
        self._plot_track()
        if Config.track_change_interval > 0:
            self._generate_next_track()

        self.car_circles = []
        self.heading_lines = []
        for car in self.cars:
            car_circle = Circle(car.point.coords[0], Config.car_radius, zorder=2)
            self.car_circles.append(car_circle)
            self.ax_animation.add_patch(car_circle)
            self.heading_lines.append(self.ax_animation.plot([], [], linewidth=3, zorder=2)[0])

        self.fitness_averages = []
        self.top_car_fitness = 0
        self.ave_fitness_line = self.ax_fitness.plot(self.fitness_averages, color='gold', marker='o')[0]
        self.max_fitness_line = self.ax_fitness.axhline(y=-1, color='orange', linestyle=':')

        texts = {
            'frame': 'Current Frame: 0',
            'generation': 'Current Generation: 1',
            'completed': f'Completed Cars: 0 / {Config.population}',
            'max_ave_fitness': 'Maximum Average Fitness: None'
        }
        text_interval = 0.25
        self.data_texts = {
            key: self.ax_data.text(0, 1 - i * text_interval, content, ha='left', va='top')
            for i, (key, content) in enumerate(texts.items())
        }

        self.distance_labels = [f'{i}+' for i in range(0, 100, 10)]
        self.distance_bars = {
            CarStatus.CRASHED: self.ax_completion.bar(
                self.distance_labels, [0] * 10, color=['red'] * 10, label='Crashed'
            ),
            CarStatus.LOOPING: self.ax_completion.bar(
                self.distance_labels, [0] * 10, color=['gray'] * 10, label='Looping'
            ),
            CarStatus.COMPLETED: self.ax_completion.bar(
                self.distance_labels, [0] * 10, color=['lime'] * 10, label='Completed'
            )
        }
        self.ax_completion.legend(loc='upper center', ncol=3)

        self.animation = FuncAnimation(
            fig, self._animate, frames=Config().max_frames, interval=Config.frame_interval, blit=True, repeat=False
        )
        plt.show()
        self.track_executor.shutdown()

    def _animate(self, frame: int) -> list[Artist]:
        if frame > 0:
            all_stop = True
            self.data_texts['frame'].set_text(f'Current Frame: {frame + 1} / {Config.max_frames}')
            for i, car in enumerate(self.cars):
                if car.status == CarStatus.RUNNING:
                    car.move()
                    self.car_circles[i].set_center(car.point.coords[0])
                    self.heading_lines[i].set_data(
                        [car.point.x, car.point.x + Config.car_heading_line_len * np.cos(car.theta)],
                        [car.point.y, car.point.y + Config.car_heading_line_len * np.sin(car.theta)]
                    )
                    all_stop = False
                self.car_circles[i].set_color(car.status.get_color())
                self.heading_lines[i].set_color(car.status.get_color())

            if self.stop:
                self._next_generation()
                if self.completed_cars >= Config.population * Config.completion_criteria:
                    self.animation.event_source.stop()
                    return []
                min_fitness, max_fitness = min(self.fitness_averages), max(self.fitness_averages)
                margin = (max_fitness - min_fitness) * 0.2
                self.ax_fitness.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
                self.ax_fitness.set_xlim(0.8, len(self.fitness_averages) + 1.5)
                if margin == 0:
                    margin = 1
                    self.ave_fitness_line.set_data([1, 1], self.fitness_averages * 2)
                else:
                    self.ave_fitness_line.set_data(range(1, len(self.fitness_averages) + 1), self.fitness_averages)
                self.ax_fitness.set_ylim(min_fitness - margin, max_fitness + margin)
                self.max_fitness_line.set_ydata([max_fitness])
                self.ax_fitness.figure.canvas.draw()
                self.data_texts['generation'].set_text(f'Current Generation: {self.generation}')
                self.data_texts['completed'].set_text(f'Completed Cars: {self.completed_cars} / {Config.population}')
                self.data_texts['max_ave_fitness'].set_text(f'Maximum Average Fitness: {round(max(self.fitness_averages), 3)}')
                self.stop = False
            elif all_stop:
                self.stop = True
        return [
            *self.car_circles, *self.heading_lines, self.ave_fitness_line,
            *self.data_texts.values(), *chain(*self.distance_bars.values())
        ]

    def _next_generation(self):
        self.generation += 1
        next_track = Config.track_change_interval > 0 and self.generation % Config.track_change_interval == 0

        self.fitness_averages.append(mean(car.fitness for car in self.cars))
        self.top_car_fitness = max(car.fitness for car in self.cars)

        bins = np.linspace(0, self.track.approx_path.length, 11)
        distances = {
            CarStatus.CRASHED: [],
            CarStatus.LOOPING: [],
            CarStatus.COMPLETED: []
        }
        for car in self.cars:
            distances[car.status].append(car.distance)
        histograms = {status: np.histogram(distances, bins=bins)[0] for status, distances in distances.items()}
        for i in range(10):
            bottom = 0
            for status, bars in self.distance_bars.items():
                bars[i].set_height(histograms[status][i])
                bars[i].set_y(bottom)
                bottom += histograms[status][i]

        if Config.eliminate_percentile:
            threshold = np.percentile([car.fitness for car in self.cars], Config.eliminate_percentile)
            for car in self.cars:
                if car.fitness < threshold:
                    car.fitness = 0

        if next_track:
            self.track = self.track_future.result()
            self._generate_next_track()
            offsprings = [Car(self.track, car.network) for car in self.cars if car.status == CarStatus.COMPLETED]
            self.completed_cars = 0
        else:
            offsprings = [car for car in self.cars if car.status == CarStatus.COMPLETED]
            for i in range(self.completed_cars, len(offsprings)):
                self.car_circles[i].set_visible(False)
                self.heading_lines[i].set_visible(False)
            self.completed_cars = len(offsprings)

        while len(offsprings) < Config.population:
            parents = self.selection(self.cars)
            new_weights = [[], []]
            for w1, w2, mutation_ranges in zip(parents[0].network.weights, parents[1].network.weights, MLP.generate_weights()):
                crossover_mask = np.random.rand(*w1.shape) > 0.5
                new_w1 = np.where(crossover_mask, w1, w2)
                new_w2 = np.where(crossover_mask, w2, w1)
                mutation_mask1 = np.random.rand(*new_w1.shape) < 0.2
                mutation_mask2 = np.random.rand(*new_w2.shape) < 0.2
                new_w1 += mutation_mask1 * mutation_ranges
                new_w2 += mutation_mask2 * mutation_ranges
                new_weights[0].append(new_w1)
                new_weights[1].append(new_w2)
            offsprings.extend([
                Car(self.track, MLP(weights=new_weights[0])),
                Car(self.track, MLP(weights=new_weights[1]))
            ])
        if len(offsprings) > Config.population:
            offsprings = offsprings[:Config.population]
        self.cars = offsprings

        if next_track:
            self._plot_track()

    def _generate_next_track(self):
        self.track_future = self.track_executor.submit(Utils.generate_simple_track)

    def _plot_track(self):
        for artist in self.track_artists:
            artist.remove()
        self.track_artists = []

        self.track_artists.append(
            self.ax_animation.plot(*self.track.finish_line.xy, color='red', linewidth=2, zorder=1)[0]
        )
        self.track_artists.append(
            self.ax_animation.plot(*self.track.polygon.exterior.xy, color='black', zorder=1)[0]
        )
        for interior in self.track.polygon.interiors:
            self.track_artists.append(
                self.ax_animation.plot(*interior.xy, color='black', zorder=1)[0]
            )

        cmap = matplotlib.colormaps.get_cmap('rainbow_r')
        norm = Normalize(vmin=0, vmax=len(self.track.approx_path.coords) - 1)
        for i, (start, end) in enumerate(zip(self.track.approx_path.coords[:-1], self.track.approx_path.coords[1:])):
            arrow = FancyArrow(
                *start, end[0] - start[0], end[1] - start[1],
                width=Config.car_radius,
                length_includes_head=True,
                head_width=Config.car_radius * 2,
                head_length=Config.car_radius,
                color=cmap(norm(i)),
                alpha=0.3
            )
            self.ax_animation.add_patch(arrow)
            self.track_artists.append(arrow)
        self.ax_animation.set_xlim(
            self.track.min_x - self.track.width * 0.1,
            self.track.max_x + self.track.width * 0.1
        )
        self.ax_animation.set_ylim(
            self.track.min_y - self.track.height * 0.1,
            self.track.max_y + self.track.height * 0.1
        )

        self.ax_animation.draw_artist(self.ax_animation.patch)
        for artist in self.ax_animation.get_children():
            self.ax_animation.draw_artist(artist)
        self.ax_animation.figure.canvas.blit(self.ax_animation.bbox)

    def _load_all_completed_weights(self, filename: str):
        with h5py.File(filename, 'r') as file:
            for group_name in file.keys():
                group = file[group_name]
                weights = [np.array(group[layer_name]) for layer_name in group if layer_name.startswith('layer_')]
                self.cars.append(Car(track=self.track, network=MLP(weights=weights)))
        Config.population = len(self.cars)

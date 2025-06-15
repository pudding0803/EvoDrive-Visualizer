from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from car import Car


class Selection(ABC):
    @abstractmethod
    def select(self, cars: list[Car]) -> tuple[Car, Car]:
        pass


class RouletteWheelSelection(Selection):
    def select(self, cars: list[Car]):
        weights = [car.fitness for car in cars]
        return tuple(random.choices(cars, weights=weights, k=2))


class LinearRankingSelection(Selection):
    def __init__(self, pressure: float = 1.5):
        if not (1 <= pressure <= 2):
            raise ValueError('The pressure must be in the range [1, 2] for linear ranking.')
        self.s = pressure

    def select(self, cars: list[Car]):
        sorted_cars = sorted(cars, key=lambda car: car.fitness, reverse=True)
        fitness = np.array([car.fitness for car in sorted_cars])
        _, ranks = np.unique(-fitness, return_inverse=True)
        mu = len(cars)
        weights = (2 - self.s) / mu + (2 * ranks * (self.s - 1)) / (mu * (mu - 1))
        return tuple(random.choices(sorted_cars, weights=weights, k=2))


class TournamentSelection(Selection):
    def __init__(self, tournament_size: int = 2):
        self.tournament_size = tournament_size

    def select(self, cars: list[Car]):
        return self.tournament(cars), self.tournament(cars)

    def tournament(self, cars: list[Car]):
        candidates = random.choices(cars, k=self.tournament_size)
        return max(candidates, key=lambda car: car.fitness)


class SelectionStrategy:
    strategies = {
        'roulette_wheel': RouletteWheelSelection,
        'linear_ranking': LinearRankingSelection,
        'tournament': TournamentSelection
    }

    def __init__(self, strategy: str, **kwargs):
        if strategy not in SelectionStrategy.strategies:
            raise ValueError(f'Invalid strategy name: {strategy}')
        self.strategy = self.strategies[strategy](**kwargs)

    def __call__(self, cars: list[Car]) -> tuple[Car, Car]:
        return self.strategy.select(cars)

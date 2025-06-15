import numpy as np
from config import Config

class MLP:
    layer_sizes = (5, *Config.hidden_layer_sizes, 1)

    def __init__(self, weights: list[np.ndarray] = None):
        self.weights = MLP.generate_weights() if weights is None else weights

    @staticmethod
    def generate_weights() -> list[np.ndarray]:
        return [
            np.vstack([
                np.zeros((1, j)),
                np.random.uniform(
                    -np.sqrt(6 / (i + j)) if j == 1 else -np.sqrt(2 / i),
                    np.sqrt(6 / (i + j)) if j == 1 else np.sqrt(2 / i),
                    (i, j)
                )
            ])
            for i, j in zip(MLP.layer_sizes[:-1], MLP.layer_sizes[1:])
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, w in enumerate(self.weights):
            x = np.concatenate(([1], x)) @ w
            x = MLP.sigmoid(x) if i == len(self.weights) - 1 else np.maximum(x, 0)
        return x

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

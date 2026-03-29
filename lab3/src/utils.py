import numpy as np


def generate_random_vector(size: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.uniform(1.0, 1000.0, size)

import numpy as np


def generate_random_vector(size: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.uniform(1.0, 100.0, size)


def calculate_chunk_bounds(rank: int, size: int, total_elements: int) -> tuple[int, int]:
    chunk_size = total_elements // size
    remainder = total_elements % size

    local_start = rank * chunk_size + min(rank, remainder)
    local_count = chunk_size + (1 if rank < remainder else 0)
    local_end = local_start + local_count

    return local_start, local_end

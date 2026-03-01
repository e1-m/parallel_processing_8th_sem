import random
from typing import Literal


def generate_array(
        size: int,
        array_type: Literal['sorted', 'reverse', 'random'] = "random",
        low: int = 0,
        high: int = 1000
):
    if array_type == "sorted":
        return list(range(size))
    elif array_type == "reverse":
        return list(range(size, 0, -1))
    else:
        return [random.randint(low, high) for _ in range(size)]

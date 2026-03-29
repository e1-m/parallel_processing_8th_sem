import argparse
import dataclasses
import json
import os
from typing import Optional

import numpy as np
from mpi4py import MPI

from src.shell_sort import parallel_shell_sort
from src.odd_even_sort import parallel_odd_even_sort
from src.utils import generate_random_vector


@dataclasses.dataclass
class ExperimentMetrics:
    task_name: str
    p: int
    avg_time: float


def save_metrics_to_json(metrics_list: list[ExperimentMetrics], filename="metrics.json"):
    data_to_append = [dataclasses.asdict(m) for m in metrics_list if m is not None]
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    existing_data.extend(data_to_append)
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)


def run_experiment(
        comm: MPI.Comm,
        task_name: str,
        vec: np.ndarray,
        sort_func,
        tries: int
) -> Optional[ExperimentMetrics]:
    rank = comm.Get_rank()
    size = comm.Get_size()
    total_time = 0.0

    for _ in range(tries):
        start_time = MPI.Wtime()
        arr = sort_func(comm, vec)
        end_time = MPI.Wtime() - start_time

        if rank == 0:
            assert np.array_equal(arr, np.sort(arr))
            total_time += end_time

    if rank == 0:
        return ExperimentMetrics(
            task_name=task_name,
            p=size,
            avg_time=total_time / tries
        )
    return None


def main():
    parser = argparse.ArgumentParser(description="MPI Lab 3")
    parser.add_argument("--size_shell", type=int, default=5000, help="Array size for Shell sort")
    parser.add_argument("--size_odd_even", type=int, default=15000, help="Array size for Odd-Even sort")
    parser.add_argument("--tries", type=int, default=5, help="Number of tries to average")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    tries = args.tries

    vec_shell = generate_random_vector(args.size_shell, seed=42)
    vec_odd_even = generate_random_vector(args.size_odd_even, seed=99)

    shell_metrics = run_experiment(
        comm, "Task 1: Shell Sort", vec_shell, parallel_shell_sort, tries
    )
    odd_even_metrics = run_experiment(
        comm, "Task 3: Odd-Even Sort", vec_odd_even, parallel_odd_even_sort, tries
    )

    if rank == 0:
        save_metrics_to_json([shell_metrics, odd_even_metrics])
        print(f"--> Done p={size} | Averaged over {tries} tries.")


if __name__ == '__main__':
    main()

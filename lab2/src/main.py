import argparse
import dataclasses
import json
import os
from typing import Optional

from mpi4py import MPI

from src.dot_product import calculate_sequential_dot_product, calculate_parallel_dot_product
from src.min_max import calculate_sequential_min_max, calculate_parallel_min_max
from src.utils import generate_random_vector


@dataclasses.dataclass
class ExperimentMetrics:
    task_name: str
    p: int
    sequential_time: float
    parallel_time: float
    speedup: float
    efficiency: float


def save_metrics_to_json(metrics_list: list[ExperimentMetrics], filename="metrics.json"):
    data_to_append = [dataclasses.asdict(m) for m in metrics_list]

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


def run_dot_product_experiment(comm: MPI.Comm, vec_a, vec_b, tries: int) -> Optional[ExperimentMetrics]:
    rank = comm.Get_rank()
    size = comm.Get_size()
    total_seq_time = 0.0
    total_par_time = 0.0

    for _ in range(tries):
        if rank == 0:
            start_seq = MPI.Wtime()
            _ = calculate_sequential_dot_product(vec_a, vec_b)
            total_seq_time += MPI.Wtime() - start_seq

        comm.Barrier()

        start_par = MPI.Wtime()
        _ = calculate_parallel_dot_product(comm, vec_a, vec_b)
        par_time = MPI.Wtime() - start_par

        if rank == 0:
            total_par_time += par_time

    if rank == 0:
        avg_seq = total_seq_time / tries
        avg_par = total_par_time / tries
        speedup = avg_seq / avg_par if avg_par > 0 else 0
        efficiency = speedup / size

        return ExperimentMetrics(
            task_name="Task 1: Dot Product",
            p=size, sequential_time=avg_seq, parallel_time=avg_par,
            speedup=speedup, efficiency=efficiency
        )
    return None


def run_min_max_experiment(comm: MPI.Comm, vec_a, tries: int) -> Optional[ExperimentMetrics]:
    rank = comm.Get_rank()
    size = comm.Get_size()
    total_seq_time = 0.0
    total_par_time = 0.0

    for _ in range(tries):
        if rank == 0:
            start_seq = MPI.Wtime()
            _ = calculate_sequential_min_max(vec_a)
            total_seq_time += MPI.Wtime() - start_seq

        comm.Barrier()

        start_par = MPI.Wtime()
        _ = calculate_parallel_min_max(comm, vec_a)
        par_time = MPI.Wtime() - start_par

        if rank == 0:
            total_par_time += par_time

    if rank == 0:
        avg_seq = total_seq_time / tries
        avg_par = total_par_time / tries
        speedup = avg_seq / avg_par if avg_par > 0 else 0
        efficiency = speedup / size

        return ExperimentMetrics(
            task_name="Task 3: Min/Max Search",
            p=size, sequential_time=avg_seq, parallel_time=avg_par,
            speedup=speedup, efficiency=efficiency
        )
    return None


def main():
    parser = argparse.ArgumentParser(description="MPI Lab 2")
    parser.add_argument("--size", type=int, default=10_000_000, help="Array size N")
    parser.add_argument("--tries", type=int, default=5, help="Number of tries to average")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = args.size
    tries = args.tries

    vec_a = generate_random_vector(n, seed=42)
    vec_b = generate_random_vector(n, seed=99)

    dot_metrics = run_dot_product_experiment(comm, vec_a, vec_b, tries)
    mm_metrics = run_min_max_experiment(comm, vec_a, tries)

    if rank == 0:
        save_metrics_to_json([dot_metrics, mm_metrics])
        print(f"--> Done p={size} | N={n} | Averaged over {tries} tries.")


if __name__ == '__main__':
    main()

import numpy as np
from mpi4py import MPI
from src.utils import calculate_chunk_bounds


def calculate_sequential_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def calculate_parallel_dot_product(comm: MPI.Comm, a: np.ndarray, b: np.ndarray) -> float:
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_idx, end_idx = calculate_chunk_bounds(rank, size, len(a))
    a_local = a[start_idx:end_idx]
    b_local = b[start_idx:end_idx]

    local_dot = np.array(np.dot(a_local, b_local), dtype='d')

    if rank == 0:
        global_dot = local_dot.copy()
        worker_result = np.array(0.0, dtype='d')

        for i in range(1, size):
            comm.Recv([worker_result, MPI.DOUBLE], source=i, tag=1)
            global_dot += worker_result

        return float(global_dot)
    else:
        comm.Send([local_dot, MPI.DOUBLE], dest=0, tag=1)
        return 0.0

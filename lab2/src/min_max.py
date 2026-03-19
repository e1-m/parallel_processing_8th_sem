import numpy as np
from mpi4py import MPI
from src.utils import calculate_chunk_bounds


def calculate_sequential_min_max(data: np.ndarray) -> tuple[float, float]:
    return float(np.min(data)), float(np.max(data))


def calculate_parallel_min_max(comm: MPI.Comm, data: np.ndarray) -> tuple[float, float]:
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_idx, end_idx = calculate_chunk_bounds(rank, size, len(data))
    data_local = data[start_idx:end_idx]

    local_min = np.array(np.min(data_local), dtype='d')
    local_max = np.array(np.max(data_local), dtype='d')

    if rank == 0:
        global_min = local_min.copy()
        global_max = local_max.copy()
        worker_min = np.array(0.0, dtype='d')
        worker_max = np.array(0.0, dtype='d')

        for i in range(1, size):
            comm.Recv([worker_min, MPI.DOUBLE], source=i, tag=2)
            comm.Recv([worker_max, MPI.DOUBLE], source=i, tag=3)

            if worker_min < global_min:
                global_min = worker_min.copy()

            if worker_max > global_max:
                global_max = worker_max.copy()

        return float(global_min), float(global_max)
    else:
        comm.Send([local_min, MPI.DOUBLE], dest=0, tag=2)
        comm.Send([local_max, MPI.DOUBLE], dest=0, tag=3)
        return 0.0, 0.0

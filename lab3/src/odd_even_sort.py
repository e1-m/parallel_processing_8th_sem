import numpy as np
from mpi4py import MPI


def compare_split(comm, local_data, partner, rank):
    n_local = len(local_data)
    n_partner = comm.sendrecv(n_local, dest=partner, source=partner)

    partner_data = np.empty(n_partner, dtype=local_data.dtype)

    comm.Sendrecv(
        sendbuf=local_data, dest=partner, sendtag=0,
        recvbuf=partner_data, source=partner, recvtag=0
    )

    combined = np.concatenate((local_data, partner_data))
    combined.sort(kind='mergesort')

    if rank < partner:
        new_local = combined[:n_local]
    else:
        new_local = combined[-n_local:]

    return new_local, not np.array_equal(local_data, new_local)


def parallel_odd_even_sort(comm: MPI.Comm, data: np.ndarray) -> np.ndarray:
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunks = np.array_split(data, size) if rank == 0 else None

    local_data = comm.scatter(chunks, root=0)
    local_data = np.sort(local_data)

    done = False
    while not done:
        local_changed = False

        partner = rank + 1 if rank % 2 == 0 else rank - 1
        if 0 <= partner < size:
            local_data, changed = compare_split(comm, local_data, partner, rank)
            local_changed = local_changed or changed

        partner = rank + 1 if rank % 2 != 0 else rank - 1

        if 0 <= partner < size:
            local_data, changed = compare_split(comm, local_data, partner, rank)
            local_changed = local_changed or changed

        global_changed = comm.allreduce(local_changed, op=MPI.LOR)
        done = not global_changed

    sorted_chunks = comm.gather(local_data, root=0)
    return np.concatenate(sorted_chunks) if rank == 0 else None

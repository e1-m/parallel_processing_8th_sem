use mpi::topology::SimpleCommunicator;
use mpi::traits::*;

pub fn integrate_mpi(
    world: &SimpleCommunicator,
    a: f64,
    b: f64,
    n: usize,
    f: fn(f64) -> f64,
    method: fn(f64, f64, usize, fn(f64) -> f64) -> f64,
) -> Result<Option<f64>, String> {
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    // Distribute intervals evenly, handing remainders to the first few ranks
    let base_n = n / size;
    let remainder = n % size;
    let local_n = if rank < remainder { base_n + 1 } else { base_n };

    // Calculate exact integration boundaries for this specific process
    let step = (b - a) / (n as f64);
    let local_start_i = rank * base_n + rank.min(remainder);

    let local_a = a + (local_start_i as f64) * step;
    let local_b = local_a + (local_n as f64) * step;

    // Evaluate the local chunk by passing 'f' down to the sequential method
    let local_res = if local_n > 0 {
        method(local_a, local_b, local_n, f)
    } else {
        0.0
    };

    let root_process = world.process_at_rank(0);
    let mut global_res = 0.0;

    // Reduce local results into a total sum at rank 0
    if rank == 0 {
        root_process.reduce_into_root(
            &local_res,
            &mut global_res,
            mpi::collective::SystemOperation::sum(),
        );
        Ok(Some(global_res))
    } else {
        root_process.reduce_into(&local_res, mpi::collective::SystemOperation::sum());
        Ok(None)
    }
}

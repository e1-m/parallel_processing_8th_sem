use crate::algs::models::Matrix;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;

pub fn solve_mpi(
    world: &SimpleCommunicator,
    a: &Matrix,
    b: &[f64],
) -> Result<Option<Vec<f64>>, String> {
    let rank = world.rank();
    let size = world.size() as usize;

    // Broadcast matrix size
    let mut n_buf = if rank == 0 { [a.rows() as i32] } else { [0] };
    world.process_at_rank(0).broadcast_into(&mut n_buf[..]);
    let n = n_buf[0] as usize;

    if rank == 0 && (a.rows() != a.cols() || a.rows() != b.len()) {
        return Err("Matrix dimensions do not match".to_string());
    }

    // Cyclic row distribution: Process `p` owns rows where `row_idx % size == p`
    let mut local_n = n / size;
    if (n % size) > rank as usize {
        local_n += 1;
    }

    let mut local_rows = vec![vec![0.0; n + 1]; local_n];

    // Root distributes augmented matrix rows
    if rank == 0 {
        let mut aug = vec![0.0; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = a[(i, j)];
            }
            aug[i * (n + 1) + n] = b[i];
        }

        let mut local_idx_counts = vec![0; size];
        for i in 0..n {
            let target_rank = i % size;
            let local_idx = local_idx_counts[target_rank];
            local_idx_counts[target_rank] += 1;

            let row_start = i * (n + 1);
            let row_slice = &aug[row_start..row_start + (n + 1)];

            if target_rank == 0 {
                local_rows[local_idx].copy_from_slice(row_slice);
            } else {
                world.process_at_rank(target_rank as i32).send(row_slice);
            }
        }
    } else {
        for i in 0..local_n {
            let (msg, _) = world.process_at_rank(0).receive_vec::<f64>();
            local_rows[i] = msg;
        }
    }

    // Forward Elimination
    for k in 0..n {
        let owner_rank = (k % size) as i32;
        let local_k = k / size;

        let mut pivot_row = vec![0.0; n + 1];
        if rank == owner_rank {
            pivot_row.copy_from_slice(&local_rows[local_k]);
        }

        world.process_at_rank(owner_rank).broadcast_into(&mut pivot_row[..]);

        let pivot_val = pivot_row[k];
        if pivot_val.abs() < 1e-12 {
            return Err(format!("Zero pivot encountered at step {}. Matrix needs pivoting.", k));
        }

        // Eliminate
        for i in 0..local_n {
            let global_row = i * size + rank as usize;
            if global_row > k {
                let factor = local_rows[i][k] / pivot_val;
                for j in k..=n {
                    local_rows[i][j] -= factor * pivot_row[j];
                }
            }
        }
    }

    // Backward Substitution
    let mut x = vec![0.0; n];
    for k in (0..n).rev() {
        let owner_rank = (k % size) as i32;
        let local_k = k / size;

        let mut x_k = [0.0];
        if rank == owner_rank {
            let mut sum = local_rows[local_k][n];
            for j in k + 1..n {
                sum -= local_rows[local_k][j] * x[j];
            }
            x_k[0] = sum / local_rows[local_k][k];
            x[k] = x_k[0];
        }

        // Broadcast solved variable to remaining steps
        world.process_at_rank(owner_rank).broadcast_into(&mut x_k[..]);
        x[k] = x_k[0];
    }

    if rank == 0 {
        Ok(Some(x))
    } else {
        Ok(None)
    }
}
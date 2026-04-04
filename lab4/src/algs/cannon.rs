use mpi::topology::SimpleCommunicator;
use mpi::traits::*;

use crate::algs::models::Matrix;
use crate::algs::naive::multiply;

fn validate_cannon_requirements(
    world: &SimpleCommunicator,
    a: &Matrix,
    b: &Matrix,
) -> Result<(usize, i32, usize), String> {
    let rank = world.rank();
    let size = world.size() as i32;
    let q = (size as f64).sqrt() as i32;

    if q * q != size {
        return Err(format!(
            "Cannon's algorithm requires a perfect square number of processes, got {}",
            size
        ));
    }

    let mut dims = [0_usize; 1];
    if rank == 0 {
        if a.rows() != a.cols() || b.rows() != b.cols() || a.cols() != b.rows() {
            return Err("Cannon's algorithm requires square matrices".to_string());
        }

        if a.rows() % (q as usize) != 0 {
            return Err(
                "Matrix dimensions must be perfectly divisible by sqrt(process_count)".to_string(),
            );
        }
        dims[0] = a.rows();
    }

    // Broadcast the matrix size to all non-root processes
    world.process_at_rank(0).broadcast_into(&mut dims[..]);
    let n = dims[0];
    let block_size = n / (q as usize);

    Ok((n, q, block_size))
}

fn scatter_blocks(
    world: &SimpleCommunicator,
    matrix: &Matrix,
    n: usize,
    q: i32,
    block_size: usize,
) -> Result<Matrix, String> {
    let rank = world.rank();
    let mut local_data = vec![0.0; block_size * block_size];

    if rank == 0 {
        for i in 0..q {
            for j in 0..q {
                let dest_rank = i * q + j;
                let mut block_data = vec![0.0; block_size * block_size];

                // Extract the block (i, j) out of the flat slice
                for r in 0..block_size {
                    let global_r = (i as usize) * block_size + r;
                    let global_c = (j as usize) * block_size;
                    let start = global_r * n + global_c;
                    let end = start + block_size;
                    block_data[r * block_size..(r + 1) * block_size]
                        .copy_from_slice(&matrix.as_slice()[start..end]);
                }

                if dest_rank == 0 {
                    local_data = block_data;
                } else {
                    world.process_at_rank(dest_rank).send(&block_data[..]);
                }
            }
        }
    } else {
        let (msg, _) = world.process_at_rank(0).receive_vec::<f64>();
        local_data = msg;
    }

    Ok(Matrix::new(local_data, block_size, block_size))
}

fn gather_blocks(
    world: &SimpleCommunicator,
    local_c: Matrix,
    n: usize,
    q: i32,
    block_size: usize,
) -> Result<Option<Matrix>, String> {
    let rank = world.rank();

    if rank == 0 {
        let mut final_data = vec![0.0; n * n];

        for i in 0..q {
            for j in 0..q {
                let src_rank = i * q + j;
                let block_data = if src_rank == 0 {
                    local_c.as_slice().to_vec()
                } else {
                    let (msg, _) = world.process_at_rank(src_rank).receive_vec::<f64>();
                    msg
                };

                // Place the block (i, j) back into the full N x N matrix slice
                for r in 0..block_size {
                    let global_r = (i as usize) * block_size + r;
                    let global_c = (j as usize) * block_size;
                    let start = global_r * n + global_c;
                    let end = start + block_size;
                    final_data[start..end]
                        .copy_from_slice(&block_data[r * block_size..(r + 1) * block_size]);
                }
            }
        }
        Ok(Some(Matrix::new(final_data, n, n)))
    } else {
        world.process_at_rank(0).send(local_c.as_slice());
        Ok(None)
    }
}

fn exchange_blocks(
    world: &SimpleCommunicator,
    local_mat: &Matrix,
    dest_rank: i32,
    src_rank: i32,
) -> Result<Matrix, String> {
    let rank = world.rank();

    if rank == dest_rank && rank == src_rank {
        return Ok(Matrix::new(
            local_mat.as_slice().to_vec(),
            local_mat.rows(),
            local_mat.cols(),
        ));
    }

    let new_data: Vec<f64>;

    // Use deterministic inequality to safely break cyclic message-passing
    // dependencies and prevent deadlocks on the ring topology.
    if rank < dest_rank {
        world.process_at_rank(dest_rank).send(local_mat.as_slice());
        let (msg, _) = world.process_at_rank(src_rank).receive_vec::<f64>();
        new_data = msg;
    } else {
        let (msg, _) = world.process_at_rank(src_rank).receive_vec::<f64>();
        world.process_at_rank(dest_rank).send(local_mat.as_slice());
        new_data = msg;
    }

    Ok(Matrix::new(new_data, local_mat.rows(), local_mat.cols()))
}

fn shift_left(
    world: &SimpleCommunicator,
    local_mat: &Matrix,
    row: i32,
    col: i32,
    q: i32,
    amount: i32,
) -> Result<Matrix, String> {
    if amount == 0 {
        return Ok(Matrix::new(
            local_mat.as_slice().to_vec(),
            local_mat.rows(),
            local_mat.cols(),
        ));
    }

    let dest_col = (col + q - (amount % q)) % q;
    let src_col = (col + amount) % q;

    let dest_rank = row * q + dest_col;
    let src_rank = row * q + src_col;

    exchange_blocks(world, local_mat, dest_rank, src_rank)
}

fn shift_up(
    world: &SimpleCommunicator,
    local_mat: &Matrix,
    row: i32,
    col: i32,
    q: i32,
    amount: i32,
) -> Result<Matrix, String> {
    if amount == 0 {
        return Ok(Matrix::new(
            local_mat.as_slice().to_vec(),
            local_mat.rows(),
            local_mat.cols(),
        ));
    }

    let dest_row = (row + q - (amount % q)) % q;
    let src_row = (row + amount) % q;

    let dest_rank = dest_row * q + col;
    let src_rank = src_row * q + col;

    exchange_blocks(world, local_mat, dest_rank, src_rank)
}

pub fn multiply_mpi(
    world: &SimpleCommunicator,
    a: &Matrix,
    b: &Matrix,
) -> Result<Option<Matrix>, String> {
    let (n, q, block_size) = validate_cannon_requirements(world, a, b)?;

    // Scatter initial blocks
    let mut local_a = scatter_blocks(world, a, n, q, block_size)?;
    let mut local_b = scatter_blocks(world, b, n, q, block_size)?;

    let rank = world.rank();
    let row = rank / q;
    let col = rank % q;

    // Initial alignment (Skewing)
    // Row i is shifted left by i
    local_a = shift_left(world, &local_a, row, col, q, row)?;
    // Column j is shifted up by j
    local_b = shift_up(world, &local_b, row, col, q, col)?;

    let mut local_c = Matrix::zero(block_size, block_size);

    // Main computation and shift loop
    for _ in 0..q {
        // Multiply local blocks
        let temp_c = multiply(&local_a, &local_b)?;

        // Accumulate into local C
        for i in 0..block_size {
            for j in 0..block_size {
                let current_val = local_c[(i, j)];
                local_c[(i, j)] = current_val + temp_c[(i, j)];
            }
        }

        // Shift A left by 1 and B up by 1
        local_a = shift_left(world, &local_a, row, col, q, 1)?;
        local_b = shift_up(world, &local_b, row, col, q, 1)?;
    }

    // Gather the resulting blocks back to root
    gather_blocks(world, local_c, n, q, block_size)
}

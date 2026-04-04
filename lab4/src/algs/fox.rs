use mpi::topology::SimpleCommunicator;
use mpi::traits::*;

use crate::algs::models::Matrix;
use crate::algs::naive::multiply;

fn scatter_2d(
    world: &SimpleCommunicator,
    matrix: &Matrix,
    n: usize,
    q: usize,
) -> Result<Matrix, String> {
    let rank = world.rank() as usize;
    let root = world.process_at_rank(0);
    let block_size = n / q;

    if rank == 0 {
        let mut local_data = Vec::with_capacity(block_size * block_size);

        for i in 0..q {
            for j in 0..q {
                let target_rank = i * q + j;
                let mut block_data = Vec::with_capacity(block_size * block_size);

                // Extract the sub-matrix block
                for r in 0..block_size {
                    let start = (i * block_size + r) * n + (j * block_size);
                    let end = start + block_size;
                    block_data.extend_from_slice(&matrix.as_slice()[start..end]);
                }

                if target_rank != 0 {
                    world.process_at_rank(target_rank as i32).send(&block_data);
                } else {
                    local_data = block_data;
                }
            }
        }
        Ok(Matrix::new(local_data, block_size, block_size))
    } else {
        let (msg, _) = root.receive_vec::<f64>();
        Ok(Matrix::new(msg, block_size, block_size))
    }
}

fn gather_2d(
    world: &SimpleCommunicator,
    local_c: Matrix,
    q: usize,
) -> Result<Option<Matrix>, String> {
    let rank = world.rank() as usize;
    let root = world.process_at_rank(0);

    // Derive dimensions locally
    let block_size = local_c.rows();
    let n = q * block_size;

    if rank == 0 {
        let mut final_data = vec![0.0; n * n];

        for i in 0..q {
            for j in 0..q {
                let source_rank = i * q + j;
                let block_data = if source_rank == 0 {
                    local_c.as_slice().to_vec()
                } else {
                    let (msg, _) = world
                        .process_at_rank(source_rank as i32)
                        .receive_vec::<f64>();
                    msg
                };

                // Place the sub-matrix block back into the final matrix
                for r in 0..block_size {
                    let start = (i * block_size + r) * n + (j * block_size);
                    let end = start + block_size;
                    final_data[start..end]
                        .copy_from_slice(&block_data[r * block_size..(r + 1) * block_size]);
                }
            }
        }
        Ok(Some(Matrix::new(final_data, n, n)))
    } else {
        root.send(local_c.as_slice());
        Ok(None)
    }
}

fn broadcast_a_block(
    world: &SimpleCommunicator,
    local_a: &Matrix,
    root_col: usize,
    my_row: usize,
    my_col: usize,
    q: usize,
) -> Result<Matrix, String> {
    let root_rank = my_row * q + root_col;
    let block_size = local_a.rows();

    if my_col == root_col {
        // We are the sender for this specific row
        for c in 0..q {
            if c != my_col {
                let target_rank = my_row * q + c;
                world
                    .process_at_rank(target_rank as i32)
                    .send(local_a.as_slice());
            }
        }
        Ok(local_a.clone())
    } else {
        // We are the receiver
        let (msg, _) = world.process_at_rank(root_rank as i32).receive_vec::<f64>();
        Ok(Matrix::new(msg, block_size, block_size))
    }
}

fn shift_b_up(
    world: &SimpleCommunicator,
    local_b: &Matrix,
    my_row: usize,
    my_col: usize,
    q: usize,
) -> Result<Matrix, String> {
    if q <= 1 {
        return Ok(local_b.clone());
    }

    let up_row = (my_row + q - 1) % q;
    let down_row = (my_row + 1) % q;

    let up_rank = up_row * q + my_col;
    let down_rank = down_row * q + my_col;

    let block_size = local_b.rows();

    // Deadlock prevention by alternating send/receive order based on even/odd rows
    if my_row % 2 == 0 {
        world
            .process_at_rank(up_rank as i32)
            .send(local_b.as_slice());
        let (msg, _) = world.process_at_rank(down_rank as i32).receive_vec::<f64>();
        Ok(Matrix::new(msg, block_size, block_size))
    } else {
        let (msg, _) = world.process_at_rank(down_rank as i32).receive_vec::<f64>();
        world
            .process_at_rank(up_rank as i32)
            .send(local_b.as_slice());
        Ok(Matrix::new(msg, block_size, block_size))
    }
}

fn add_assign_matrix(a: &mut Matrix, b: &Matrix) {
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            a[(i, j)] += b[(i, j)];
        }
    }
}

pub fn multiply_mpi(
    world: &SimpleCommunicator,
    a: &Matrix,
    b: &Matrix,
) -> Result<Option<Matrix>, String> {
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    let q = (size as f64).sqrt() as usize;

    let mut n_payload: u64 = 0;

    if rank == 0 {
        if a.rows() != a.cols() || q * q != size || a.rows() % q != 0 {
            return Err(
                "Fox's algorithm requires: square matrices, total processes = q^2, and dimension 'n' perfectly divisible by 'q'.".into(),
            );
        }
        n_payload = a.rows() as u64;
    }

    world.process_at_rank(0).broadcast_into(&mut n_payload);

    let n = n_payload as usize;
    let block_size = n / q;

    let my_row = rank / q;
    let my_col = rank % q;

    // Data Distribution
    let local_a = scatter_2d(world, a, n, q)?;
    let mut local_b = scatter_2d(world, b, n, q)?;
    let mut local_c = Matrix::zero(block_size, block_size);

    // Fox's Algorithm Iterations
    for step in 0..q {
        let root_col = (my_row + step) % q;

        // Broadcast block A across the row
        let temp_a = broadcast_a_block(world, &local_a, root_col, my_row, my_col, q)?;

        // Multiplication and addition
        let partial_c = multiply(&temp_a, &local_b)?;
        add_assign_matrix(&mut local_c, &partial_c);

        // Shift block B up the column
        local_b = shift_b_up(world, &local_b, my_row, my_col, q)?;
    }

    // Gather final computed matrix
    gather_2d(world, local_c, q)
}

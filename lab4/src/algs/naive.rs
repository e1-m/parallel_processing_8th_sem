use mpi::topology::SimpleCommunicator;
use mpi::traits::*;

use crate::algs::models::Matrix;

pub fn multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.cols() != b.rows() {
        return Err("Matrix dimensions do not match for multiplication".to_string());
    }

    let mut c = Matrix::zero(a.rows(), b.cols());

    for i in 0..a.rows() {
        for j in 0..b.cols() {
            let mut sum = 0.0;
            for k in 0..a.cols() {
                sum += a[(i, k)] * b[(k, j)];
            }
            c[(i, j)] = sum;
        }
    }

    Ok(c)
}

fn validate_dimensions(a: &Matrix, b: &Matrix) -> Result<(), String> {
    if a.cols() != b.rows() {
        return Err("Matrix dimensions do not match".to_string());
    }
    Ok(())
}

fn scatter_matrix_a(world: &SimpleCommunicator, a: &Matrix) -> Result<Matrix, String> {
    let rank = world.rank();
    let size = world.size() as usize;
    let total_rows = a.rows();
    let cols = a.cols();

    let base_rows = total_rows / size;
    let remainder = total_rows % size;

    let get_local_rows = |r: usize| {
        if r < remainder {
            base_rows + 1
        } else {
            base_rows
        }
    };

    let root_process = world.process_at_rank(0);

    if rank == 0 {
        let mut offset = get_local_rows(0) * cols;

        for p in 1..size {
            let p_rows = get_local_rows(p);
            let p_len = p_rows * cols;

            if p_len > 0 {
                let start = offset;
                let end = offset + p_len;
                world
                    .process_at_rank(p as i32)
                    .send(&a.as_slice()[start..end]);
                offset += p_len;
            }
        }

        let root_rows = get_local_rows(0);
        let root_data = a.as_slice()[0..root_rows * cols].to_vec();
        Ok(Matrix::new(root_data, root_rows, cols))
    } else {
        let local_rows = get_local_rows(rank as usize);

        if local_rows > 0 {
            let (msg, _) = root_process.receive_vec::<f64>();
            Ok(Matrix::new(msg, local_rows, cols))
        } else {
            // Edge case: more processes than rows
            Ok(Matrix::new(vec![], 0, cols))
        }
    }
}

fn broadcast_matrix_b(world: &SimpleCommunicator, b: &Matrix) -> Result<Matrix, String> {
    let rank = world.rank();
    let root_process = world.process_at_rank(0);

    let mut b_buffer = if rank == 0 {
        b.as_slice().to_vec()
    } else {
        vec![0.0; b.rows() * b.cols()]
    };

    root_process.broadcast_into(&mut b_buffer[..]);

    Ok(Matrix::new(b_buffer, b.rows(), b.cols()))
}

fn gather_matrix_c(
    world: &SimpleCommunicator,
    local_c: Matrix,
    total_rows: usize,
    total_cols: usize,
) -> Result<Option<Matrix>, String> {
    let rank = world.rank();
    let size = world.size() as usize;
    let root = world.process_at_rank(0);

    let base_rows = total_rows / size;
    let remainder = total_rows % size;
    let get_local_rows = |r: usize| {
        if r < remainder {
            base_rows + 1
        } else {
            base_rows
        }
    };

    if rank == 0 {
        let mut final_data = vec![0.0; total_rows * total_cols];

        let root_rows = get_local_rows(0);
        let root_len = root_rows * total_cols;
        final_data[0..root_len].copy_from_slice(local_c.as_slice());

        let mut current_offset = root_len;

        for p in 1..size {
            let p_rows = get_local_rows(p);
            let p_len = p_rows * total_cols;

            if p_len > 0 {
                let (msg, _) = world.process_at_rank(p as i32).receive_vec::<f64>();

                if msg.len() != p_len {
                    return Err(format!(
                        "Rank {} sent {} elements, expected {}",
                        p,
                        msg.len(),
                        p_len
                    ));
                }

                final_data[current_offset..current_offset + p_len].copy_from_slice(&msg);
                current_offset += p_len;
            }
        }
        Ok(Some(Matrix::new(final_data, total_rows, total_cols)))
    } else {
        if local_c.rows() > 0 {
            root.send(local_c.as_slice());
        }
        Ok(None)
    }
}

pub fn multiply_mpi(
    world: &SimpleCommunicator,
    a: &Matrix,
    b: &Matrix,
) -> Result<Option<Matrix>, String> {
    validate_dimensions(a, b)?;

    let b_matrix = broadcast_matrix_b(world, b)?;
    let local_a_matrix = scatter_matrix_a(world, a)?;

    let local_c = multiply(&local_a_matrix, &b_matrix)?;

    gather_matrix_c(world, local_c, a.rows(), b.cols())
}

use crate::algs::models::Matrix;

pub fn solve_seq(a: &Matrix, b: &[f64]) -> Result<Vec<f64>, String> {
    let n = a.rows();
    if n != a.cols() || n != b.len() {
        return Err("Matrix dimensions do not match".to_string());
    }

    // Augment matrix A with vector b
    let mut aug = Matrix::zero(n, n + 1);
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = a[(i, j)];
        }
        aug[(i, n)] = b[i];
    }

    // Forward elimination
    for k in 0..n {
        if aug[(k, k)].abs() < 1e-12 {
            return Err(format!("Matrix is singular at step {}", k));
        }

        for i in k + 1..n {
            let factor = aug[(i, k)] / aug[(k, k)];
            for j in k..=n {
                aug[(i, j)] -= factor * aug[(k, j)];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[(i, n)];
        for j in i + 1..n {
            sum -= aug[(i, j)] * x[j];
        }
        x[i] = sum / aug[(i, i)];
    }

    Ok(x)
}

#[allow(dead_code)]
pub fn solve_mpi_mock(
    _world: &mpi::topology::SimpleCommunicator,
    a: &Matrix,
    b: &[f64],
) -> Result<Option<Vec<f64>>, String> {
    solve_seq(a, b).map(Some)
}

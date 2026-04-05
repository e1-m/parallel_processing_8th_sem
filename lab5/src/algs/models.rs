use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        Self { data, rows, cols }
    }

    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Generates a diagonally dominant matrix to guarantee stability
    /// for basic Gaussian elimination without partial pivoting.
    pub fn random_diagonally_dominant(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            let mut row_sum = 0.0;
            for j in 0..cols {
                let val = rand::random::<f64>();
                data[i * cols + j] = val;
                if i != j {
                    row_sum += val;
                }
            }
            // Ensure diagonal dominance
            if i < cols {
                data[i * cols + i] = row_sum + 1.0 + rand::random::<f64>();
            }
        }
        Self { data, rows, cols }
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}
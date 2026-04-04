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

    pub fn random(rows: usize, cols: usize) -> Self {
        let data = (0..rows * cols).map(|_| rand::random::<f64>()).collect();
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

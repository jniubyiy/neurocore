// src/tensor/tensor2d.rs

#[derive(Debug, Clone)]
pub struct Tensor2D {
    pub dim1: usize,
    pub dim2: usize,
    pub data: Vec<Vec<f32>>,
}

impl Tensor2D {
    pub fn new(data: Vec<Vec<f32>>) -> Self {
        let dim1 = data.len();
        let dim2 = if dim1 > 0 { data[0].len() } else { 0 };
        Tensor2D { dim1, dim2, data }
    }
    pub fn zeros(dim1: usize, dim2: usize) -> Self {
        Tensor2D { dim1, dim2, data: vec![vec![0.0; dim2]; dim1] }
    }
    pub fn row(&self, r: usize) -> Vec<f32> {
        self.data[r].clone()
    }
    pub fn from_scalar(value: f32) -> Self {
        Tensor2D { dim1: 1, dim2: 1, data: vec![vec![value]] }
    }
}





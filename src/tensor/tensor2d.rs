use super::tensor1d::Tensor1D;

#[derive(Debug, Clone)]
pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f32>>,
}

impl Tensor2D {
    pub fn new(data: Vec<Vec<f32>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Tensor2D { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor2D {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn row(&self, r: usize) -> Tensor1D {
        Tensor1D::new(self.data[r].clone())
    }
}






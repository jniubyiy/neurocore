use super::tensor2d::Tensor2D;

#[derive(Debug, Clone)]
pub struct Tensor3D {
    pub depth: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<Vec<f32>>>,
}

impl Tensor3D {
    pub fn new(data: Vec<Vec<Vec<f32>>>) -> Self {
        let depth = data.len();
        let rows = if depth > 0 { data[0].len() } else { 0 };
        let cols = if rows > 0 { data[0][0].len() } else { 0 };
        Tensor3D { depth, rows, cols, data }
    }

    pub fn zeros(depth: usize, rows: usize, cols: usize) -> Self {
        Tensor3D {
            depth,
            rows,
            cols,
            data: vec![vec![vec![0.0; cols]; rows]; depth],
        }
    }

    pub fn slice_2d(&self, d: usize) -> Tensor2D {
        Tensor2D::new(self.data[d].clone())
    }
}






use super::tensor3d::Tensor3D;

#[derive(Debug, Clone)]
pub struct Tensor4D {
    pub dim1: usize,
    pub depth: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<Vec<Vec<f32>>>>,
}

impl Tensor4D {
    pub fn new(data: Vec<Vec<Vec<Vec<f32>>>>) -> Self {
        let dim1 = data.len();
        let depth = if dim1 > 0 { data[0].len() } else { 0 };
        let rows = if depth > 0 { data[0][0].len() } else { 0 };
        let cols = if rows > 0 { data[0][0][0].len() } else { 0 };
        Tensor4D { dim1, depth, rows, cols, data }
    }

    pub fn zeros(dim1: usize, depth: usize, rows: usize, cols: usize) -> Self {
        Tensor4D {
            dim1,
            depth,
            rows,
            cols,
            data: vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1],
        }
    }

    pub fn slice_3d(&self, d: usize) -> Tensor3D {
        Tensor3D::new(self.data[d].clone())
    }
}






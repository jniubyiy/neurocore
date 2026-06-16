use super::tensor4d::Tensor4D;

#[derive(Debug, Clone)]
pub struct Tensor5D {
    pub outer: usize,
    pub dim1: usize,
    pub depth: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
}

impl Tensor5D {
    pub fn new(data: Vec<Vec<Vec<Vec<Vec<f32>>>>>) -> Self {
        let outer = data.len();
        let dim1 = if outer > 0 { data[0].len() } else { 0 };
        let depth = if dim1 > 0 { data[0][0].len() } else { 0 };
        let rows = if depth > 0 { data[0][0][0].len() } else { 0 };
        let cols = if rows > 0 { data[0][0][0][0].len() } else { 0 };
        Tensor5D { outer, dim1, depth, rows, cols, data }
    }

    pub fn zeros(outer: usize, dim1: usize, depth: usize, rows: usize, cols: usize) -> Self {
        Tensor5D {
            outer,
            dim1,
            depth,
            rows,
            cols,
            data: vec![vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1]; outer],
        }
    }

    pub fn slice_4d(&self, o: usize) -> Tensor4D {
        Tensor4D::new(self.data[o].clone())
    }
}






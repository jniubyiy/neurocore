use super::tensor2d::Tensor2D;

#[derive(Debug, Clone)]
pub struct Tensor3D {
    pub dim1: usize,
    pub dim2: usize,
    pub dim3: usize,
    pub data: Vec<Vec<Vec<f32>>>,
}

impl Tensor3D {
    pub fn new(data: Vec<Vec<Vec<f32>>>) -> Self {
        let dim1 = data.len();
        let dim2 = if dim1 > 0 { data[0].len() } else { 0 };
        let dim3 = if dim2 > 0 { data[0][0].len() } else { 0 };
        Tensor3D { dim1, dim2, dim3, data }
    }
    pub fn zeros(dim1: usize, dim2: usize, dim3: usize) -> Self {
        Tensor3D { dim1, dim2, dim3, data: vec![vec![vec![0.0; dim3]; dim2]; dim1] }
    }
    pub fn slice_2d(&self, i: usize) -> Tensor2D {
        Tensor2D::new(self.data[i].clone())
    }
}






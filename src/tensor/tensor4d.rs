use super::tensor3d::Tensor3D;

#[derive(Debug, Clone)]
pub struct Tensor4D {
    pub dim1: usize,
    pub dim2: usize,
    pub dim3: usize,
    pub dim4: usize,
    pub data: Vec<Vec<Vec<Vec<f32>>>>,
}

impl Tensor4D {
    pub fn new(data: Vec<Vec<Vec<Vec<f32>>>>) -> Self {
        let dim1 = data.len();
        let dim2 = if dim1 > 0 { data[0].len() } else { 0 };
        let dim3 = if dim2 > 0 { data[0][0].len() } else { 0 };
        let dim4 = if dim3 > 0 { data[0][0][0].len() } else { 0 };
        Tensor4D { dim1, dim2, dim3, dim4, data }
    }
    pub fn zeros(dim1: usize, dim2: usize, dim3: usize, dim4: usize) -> Self {
        Tensor4D { dim1, dim2, dim3, dim4, data: vec![vec![vec![vec![0.0; dim4]; dim3]; dim2]; dim1] }
    }
    pub fn slice_3d(&self, i: usize) -> Tensor3D {
        Tensor3D::new(self.data[i].clone())
    }
}






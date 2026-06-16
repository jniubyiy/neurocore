#[derive(Debug, Clone)]
pub struct Tensor1D {
    pub data: Vec<f32>,
}

impl Tensor1D {
    pub fn new(data: Vec<f32>) -> Self {
        Tensor1D { data }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn zeros(size: usize) -> Self {
        Tensor1D {
            data: vec![0.0; size],
        }
    }
    pub fn from_scalar(value: f32) -> Self {
        Tensor1D {
            data: vec![value],
        }
    }
}






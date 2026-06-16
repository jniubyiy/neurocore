use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;

pub trait Parameter {
    fn value(&self) -> f32;
    fn global_index(&self) -> usize;
    fn as_dual(&self, total_params: usize) -> (Tensor1D, Jacobian) {
        let mut j = Jacobian::new(1, total_params);
        if self.global_index() < total_params {
            j.data[0][self.global_index()] = 1.0;
        }
        (Tensor1D::from_scalar(self.value()), j)
    }
}

pub struct ScalarParam {
    pub val: f32,
    pub index: usize,
}

impl ScalarParam {
    pub fn new(val: f32, index: usize) -> Self {
        ScalarParam { val, index }
    }
}

impl Parameter for ScalarParam {
    fn value(&self) -> f32 { self.val }
    fn global_index(&self) -> usize { self.index }
}






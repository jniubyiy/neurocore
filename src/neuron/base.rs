use faer::Mat;
use crate::tensor::Tensor1D;

pub trait Neuron {
    fn apply(&self, x: f32) -> f32;
    fn forward(&self, input: &Tensor1D) -> Tensor1D {
        let out: Vec<f32> = input.data.iter().map(|&x| self.apply(x)).collect();
        Tensor1D::new(out)
    }
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32>;
}



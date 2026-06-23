use faer::Mat;
use crate::neuron::base::Neuron;

pub struct ReLU;

impl Neuron for ReLU {
    /// Принимает матрицу (batch, 1), возвращает (batch, 1) с ReLU-активацией.
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let mut out = Mat::zeros(input.nrows(), 1);
        for i in 0..input.nrows() {
            out[(i, 0)] = input[(i, 0)].max(0.0);
        }
        out
    }
}





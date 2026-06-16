use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::neuron::base::Neuron;

pub struct ReLU;

impl Neuron for ReLU {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let n = input.len();
        let p = j_input.cols();
        let mut out_data = vec![0.0; n];
        let mut j_out = Jacobian::new(n, p);

        for i in 0..n {
            let x = input.data[i];
            let activated = if x > 0.0 { x } else { 0.0 };
            out_data[i] = activated;
            let grad = if x > 0.0 { 1.0 } else { 0.0 };
            for j in 0..p {
                j_out.data[i][j] = j_input.data[i][j] * grad;
            }
        }
        (Tensor1D::new(out_data), j_out)
    }
}






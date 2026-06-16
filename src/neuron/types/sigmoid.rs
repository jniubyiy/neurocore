use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::neuron::base::Neuron;

pub struct Sigmoid;

impl Neuron for Sigmoid {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let n = input.len();
        let p = j_input.cols();
        let mut out_data = vec![0.0; n];
        let mut j_out = Jacobian::new(n, p);

        for i in 0..n {
            let x = input.data[i];
            let sig = 1.0 / (1.0 + (-x).exp());
            out_data[i] = sig;
            let grad = sig * (1.0 - sig);
            for j in 0..p {
                j_out.data[i][j] = j_input.data[i][j] * grad;
            }
        }
        (Tensor1D::new(out_data), j_out)
    }
}






use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::neuron::base::Neuron;

pub struct LeakyReLU {
    alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self {
        LeakyReLU { alpha }
    }
}

impl Neuron for LeakyReLU {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let n = input.len();
        let p = j_input.cols();
        let mut out_data = vec![0.0; n];
        let mut j_out = Jacobian::new(n, p);

        for i in 0..n {
            let x = input.data[i];
            if x > 0.0 {
                out_data[i] = x;
                for j in 0..p {
                    j_out.data[i][j] = j_input.data[i][j]; // градиент = 1
                }
            } else {
                out_data[i] = self.alpha * x;
                for j in 0..p {
                    j_out.data[i][j] = j_input.data[i][j] * self.alpha;
                }
            }
        }
        (Tensor1D::new(out_data), j_out)
    }
}






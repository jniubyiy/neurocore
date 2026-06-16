use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::neuron::base::Neuron;

pub struct Linear {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Linear {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Linear { weights, bias }
    }
}

impl Neuron for Linear {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let n = input.len();
        assert_eq!(n, self.weights.len(), "Linear: input length must match weights length");
        let p = j_input.num_params;          // <-- исправлено

        let mut out_val = self.bias;
        for i in 0..n {
            out_val += self.weights[i] * input.data[i];
        }

        let mut out_jac = vec![0.0_f32; p];
        for i in 0..n {
            for j in 0..p {
                out_jac[j] += self.weights[i] * j_input.data[i][j];
            }
        }

        (Tensor1D::from_scalar(out_val), Jacobian {
            out_features: 1,
            num_params: p,
            data: vec![out_jac],
        })
    }
}






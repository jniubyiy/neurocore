use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::neuron::base::Neuron;

pub struct SoftSparseGate {
    thresholds: Vec<f32>,
    temperature: f32,
}

impl SoftSparseGate {
    pub fn new(input_dim: usize, _start_param_index: usize, temperature: f32) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        Self {
            thresholds: vec![0.0; input_dim],
            temperature,
        }
    }
}

impl Neuron for SoftSparseGate {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let n = input.len();
        assert_eq!(n, self.thresholds.len(), "input size mismatch");
        let p = j_input.cols();
        let temp = self.temperature;

        let mut out_data = vec![0.0; n];
        let mut j_out = Jacobian::new(n, p);

        for i in 0..n {
            let x = input.data[i];
            let thr = self.thresholds[i];
            let abs_x = x.abs();
            let z = (abs_x - thr) / temp;
            let gate = 1.0 / (1.0 + (-z).exp());
            out_data[i] = x * gate;

            let dgate_dz = gate * (1.0 - gate);
            let dz_dx = if x >= 0.0 { 1.0 / temp } else { -1.0 / temp };
            let dout_dx = gate + x * dgate_dz * dz_dx;

            for j in 0..p {
                j_out.data[i][j] = dout_dx * j_input.data[i][j];
                // пороги считаем константами, поэтому дополнительных слагаемых нет
            }
        }
        (Tensor1D::new(out_data), j_out)
    }
}






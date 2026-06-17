use faer::Mat;
use crate::neuron::base::Neuron;

pub struct SoftKeepGate {
    pub thresholds: Vec<f32>,
    pub temperature: f32,
}

impl SoftKeepGate {
    pub fn new(thresholds: Vec<f32>, temperature: f32) -> Self {
        assert!(temperature > 0.0);
        assert!(!thresholds.is_empty());
        SoftKeepGate { thresholds, temperature }
    }

    pub fn param_count(&self) -> usize { self.thresholds.len() }
    pub fn get_params(&self) -> Vec<f32> { self.thresholds.clone() }
    pub fn set_params(&mut self, values: &[f32]) {
        assert_eq!(values.len(), self.thresholds.len());
        self.thresholds.copy_from_slice(values);
    }
}

impl Neuron for SoftKeepGate {
    fn apply(&self, _x: f32) -> f32 {
        panic!("SoftKeepGate does not support element‑wise apply; use forward()");
    }

    fn forward(&self, input: &crate::tensor::Tensor1D) -> crate::tensor::Tensor1D {
        let temp = self.temperature;
        let out: Vec<f32> = input.data.iter().enumerate().map(|(i, &x)| {
            let thr = self.thresholds[i];
            let abs_x = x.abs();
            let z = (thr - abs_x) / temp;
            let gate = 1.0 / (1.0 + (-z).exp());
            x * gate
        }).collect();
        crate::tensor::Tensor1D::new(out)
    }

    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let dim = input.ncols();
        assert_eq!(dim, self.thresholds.len());
        let temp = self.temperature;
        let mut out = Mat::zeros(batch, dim);
        for i in 0..batch {
            for j in 0..dim {
                let x = input[(i, j)];
                let thr = self.thresholds[j];
                let abs_x = x.abs();
                let z = (thr - abs_x) / temp;
                let gate = 1.0 / (1.0 + (-z).exp());
                out[(i, j)] = x * gate;
            }
        }
        out
    }
}




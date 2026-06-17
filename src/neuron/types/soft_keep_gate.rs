use crate::tensor::Tensor1D;
use crate::neuron::base::Neuron;

pub struct SoftKeepGate {
    pub thresholds: Vec<f32>,
    pub temperature: f32,
}

impl SoftKeepGate {
    pub fn new(thresholds: Vec<f32>, temperature: f32) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        assert!(!thresholds.is_empty(), "thresholds cannot be empty");
        SoftKeepGate { thresholds, temperature }
    }

    pub fn param_count(&self) -> usize {
        self.thresholds.len()
    }

    pub fn get_params(&self) -> Vec<f32> {
        self.thresholds.clone()
    }

    pub fn set_params(&mut self, values: &[f32]) {
        assert_eq!(values.len(), self.thresholds.len());
        self.thresholds.copy_from_slice(values);
    }
}

impl Neuron for SoftKeepGate {
    fn apply(&self, x: f32) -> f32 {
        panic!("SoftKeepGate does not support element‑wise apply; use forward()");
    }

    fn forward(&self, input: &Tensor1D) -> Tensor1D {
        assert_eq!(input.len(), self.thresholds.len(), "SoftKeepGate: input size mismatch");
        let temp = self.temperature;
        let out: Vec<f32> = input.data.iter().enumerate().map(|(i, &x)| {
            let thr = self.thresholds[i];
            let abs_x = x.abs();
            let z = (thr - abs_x) / temp;
            let gate = 1.0 / (1.0 + (-z).exp());
            x * gate
        }).collect();
        Tensor1D::new(out)
    }
}





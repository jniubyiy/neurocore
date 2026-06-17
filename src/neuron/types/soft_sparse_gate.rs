use crate::tensor::Tensor1D;
use crate::neuron::base::Neuron;

pub struct SoftSparseGate {
    pub thresholds: Vec<f32>,
    pub temperature: f32,
}

impl SoftSparseGate {
    pub fn new(thresholds: Vec<f32>, temperature: f32) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        assert!(!thresholds.is_empty(), "thresholds cannot be empty");
        SoftSparseGate { thresholds, temperature }
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

impl Neuron for SoftSparseGate {
    fn apply(&self, x: f32) -> f32 {
        // поэлементное применение не имеет смысла, т.к. нужен индекс
        panic!("SoftSparseGate does not support element‑wise apply; use forward()");
    }

    fn forward(&self, input: &Tensor1D) -> Tensor1D {
        assert_eq!(input.len(), self.thresholds.len(), "SoftSparseGate: input size mismatch");
        let temp = self.temperature;
        let out: Vec<f32> = input.data.iter().enumerate().map(|(i, &x)| {
            let thr = self.thresholds[i];
            let abs_x = x.abs();
            let z = (abs_x - thr) / temp;
            let gate = 1.0 / (1.0 + (-z).exp());
            x * gate
        }).collect();
        Tensor1D::new(out)
    }
}






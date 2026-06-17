use crate::tensor::Tensor1D;

pub trait Neuron {
    /// Поэлементное применение активации к одному числу.
    fn apply(&self, x: f32) -> f32;

    /// Применение ко всему 1D‑тензору.
    fn forward(&self, input: &Tensor1D) -> Tensor1D {
        let out: Vec<f32> = input.data.iter().map(|&x| self.apply(x)).collect();
        Tensor1D::new(out)
    }
}





use faer::Mat;
use faer::zip;
use crate::neuron::base::Neuron;

pub struct LeakyReLU {
    pub alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self { Self { alpha } }
}

impl Neuron for LeakyReLU {
    fn apply(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { self.alpha * x }
    }

    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let alpha = self.alpha;
        zip!(input.as_ref()).map(|x| {
            let val = x.0;          // &f32
            if *val > 0.0 { *val }  // разыменовываем для сравнения и возврата
            else { alpha * (*val) }
        })
    }
}




use faer::Mat;
use crate::neuron::base::Neuron;

pub struct Softmax;

impl Neuron for Softmax {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let rows = input.nrows();
        let cols = input.ncols();
        let mut out = Mat::zeros(rows, cols);
        for i in 0..rows {
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..cols {
                max_val = max_val.max(input[(i, j)]);
            }
            let mut exps = vec![0.0_f32; cols];
            let mut sum = 0.0;
            for j in 0..cols {
                let e = (input[(i, j)] - max_val).exp();
                exps[j] = e;
                sum += e;
            }
            for j in 0..cols {
                out[(i, j)] = exps[j] / sum;
            }
        }
        out
    }
}


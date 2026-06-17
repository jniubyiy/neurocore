// ============================================================
// Файл: src/layers/layers1d/linear1d.rs
// ============================================================
use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::model_plan::blueprint::assert_power_of_two;
use crate::neuron::Linear;
use super::{Layer, LayerInfo};

pub struct LinearLayer {
    input_dim: usize,
    output_dim: usize,
    last_input: Option<Tensor1D>,
    grad_W: Vec<f32>,
    grad_b: Vec<f32>,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_power_of_two(input_dim);
        assert_power_of_two(output_dim);
        Self {
            input_dim,
            output_dim,
            last_input: None,
            grad_W: vec![0.0; input_dim * output_dim],
            grad_b: vec![0.0; output_dim],
        }
    }

    fn weight_index(&self, out_idx: usize, in_idx: usize, slice: &ParamSlice) -> usize {
        slice.start + out_idx * self.input_dim + in_idx
    }

    fn bias_index(&self, out_idx: usize, slice: &ParamSlice) -> usize {
        slice.start + self.input_dim * self.output_dim + out_idx
    }
}

impl Layer for LinearLayer {
    fn forward_into(&mut self, input: &Tensor1D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<f32>) {
        assert_eq!(input.len(), self.input_dim);
        assert_eq!(out_buf.len(), self.output_dim);
        self.last_input = Some(input.clone());

        for o in 0..self.output_dim {
            let w_start = self.weight_index(o, 0, slice);
            let w_end = w_start + self.input_dim;
            let weights = &params[w_start..w_end];
            let bias = params[self.bias_index(o, slice)];
            out_buf[o] = Linear::forward_slice(weights, bias, &input.data);
        }
    }

    fn backward(&mut self, delta: &Tensor1D, params: &[f32], slice: &ParamSlice) -> Tensor1D {
        let input = self.last_input.take()
            .expect("backward called without forward or input already consumed");
        assert_eq!(delta.len(), self.output_dim);

        for o in 0..self.output_dim {
            let d = delta.data[o];
            for i in 0..self.input_dim {
                self.grad_W[o * self.input_dim + i] += d * input.data[i];
            }
            self.grad_b[o] += d;
        }

        let mut delta_prev = vec![0.0; self.input_dim];
        for i in 0..self.input_dim {
            let mut sum = 0.0;
            for o in 0..self.output_dim {
                sum += params[self.weight_index(o, i, slice)] * delta.data[o];
            }
            delta_prev[i] = sum;
        }
        Tensor1D::new(delta_prev)
    }

    fn apply_gradients(&mut self, store: &mut ParamStore, lr: f32, slice: &ParamSlice) {
        for (idx, g) in self.grad_W.iter().enumerate() {
            store.add_to_param(slice.start + idx, -lr * g);
        }
        for (idx, g) in self.grad_b.iter().enumerate() {
            store.add_to_param(slice.start + self.input_dim * self.output_dim + idx, -lr * g);
        }
        self.grad_W.fill(0.0);
        self.grad_b.fill(0.0);
    }

    fn param_len(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Linear".to_string(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}





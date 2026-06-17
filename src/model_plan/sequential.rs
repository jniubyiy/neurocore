use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::layers::{Layer, LayerInfo};

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    slices: Vec<ParamSlice>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>, slices: Vec<ParamSlice>) -> Self {
        assert_eq!(layers.len(), slices.len());
        Self { layers, slices }
    }
}

impl Layer for Sequential {
    fn forward_into(&mut self, input: &Tensor1D, params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<f32>) {
        let mut current = input.clone();
        let mut temp_buf = Vec::new();
        for (layer, slice) in self.layers.iter_mut().zip(&self.slices) {
            let output_dim = layer.layer_info().output_dim;
            if output_dim == 0 {
                temp_buf.resize(current.len(), 0.0);
            } else {
                temp_buf.resize(output_dim, 0.0);
            }
            layer.forward_into(&current, params, slice, &mut temp_buf);
            current = Tensor1D::new(temp_buf.clone());
        }
        out_buf.clear();
        out_buf.extend_from_slice(&current.data);
    }

    fn backward(&mut self, delta: &Tensor1D, params: &[f32], _slice: &ParamSlice) -> Tensor1D {
        let mut d = delta.clone();
        for (layer, slice) in self.layers.iter_mut().rev().zip(self.slices.iter().rev()) {
            d = layer.backward(&d, params, slice);
        }
        d
    }

    fn apply_gradients(&mut self, store: &mut ParamStore, lr: f32, _slice: &ParamSlice) {
        for (layer, slice) in self.layers.iter_mut().zip(&self.slices) {
            layer.apply_gradients(store, lr, slice);
        }
    }

    fn param_len(&self) -> usize {
        self.slices.iter().map(|s| s.len).sum()
    }

    fn layer_info(&self) -> LayerInfo {
        let first = self.layers.first().map(|l| l.layer_info());
        let last = self.layers.last().map(|l| l.layer_info());
        LayerInfo {
            layer_type: "Sequential".to_string(),
            input_dim: first.map(|i| i.input_dim).unwrap_or(0),
            output_dim: last.map(|i| i.output_dim).unwrap_or(0),
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}
// src/model_plan/sequential.rs

use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::context1d::{Layer, LayerContext1D, LayerInfo};

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
    fn input_dim1s(&self) -> Vec<usize> {
        self.layers.first().map_or(vec![], |l| l.input_dim1s())
    }
    fn output_dim1s(&self) -> Vec<usize> {
        self.layers.last().map_or(vec![], |l| l.output_dim1s())
    }

    fn forward_into(
        &self,
        inputs: &[Tensor2D],
        params: &[f32],
        _slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D> {
        assert_eq!(inputs.len(), 1);
        let mut current = vec![inputs[0].clone()];
        let mut all_ctxs = Vec::new();

        for (layer, slice) in self.layers.iter().zip(&self.slices) {
            let out_sizes = layer.output_dim1s();
            let mut temp_bufs: Vec<Vec<f32>> = out_sizes.iter().map(|&sz| vec![0.0; sz]).collect();
            let ctxs = layer.forward_into(&current, params, slice, &mut temp_bufs);
            current = temp_bufs.into_iter().map(|buf| Tensor2D::new(vec![buf])).collect();
            if let Some(ctx) = ctxs.into_iter().next() {
                all_ctxs.push(ctx);
            }
        }

        assert_eq!(out_bufs.len(), current.len());
        for (out_buf, tensor) in out_bufs.iter_mut().zip(current) {
            out_buf.copy_from_slice(&tensor.data[0]);
        }
        all_ctxs
    }

    fn backward(
        &self,
        _ctxs: &[LayerContext1D],
        _deltas: &[Tensor2D],
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Vec<Tensor2D>, Vec<f32>) {
        (vec![Tensor2D::zeros(1, 0)], vec![])
    }

    fn param_len(&self) -> usize {
        self.slices.iter().map(|s| s.len).sum()
    }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Sequential".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}
// ============================================================
// Файл: src/layers/layers2d/sequential2d.rs (исправленный)
// ============================================================
use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use super::{Layer2D, LayerContext};

pub struct Sequential2D {
    layers: Vec<Box<dyn Layer2D>>,
    slices: Vec<ParamSlice>,
}

impl Sequential2D {
    pub fn new(layers: Vec<Box<dyn Layer2D>>, slices: Vec<ParamSlice>) -> Self {
        assert_eq!(layers.len(), slices.len());
        Self { layers, slices }
    }

    fn total_param_len(&self) -> usize {
        self.layers.iter().map(|l| l.param_len()).sum()
    }
}

impl Layer2D for Sequential2D {
    fn forward_into(&self, input: &Tensor2D, params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<f32>>) -> LayerContext {
        let mut contexts = Vec::with_capacity(self.layers.len());
        let mut current = input.clone();
        let mut temp_buf = Vec::new();

        for (layer, slice) in self.layers.iter().zip(&self.slices) {
            let rows = current.rows;
            let cols = layer.out_features();
            temp_buf = vec![vec![0.0; cols]; rows];
            let ctx = layer.forward_into(&current, params, slice, &mut temp_buf);
            current = Tensor2D::new(temp_buf.clone());
            contexts.push(ctx);
        }

        *out_buf = current.data;
        LayerContext::Sequential2D { contexts }
    }

    fn backward(&self, ctx: &LayerContext, delta: &Tensor2D, params: &[f32], slice: &ParamSlice) -> (Tensor2D, Vec<f32>) {
        let contexts = match ctx {
            LayerContext::Sequential2D { contexts } => contexts,
            _ => panic!("Sequential2D: invalid context"),
        };
        assert_eq!(contexts.len(), self.layers.len());

        let mut d = delta.clone();
        let mut all_grads = Vec::with_capacity(self.total_param_len());

        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let slice = &self.slices[i];
            let ctx = &contexts[i];
            let (d_prev, mut grads) = layer.backward(ctx, &d, params, slice);
            d = d_prev;
            // Вставляем градиенты в начало, чтобы сохранить порядок слоёв
            all_grads.splice(0..0, grads);
        }

        (d, all_grads)
    }

    fn param_len(&self) -> usize {
        self.total_param_len()
    }

    fn in_features(&self) -> usize {
        self.layers.first().map(|l| l.in_features()).unwrap_or(0)
    }

    fn out_features(&self) -> usize {
        self.layers.last().map(|l| l.out_features()).unwrap_or(0)
    }
}
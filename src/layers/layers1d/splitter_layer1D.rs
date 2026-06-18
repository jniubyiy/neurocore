// src/layers/layers1d/splitter_layer1D.rs
use crate::tensor::Tensor1D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::types::Splitter;
use super::{Layer, LayerContext1D};

/// Слой-разделитель для 1D тензоров: принимает один тензор длины `n` и возвращает два тензора длиной `p` и `q`.
/// Реализует мультивариабельный трейт `Layer`.
pub struct SplitterLayer1D {
    pub n: usize,
    pub p: usize,
    pub q: usize,
}

impl SplitterLayer1D {
    pub fn new(n: usize, p: usize, q: usize) -> Self {
        Self { n, p, q }
    }
}

impl Layer for SplitterLayer1D {
    fn input_dim1s(&self) -> Vec<usize> {
        vec![self.n]
    }

    fn output_dim1s(&self) -> Vec<usize> {
        vec![self.p, self.q]
    }

    fn forward_into(
        &self,
        inputs: &[Tensor1D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 2);
        assert_eq!(inputs[0].dim1(), self.n);
        assert_eq!(out_bufs[0].len(), self.p);
        assert_eq!(out_bufs[1].len(), self.q);

        let params_slice = &params[slice.start..slice.start + self.param_len()];
        let mut split = Splitter::new(self.n, self.p, self.q);
        split.set_params(params_slice);

        let x = &inputs[0].data;
        let mut a = vec![0.0; self.p];
        let mut b = vec![0.0; self.q];
        let (_cached_x, pre_a, pre_b) = split.forward_sample(x, &mut a, &mut b);
        out_bufs[0].copy_from_slice(&a);
        out_bufs[1].copy_from_slice(&b);

        vec![LayerContext1D::Splitter {
            input: inputs[0].clone(),
            pre_a,
            pre_b,
        }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext1D],
        deltas: &[Tensor1D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor1D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].dim1(), self.p);
        assert_eq!(deltas[1].dim1(), self.q);

        let params_slice = &params[slice.start..slice.start + self.param_len()];
        let mut split = Splitter::new(self.n, self.p, self.q);
        split.set_params(params_slice);

        let ctx = &ctxs[0];
        let da = &deltas[0].data;
        let db = &deltas[1].data;
        let (dx, d_params) = split.backward_sample(
            da,
            db,
            &(ctx.input().data.clone(), ctx.pre_a().clone(), ctx.pre_b().clone()),
        );

        let dx_tensor = Tensor1D::new(dx);
        (vec![dx_tensor], d_params)
    }

    fn param_len(&self) -> usize {
        let split = Splitter::new(self.n, self.p, self.q);
        split.param_count()
    }

    fn layer_info(&self) -> super::LayerInfo {
        super::LayerInfo {
            layer_type: "Splitter".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}

// Вспомогательные методы доступа к полям контекста Splitter
impl LayerContext1D {
    pub fn input(&self) -> &Tensor1D {
        match self {
            LayerContext1D::Splitter { input, .. } => input,
            _ => panic!("not a Splitter context"),
        }
    }
    pub fn pre_a(&self) -> &Vec<f32> {
        match self {
            LayerContext1D::Splitter { pre_a, .. } => pre_a,
            _ => panic!("not a Splitter context"),
        }
    }
    pub fn pre_b(&self) -> &Vec<f32> {
        match self {
            LayerContext1D::Splitter { pre_b, .. } => pre_b,
            _ => panic!("not a Splitter context"),
        }
    }
}
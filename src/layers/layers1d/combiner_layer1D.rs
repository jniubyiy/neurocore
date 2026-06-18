// src/layers/layers1d/combiner_layer1D.rs
use crate::tensor::Tensor1D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::types::Combiner;
use super::{Layer, LayerContext1D};

/// Слой-объединитель для 1D тензоров: принимает два тензора одинаковой длины `n` и возвращает тензор длины `m`.
/// Реализует мультивариабельный трейт `Layer`.
pub struct CombinerLayer1D {
    pub n: usize,
    pub m: usize,
}

impl CombinerLayer1D {
    pub fn new(n: usize, m: usize) -> Self {
        Self { n, m }
    }
}

impl Layer for CombinerLayer1D {
    fn input_dim1s(&self) -> Vec<usize> {
        vec![self.n, self.n]
    }

    fn output_dim1s(&self) -> Vec<usize> {
        vec![self.m]
    }

    fn forward_into(
        &self,
        inputs: &[Tensor1D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D> {
        assert_eq!(inputs.len(), 2);
        assert_eq!(out_bufs.len(), 1);
        assert_eq!(inputs[0].dim1(), self.n);
        assert_eq!(inputs[1].dim1(), self.n);
        assert_eq!(out_bufs[0].len(), self.m);

        let params_slice = &params[slice.start..slice.start + self.param_len()];
        let mut comb = Combiner::new(self.n, self.m);
        comb.set_params(params_slice);

        let a = &inputs[0].data;
        let b = &inputs[1].data;
        let mut out = vec![0.0; self.m];
        let (_cached_a, _cached_b, pre_act) = comb.forward_sample(a, b, &mut out);
        out_bufs[0].copy_from_slice(&out);

        vec![LayerContext1D::Combiner {
            input_a: inputs[0].clone(),
            input_b: inputs[1].clone(),
            pre_act,
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
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].dim1(), self.m);

        let params_slice = &params[slice.start..slice.start + self.param_len()];
        let mut comb = Combiner::new(self.n, self.m);
        comb.set_params(params_slice);

        let ctx = &ctxs[0];
        let d_out = &deltas[0].data;
        let (da, db, d_params) = comb.backward_sample(
            d_out,
            &(ctx.input_a().data.clone(), ctx.input_b().data.clone(), ctx.pre_act().clone()),
        );

        let dx1 = Tensor1D::new(da);
        let dx2 = Tensor1D::new(db);
        (vec![dx1, dx2], d_params)
    }

    fn param_len(&self) -> usize {
        let comb = Combiner::new(self.n, self.m);
        comb.param_count()
    }

    fn layer_info(&self) -> super::LayerInfo {
        super::LayerInfo {
            layer_type: "Combiner".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}

// Вспомогательные методы доступа к полям контекста
impl LayerContext1D {
    pub fn input_a(&self) -> &Tensor1D {
        match self {
            LayerContext1D::Combiner { input_a, .. } => input_a,
            _ => panic!("not a Combiner context"),
        }
    }
    pub fn input_b(&self) -> &Tensor1D {
        match self {
            LayerContext1D::Combiner { input_b, .. } => input_b,
            _ => panic!("not a Combiner context"),
        }
    }
    pub fn pre_act(&self) -> &Vec<f32> {
        match self {
            LayerContext1D::Combiner { pre_act, .. } => pre_act,
            _ => panic!("not a Combiner context"),
        }
    }
}
use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;
use super::{Layer5D, LayerContext5D};
use crate::layers::layers4d::{Linear4D, Layer4D};

pub struct Linear5D { inner: Linear4D }

impl Linear5D {
    pub fn new(in_features: usize, out_features: usize) -> Self { Self { inner: Linear4D::new(in_features, out_features) } }
}

impl Layer5D for Linear5D {
    fn forward_into(&self, input: &Tensor5D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<Vec<f32>>>>>) -> LayerContext5D {
        let outer = input.outer;
        let mut contexts = Vec::with_capacity(outer);
        for o in 0..outer {
            let slice_4d = input.slice_4d(o);
            let ctx = self.inner.forward_into(&slice_4d, params, slice, &mut out_buf[o]);
            contexts.push(ctx);
        }
        LayerContext5D::Linear5D { contexts }
    }

    fn backward(&self, ctx: &LayerContext5D, delta: &Tensor5D, params: &[f32], slice: &ParamSlice) -> (Tensor5D, Vec<f32>) {
        let contexts = match ctx { LayerContext5D::Linear5D { contexts } => contexts, _ => panic!() };
        let outer = delta.outer;
        let mut d_prev_data = Vec::with_capacity(outer);
        let mut total_grad = vec![0.0; self.param_len()];
        for o in 0..outer {
            let delta_4d = crate::tensor::Tensor4D::new(delta.data[o].clone());
            let (d_prev_4d, grads) = self.inner.backward(&contexts[o], &delta_4d, params, slice);
            d_prev_data.push(d_prev_4d.data);
            for (i, g) in grads.iter().enumerate() { total_grad[i] += g; }
        }
        (Tensor5D::new(d_prev_data), total_grad)
    }

    fn param_len(&self) -> usize { self.inner.param_len() }
    fn in_features(&self) -> usize { self.inner.in_features() }
    fn out_features(&self) -> usize { self.inner.out_features() }
}






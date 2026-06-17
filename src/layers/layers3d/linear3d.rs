use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;
use super::{Layer3D, LayerContext3D};
use crate::layers::layers2d::{Linear2D, Layer2D};

pub struct Linear3D {
    inner: Linear2D,
}

impl Linear3D {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self { inner: Linear2D::new(in_features, out_features) }
    }
}

impl Layer3D for Linear3D {
    fn forward_into(&self, input: &Tensor3D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<f32>>>) -> LayerContext3D {
        let depth = input.depth;
        let mut contexts = Vec::with_capacity(depth);
        for d in 0..depth {
            let slice_2d = input.slice_2d(d);
            let ctx = self.inner.forward_into(&slice_2d, params, slice, &mut out_buf[d]);
            contexts.push(ctx);
        }
        LayerContext3D::Linear3D { contexts }
    }

    fn backward(&self, ctx: &LayerContext3D, delta: &Tensor3D, params: &[f32], slice: &ParamSlice) -> (Tensor3D, Vec<f32>) {
        let contexts = match ctx { LayerContext3D::Linear3D { contexts } => contexts, _ => panic!() };
        let depth = delta.depth;
        let mut d_prev_data = Vec::with_capacity(depth);
        let mut total_grad = vec![0.0; self.param_len()];
        for d in 0..depth {
            let delta_2d = crate::tensor::Tensor2D::new(delta.data[d].clone());
            let (d_prev_2d, grads) = self.inner.backward(&contexts[d], &delta_2d, params, slice);
            d_prev_data.push(d_prev_2d.data);
            for (i, g) in grads.iter().enumerate() { total_grad[i] += g; }
        }
        (Tensor3D::new(d_prev_data), total_grad)
    }

    fn param_len(&self) -> usize { self.inner.param_len() }
    fn in_features(&self) -> usize { self.inner.in_features() }
    fn out_features(&self) -> usize { self.inner.out_features() }
}






use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;
use super::{Layer4D, LayerContext4D};
use crate::layers::layers3d::{Linear3D, Layer3D};

pub struct Linear4D { inner: Linear3D }

impl Linear4D {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self { inner: Linear3D::new(in_features, out_features) }
    }
}

impl Layer4D for Linear4D {
    fn forward_into(&self, input: &Tensor4D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<f32>>>>) -> LayerContext4D {
        let dim1 = input.dim1;
        let mut contexts = Vec::with_capacity(dim1);
        for d in 0..dim1 {
            let slice_3d = input.slice_3d(d);
            let ctx = self.inner.forward_into(&slice_3d, params, slice, &mut out_buf[d]);
            contexts.push(ctx);
        }
        LayerContext4D::Linear4D { contexts }
    }

    fn backward(&self, ctx: &LayerContext4D, delta: &Tensor4D, params: &[f32], slice: &ParamSlice) -> (Tensor4D, Vec<f32>) {
        let contexts = match ctx { LayerContext4D::Linear4D { contexts } => contexts, _ => panic!() };
        let dim1 = delta.dim1;
        let mut d_prev_data = Vec::with_capacity(dim1);
        let mut total_grad = vec![0.0; self.param_len()];
        for d in 0..dim1 {
            let delta_3d = crate::tensor::Tensor3D::new(delta.data[d].clone());
            let (d_prev_3d, grads) = self.inner.backward(&contexts[d], &delta_3d, params, slice);
            d_prev_data.push(d_prev_3d.data);
            for (i, g) in grads.iter().enumerate() { total_grad[i] += g; }
        }
        (Tensor4D::new(d_prev_data), total_grad)
    }

    fn param_len(&self) -> usize { self.inner.param_len() }
    fn in_features(&self) -> usize { self.inner.in_features() }
    fn out_features(&self) -> usize { self.inner.out_features() }
}






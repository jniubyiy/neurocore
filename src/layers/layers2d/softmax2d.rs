use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer2D, LayerContext};

pub struct Softmax2D;

impl Softmax2D {
    pub fn new() -> Self { Self }
}

impl Layer2D for Softmax2D {
    fn input_dims(&self) -> Vec<usize> { vec![0] }
    fn output_dims(&self) -> Vec<usize> { vec![0] }

    fn forward_into(
        &self,
        inputs: &[Tensor2D],
        _params: &[f32],
        _slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<f32>>],
    ) -> Vec<LayerContext> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let mat = linalg::tensor2d_to_faer(input);
        let out = Softmax.forward_mat(&mat);
        let out_t = linalg::faer_to_tensor2d(&out);
        *out_buf = out_t.data.clone();
        vec![LayerContext::Softmax2D { output: out_t }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext],
        deltas: &[Tensor2D],
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Vec<Tensor2D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];
        let sm = match ctx { LayerContext::Softmax2D { output } => output, _ => panic!() };
        let dim1 = sm.dim1;
        let dim2 = sm.dim2;
        let mut dprev = vec![vec![0.0; dim2]; dim1];
        for i in 0..dim1 {
            for j in 0..dim2 {
                let mut sum = 0.0;
                for k in 0..dim2 {
                    let kron = if j == k { 1.0 } else { 0.0 };
                    sum += delta.data[i][k] * sm.data[i][k] * (kron - sm.data[i][j]);
                }
                dprev[i][j] = sum;
            }
        }
        (vec![Tensor2D::new(dprev)], vec![])
    }

    fn param_len(&self) -> usize { 0 }
}





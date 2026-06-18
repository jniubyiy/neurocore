use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::ReLU;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer5D, LayerContext5D};

pub struct ReLU5D;

impl ReLU5D {
    pub fn new() -> Self { Self }
}

impl Layer5D for ReLU5D {
    fn input_dims(&self) -> Vec<usize> { vec![0] }
    fn output_dims(&self) -> Vec<usize> { vec![0] }

    fn forward_into(
        &self,
        inputs: &[Tensor5D],
        _params: &[f32],
        _slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<Vec<Vec<f32>>>>>],
    ) -> Vec<LayerContext5D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let mat = linalg::tensor5d_to_faer(input);
        let out = ReLU.forward_mat(&mat);
        let out_t = linalg::faer_to_tensor5d(&out, input.dim1, input.dim2, input.dim3, input.dim4, input.dim5);
        *out_buf = out_t.data;
        vec![LayerContext5D::ReLU5D { input: input.clone() }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext5D],
        deltas: &[Tensor5D],
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Vec<Tensor5D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];
        let input = match ctx { LayerContext5D::ReLU5D { input } => input, _ => panic!() };
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let dim4 = input.dim4;
        let dim5 = input.dim5;
        let mut dprev = vec![vec![vec![vec![vec![0.0; dim5]; dim4]; dim3]; dim2]; dim1];
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        for m in 0..dim5 {
                            dprev[i][j][k][l][m] = delta.data[i][j][k][l][m] * if input.data[i][j][k][l][m] > 0.0 { 1.0 } else { 0.0 };
                        }
                    }
                }
            }
        }
        (vec![Tensor5D::new(dprev)], vec![])
    }

    fn param_len(&self) -> usize { 0 }
}






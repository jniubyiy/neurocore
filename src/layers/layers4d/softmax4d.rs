use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer4D, LayerContext4D};

pub struct Softmax4D;

impl Softmax4D {
    pub fn new() -> Self { Self }
}

impl Layer4D for Softmax4D {
    fn input_dims(&self) -> Vec<usize> { vec![0] }
    fn output_dims(&self) -> Vec<usize> { vec![0] }

    fn forward_into(
        &self,
        inputs: &[Tensor4D],
        _params: &[f32],
        _slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<Vec<f32>>>>],
    ) -> Vec<LayerContext4D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let mat = linalg::tensor4d_to_faer(input);
        let out = Softmax.forward_mat(&mat);
        let out_t = linalg::faer_to_tensor4d(&out, input.dim1, input.dim2, input.dim3, input.dim4);
        *out_buf = out_t.data.clone();
        vec![LayerContext4D::Softmax4D { output: out_t }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext4D],
        deltas: &[Tensor4D],
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Vec<Tensor4D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];
        let sm = match ctx { LayerContext4D::Softmax4D { output } => output, _ => panic!() };
        let dim1 = sm.dim1;
        let dim2 = sm.dim2;
        let dim3 = sm.dim3;
        let dim4 = sm.dim4;
        let mut dprev = vec![vec![vec![vec![0.0; dim4]; dim3]; dim2]; dim1];
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        let mut sum = 0.0;
                        for m in 0..dim4 {
                            let kron = if l == m { 1.0 } else { 0.0 };
                            sum += delta.data[i][j][k][m] * sm.data[i][j][k][m] * (kron - sm.data[i][j][k][l]);
                        }
                        dprev[i][j][k][l] = sum;
                    }
                }
            }
        }
        (vec![Tensor4D::new(dprev)], vec![])
    }

    fn param_len(&self) -> usize { 0 }
}



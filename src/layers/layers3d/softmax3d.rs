use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer3D, LayerContext3D};

pub struct Softmax3D;

impl Softmax3D {
    pub fn new() -> Self { Self }
}

impl Layer3D for Softmax3D {
    fn input_dims(&self) -> Vec<usize> { vec![0] }
    fn output_dims(&self) -> Vec<usize> { vec![0] }

    fn forward_into(
        &self,
        inputs: &[Tensor3D],
        _params: &[f32],
        _slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<f32>>>],
    ) -> Vec<LayerContext3D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let mat = linalg::tensor3d_to_faer(input);
        let out = Softmax.forward_mat(&mat);
        let out_t = linalg::faer_to_tensor3d(&out, input.dim1, input.dim2, input.dim3);
        *out_buf = out_t.data.clone();
        vec![LayerContext3D::Softmax3D { output: out_t }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext3D],
        deltas: &[Tensor3D],
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Vec<Tensor3D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];
        let sm = match ctx { LayerContext3D::Softmax3D { output } => output, _ => panic!() };
        let dim1 = sm.dim1;
        let dim2 = sm.dim2;
        let dim3 = sm.dim3;
        let mut dprev = vec![vec![vec![0.0; dim3]; dim2]; dim1];
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    let mut sum = 0.0;
                    for l in 0..dim3 {
                        let kron = if k == l { 1.0 } else { 0.0 };
                        sum += delta.data[i][j][l] * sm.data[i][j][l] * (kron - sm.data[i][j][k]);
                    }
                    dprev[i][j][k] = sum;
                }
            }
        }
        (vec![Tensor3D::new(dprev)], vec![])
    }

    fn param_len(&self) -> usize { 0 }
}






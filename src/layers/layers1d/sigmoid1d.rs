use crate::tensor::Tensor1D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Sigmoid;
use crate::neuron::base::Neuron;
use crate::linalg;
use crate::linalg::faer_to_tensor1d;
use super::{Layer, LayerContext1D, LayerInfo};

pub struct SigmoidLayer;

impl SigmoidLayer {
    pub fn new() -> Self { Self }
}

impl Layer for SigmoidLayer {
    fn input_dim1s(&self) -> Vec<usize> { vec![0] }
    fn output_dim1s(&self) -> Vec<usize> { vec![0] }

    fn forward_into(
        &self,
        inputs: &[Tensor1D],
        _params: &[f32],
        _slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let m = linalg::tensor1d_to_faer(input);
        let out = Sigmoid.forward_mat(&m);
        let t = faer_to_tensor1d(&out);
        out_buf.clone_from(&t.data);
        vec![LayerContext1D::Sigmoid { output: t }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext1D],
        deltas: &[Tensor1D],
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Vec<Tensor1D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];
        let output = match ctx {
            LayerContext1D::Sigmoid { output } => output,
            _ => panic!(),
        };
        let mut delta_prev = vec![0.0; output.dim1()];
        for i in 0..output.dim1() {
            let sig = output.data[i];
            delta_prev[i] = delta.data[i] * sig * (1.0 - sig);
        }
        (vec![Tensor1D::new(delta_prev)], vec![])
    }

    fn param_len(&self) -> usize { 0 }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Sigmoid".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: 0,
            param_start_index: None,
        }
    }
}




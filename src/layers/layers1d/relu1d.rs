use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::neuron::ReLU;
use crate::neuron::base::Neuron;
use crate::linalg;
use crate::linalg::faer_to_tensor1d;
use super::{Layer, LayerInfo};

pub struct ReLULayer {
    last_input: Option<Tensor1D>,
}

impl ReLULayer {
    pub fn new() -> Self { Self { last_input: None } }
}

impl Layer for ReLULayer {
    fn forward_into(&mut self, input: &Tensor1D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<f32>) {
        self.last_input = Some(input.clone());
        let m = linalg::tensor1d_to_faer(input);
        let out = ReLU.forward_mat(&m);
        out_buf.copy_from_slice(&faer_to_tensor1d(&out).data);
    }

    fn backward(&mut self, delta: &Tensor1D, _params: &[f32], _slice: &ParamSlice) -> Tensor1D {
        let input = self.last_input.take().expect("ReLU backward without forward");
        let mut delta_prev = vec![0.0; input.len()];
        for i in 0..input.len() {
            delta_prev[i] = delta.data[i] * if input.data[i] > 0.0 { 1.0 } else { 0.0 };
        }
        Tensor1D::new(delta_prev)
    }

    fn apply_gradients(&mut self, _store: &mut ParamStore, _lr: f32, _slice: &ParamSlice) {}
    fn param_len(&self) -> usize { 0 }
    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "ReLU".to_string(),
            input_dim: 0,
            output_dim: 0,
            param_count: 0,
            param_start_index: None,
        }
    }
}





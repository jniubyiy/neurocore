use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::neuron::Sigmoid;
use crate::neuron::base::Neuron;
use crate::linalg;
use crate::linalg::faer_to_tensor1d;
use super::{Layer, LayerInfo};

pub struct SigmoidLayer {
    last_output: Option<Tensor1D>,
}

impl SigmoidLayer {
    pub fn new() -> Self { Self { last_output: None } }
}

impl Layer for SigmoidLayer {
    fn forward_into(&mut self, input: &Tensor1D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<f32>) {
        let m = linalg::tensor1d_to_faer(input);
        let out = Sigmoid.forward_mat(&m);
        let t = faer_to_tensor1d(&out);
        self.last_output = Some(t.clone());
        out_buf.copy_from_slice(&t.data);
    }

    fn backward(&mut self, delta: &Tensor1D, _params: &[f32], _slice: &ParamSlice) -> Tensor1D {
        let output = self.last_output.take().expect("Sigmoid backward without forward");
        let mut delta_prev = vec![0.0; output.len()];
        for i in 0..output.len() {
            let sig = output.data[i];
            delta_prev[i] = delta.data[i] * sig * (1.0 - sig);
        }
        Tensor1D::new(delta_prev)
    }

    fn apply_gradients(&mut self, _store: &mut ParamStore, _lr: f32, _slice: &ParamSlice) {}
    fn param_len(&self) -> usize { 0 }
    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Sigmoid".to_string(),
            input_dim: 0,
            output_dim: 0,
            param_count: 0,
            param_start_index: None,
        }
    }
}





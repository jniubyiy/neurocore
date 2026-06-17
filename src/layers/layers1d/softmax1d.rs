use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use crate::linalg;
use crate::linalg::faer_to_tensor1d;
use super::{Layer, LayerInfo};

pub struct SoftmaxLayer {
    last_output: Option<Tensor1D>,
}

impl SoftmaxLayer {
    pub fn new() -> Self { Self { last_output: None } }
}

impl Layer for SoftmaxLayer {
    fn forward_into(&mut self, input: &Tensor1D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<f32>) {
        let m = linalg::tensor1d_to_faer(input);
        let out = Softmax.forward_mat(&m);
        let t = faer_to_tensor1d(&out);
        self.last_output = Some(t.clone());
        out_buf.copy_from_slice(&t.data);
    }

    fn backward(&mut self, delta: &Tensor1D, _params: &[f32], _slice: &ParamSlice) -> Tensor1D {
        let sm = self.last_output.take().expect("Softmax backward without forward");
        let n = sm.len();
        let mut delta_prev = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let kron = if i == j { 1.0 } else { 0.0 };
                sum += delta.data[j] * sm.data[j] * (kron - sm.data[i]);
            }
            delta_prev[i] = sum;
        }
        Tensor1D::new(delta_prev)
    }

    fn apply_gradients(&mut self, _store: &mut ParamStore, _lr: f32, _slice: &ParamSlice) {}
    fn param_len(&self) -> usize { 0 }
    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Softmax".to_string(),
            input_dim: 0,
            output_dim: 0,
            param_count: 0,
            param_start_index: None,
        }
    }
}





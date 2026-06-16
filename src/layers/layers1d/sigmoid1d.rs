use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::types::sigmoid::Sigmoid;
use crate::neuron::base::Neuron;
use super::{Layer, LayerInfo};

pub struct SigmoidLayer;

impl SigmoidLayer {
    pub fn new() -> Self { Self }
}

impl Layer for SigmoidLayer {
    fn forward(
        &self,
        input: &Tensor1D,
        j_input: &Jacobian,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Tensor1D, Jacobian) {
        Sigmoid.forward(input, j_input)
    }

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





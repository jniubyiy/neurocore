use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::model_plan::param_store::ParamSlice;
use super::{Layer, LayerInfo};

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    slices: Vec<ParamSlice>,
}

impl Sequential {
    pub fn new(layers: Vec<(Box<dyn Layer>, ParamSlice)>) -> Self {
        let (layers, slices): (Vec<_>, Vec<_>) = layers.into_iter().unzip();
        Self { layers, slices }
    }
}

impl Layer for Sequential {
    fn forward(
        &self,
        input: &Tensor1D,
        j_input: &Jacobian,
        params: &[f32],
        _slice: &ParamSlice, // игнорируем, берём из slices
    ) -> (Tensor1D, Jacobian) {
        let mut current_val = input.clone();
        let mut current_jac = j_input.clone();
        for (layer, slice) in self.layers.iter().zip(self.slices.iter()) {
            let (val, jac) = layer.forward(&current_val, &current_jac, params, slice);
            current_val = val;
            current_jac = jac;
        }
        (current_val, current_jac)
    }

    fn param_len(&self) -> usize {
        self.slices.iter().map(|s| s.len).sum()
    }

    fn layer_info(&self) -> LayerInfo {
        let infos: Vec<LayerInfo> = self.layers.iter().map(|l| l.layer_info()).collect();
        let first_input = infos.first().map(|i| i.input_dim).unwrap_or(0);
        let last_output = infos.last().map(|i| i.output_dim).unwrap_or(0);
        let total_params: usize = infos.iter().map(|i| i.param_count).sum();
        LayerInfo {
            layer_type: "Sequential".to_string(),
            input_dim: first_input,
            output_dim: last_output,
            param_count: total_params,
            param_start_index: None,
        }
    }
}





use crate::layers::Layer5D;
use crate::tensor::Tensor5D;
use crate::jacobian::Jacobian5D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::model_trait::Model5D;
use std::sync::Arc;

pub struct SingleModel5D {
    layers: Vec<Arc<dyn Layer5D + Send + Sync>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
}

impl SingleModel5D {
    pub fn new(
        layers: Vec<Arc<dyn Layer5D + Send + Sync>>,
        slices: Vec<ParamSlice>,
        store: ParamStore,
    ) -> Self {
        Self { layers, slices, store }
    }
}

impl Model5D for SingleModel5D {
    fn forward(&mut self, input: &Tensor5D, j_input: &Jacobian5D) -> (Tensor5D, Jacobian5D) {
        let mut current_val = input.clone();
        let mut current_jac = j_input.clone();
        let params = self.store.all_params();
        for (layer, slice) in self.layers.iter().zip(self.slices.iter()) {
            let (val, jac) = layer.forward_5d(&current_val, &current_jac, params, slice);
            current_val = val;
            current_jac = jac;
        }
        (current_val, current_jac)
    }

    fn update_params(&mut self, lr: f32, grad: &[f32]) {
        self.store.apply_gradient(lr, grad);
    }

    fn num_workers(&self) -> usize { 1 }
}
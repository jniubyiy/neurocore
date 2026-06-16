use crate::layers::Layer2D;
use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::model_trait::Model2D;
use std::sync::Arc;

pub struct SingleModel2D {
    layers: Vec<Arc<dyn Layer2D + Send + Sync>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
}

impl SingleModel2D {
    pub fn new(
        layers: Vec<Arc<dyn Layer2D + Send + Sync>>,
        slices: Vec<ParamSlice>,
        store: ParamStore,
    ) -> Self {
        Self { layers, slices, store }
    }
}

impl Model2D for SingleModel2D {
    fn forward(&mut self, input: &Tensor2D, j_input: &Jacobian2D) -> (Tensor2D, Jacobian2D) {
        let mut current_val = input.clone();
        let mut current_jac = j_input.clone();
        let params = self.store.all_params();
        for (layer, slice) in self.layers.iter().zip(self.slices.iter()) {
            let (val, jac) = layer.forward_2d(&current_val, &current_jac, params, slice);
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
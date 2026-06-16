use crate::layers::Layer;
use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::model_trait::Model1D;
use std::sync::Arc;

pub struct SingleModel1D {
    layers: Vec<Arc<dyn Layer + Send + Sync>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
}

impl SingleModel1D {
    pub fn new(
        layers: Vec<Arc<dyn Layer + Send + Sync>>,
        slices: Vec<ParamSlice>,
        store: ParamStore,
    ) -> Self {
        assert_eq!(layers.len(), slices.len());
        Self { layers, slices, store }
    }
}

impl Model1D for SingleModel1D {
    fn forward(&mut self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let mut current_val = input.clone();
        let mut current_jac = j_input.clone();
        let params = self.store.all_params();
        for (layer, slice) in self.layers.iter().zip(self.slices.iter()) {
            let (val, jac) = layer.forward(&current_val, &current_jac, params, slice);
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
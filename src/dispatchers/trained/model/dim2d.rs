use crate::layers::Layer2D;
use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::Model2D;
use std::sync::Arc;

pub struct TrainedModel2D {
    layers: Vec<Arc<dyn Layer2D + Send + Sync>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
    num_workers: usize,
}

impl TrainedModel2D {
    pub fn new(
        layers: Vec<Arc<dyn Layer2D + Send + Sync>>,
        slices: Vec<ParamSlice>,
        store: ParamStore,
        _num_threads: usize,
    ) -> Self {
        assert_eq!(layers.len(), slices.len());
        Self { layers, slices, store, num_workers: 1 }
    }
}

impl Model2D for TrainedModel2D {
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

    fn num_workers(&self) -> usize { self.num_workers }
}
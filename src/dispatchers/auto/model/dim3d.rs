use crate::layers::Layer3D;
use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::model_trait::Model3D;
use std::sync::Arc;

pub struct AutoModel3D {
    layers: Vec<Arc<dyn Layer3D + Send + Sync>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
    num_workers: usize,
}

impl AutoModel3D {
    pub fn new(
        layers: Vec<Arc<dyn Layer3D + Send + Sync>>,
        slices: Vec<ParamSlice>,
        store: ParamStore,
        num_threads: usize,
    ) -> Self {
        Self {
            layers,
            slices,
            store,
            num_workers: num_threads.max(1),
        }
    }
}

impl Model3D for AutoModel3D {
    fn forward(&mut self, input: &Tensor3D, j_input: &Jacobian3D) -> (Tensor3D, Jacobian3D) {
        let mut current_val = input.clone();
        let mut current_jac = j_input.clone();
        let params = self.store.all_params();
        for (layer, slice) in self.layers.iter().zip(self.slices.iter()) {
            let (val, jac) = layer.forward_3d(&current_val, &current_jac, params, slice);
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
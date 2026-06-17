use crate::layers::Layer;
use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::model_trait::Model1D;

pub struct TrainedModel1D {
    inner: crate::dispatchers::single::SingleModel1D,
    num_workers: usize,
}

impl TrainedModel1D {
    pub fn new(layers: Vec<Box<dyn Layer>>, slices: Vec<ParamSlice>, store: ParamStore, num_threads: usize) -> Self {
        Self {
            inner: crate::dispatchers::single::SingleModel1D::new(layers, slices, store),
            num_workers: num_threads.max(1),
        }
    }
}

impl Model1D for TrainedModel1D {
    fn forward(&mut self, input: &Tensor1D) -> Tensor1D { self.inner.forward(input) }
    fn backward(&mut self, delta: &Tensor1D) -> Tensor1D { self.inner.backward(delta) }
    fn update_params(&mut self, lr: f32) { self.inner.update_params(lr) }
    fn num_workers(&self) -> usize { self.num_workers }
}
// ============================================================
// Файл: src/dispatchers/single/model/dim4d.rs
// ============================================================
use crate::layers::{Layer4D, LayerContext4D};
use crate::tensor::Tensor4D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::model_trait::Model4D;
use std::cell::RefCell;

pub struct SingleModel4D {
    layers: Vec<Box<dyn Layer4D>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
    buffers: RefCell<Vec<Vec<Vec<Vec<Vec<f32>>>>>>,
}

impl SingleModel4D {
    pub fn new(layers: Vec<Box<dyn Layer4D>>, slices: Vec<ParamSlice>, store: ParamStore, typical_dim1: usize, typical_depth: usize, typical_rows: usize) -> Self {
        let buffers = layers.iter().map(|l| vec![vec![vec![vec![0.0; l.out_features()]; typical_rows]; typical_depth]; typical_dim1]).collect();
        Self {
            layers,
            slices,
            store,
            buffers: RefCell::new(buffers),
        }
    }
}

impl Model4D for SingleModel4D {
    fn forward(&self, input: &Tensor4D) -> (Tensor4D, Vec<Vec<LayerContext4D>>) {
        let params = self.store.all_params();
        let mut buffers = self.buffers.borrow_mut();
        let mut current_data: Vec<Vec<Vec<Vec<f32>>>> = input.data.clone();
        let mut current_dim1 = input.dim1;
        let mut current_depth = input.depth;
        let mut current_rows = input.rows;
        let mut contexts = Vec::new();

        for (i, (layer, slice)) in self.layers.iter().zip(&self.slices).enumerate() {
            let out_buf = &mut buffers[i];
            if out_buf.len() != current_dim1 || out_buf[0].len() != current_depth || out_buf[0][0].len() != current_rows || out_buf[0][0][0].len() != layer.out_features() {
                *out_buf = vec![vec![vec![vec![0.0; layer.out_features()]; current_rows]; current_depth]; current_dim1];
            }
            let ctx = layer.forward_into(
                &Tensor4D::new(current_data),
                params,
                slice,
                out_buf,
            );
            current_data = out_buf.clone();
            current_dim1 = out_buf.len();
            current_depth = out_buf[0].len();
            current_rows = out_buf[0][0].len();
            contexts.push(vec![ctx]);
        }

        let out_tensor = Tensor4D::new(buffers.last().unwrap().clone());
        (out_tensor, contexts)
    }

    fn backward(&self, contexts: &[Vec<LayerContext4D>], delta: &Tensor4D) -> (Tensor4D, Vec<Vec<f32>>) {
        let params = self.store.all_params();
        let mut d = delta.clone();
        let mut all_grads = Vec::new();
        for ((layer, slice), ctxs) in self.layers.iter().rev().zip(self.slices.iter().rev()).zip(contexts.iter().rev()) {
            let (d_prev, grads) = layer.backward(&ctxs[0], &d, params, slice);
            d = d_prev;
            all_grads.push(grads);
        }
        all_grads.reverse();
        (d, all_grads)
    }

    fn update_params(&mut self, lr: f32, all_grads: &[Vec<f32>]) {
        for ((_layer, slice), grads) in self.layers.iter().zip(&self.slices).zip(all_grads.iter()) {
            for (i, &g) in grads.iter().enumerate() {
                self.store.add_to_param(slice.start + i, -lr * g);
            }
        }
    }

    fn num_workers(&self) -> usize { 1 }
}
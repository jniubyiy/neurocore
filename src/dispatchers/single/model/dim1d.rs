use crate::layers::Layer;
use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::model_trait::Model1D;

pub struct SingleModel1D {
    layers: Vec<Box<dyn Layer>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
    buffers: Vec<Vec<f32>>,
}

impl SingleModel1D {
    pub fn new(layers: Vec<Box<dyn Layer>>, slices: Vec<ParamSlice>, store: ParamStore) -> Self {
        let buffers: Vec<Vec<f32>> = layers.iter().map(|l| {
            let info = l.layer_info();
            if info.output_dim > 0 {
                vec![0.0; info.output_dim]
            } else {
                Vec::new()
            }
        }).collect();
        Self { layers, slices, store, buffers }
    }
}

impl Model1D for SingleModel1D {
    fn forward(&mut self, input: &Tensor1D) -> Tensor1D {
        let params = self.store.all_params();
        let mut current_data: Vec<f32> = input.data.clone();

        for (i, (layer, slice)) in self.layers.iter_mut().zip(&self.slices).enumerate() {
            let out_buf = &mut self.buffers[i];
            if out_buf.is_empty() {
                let output_dim = layer.layer_info().output_dim;
                let new_len = if output_dim == 0 { current_data.len() } else { output_dim };
                *out_buf = vec![0.0; new_len];
            }
            layer.forward_into(
                &Tensor1D::new(current_data),
                params,
                slice,
                out_buf,
            );
            current_data = out_buf.clone();
        }

        Tensor1D::new(self.buffers.last().unwrap().clone())
    }

    fn backward(&mut self, delta: &Tensor1D) -> Tensor1D {
        let params = self.store.all_params();
        let mut d = delta.clone();
        for (layer, slice) in self.layers.iter_mut().rev().zip(self.slices.iter().rev()) {
            d = layer.backward(&d, params, slice);
        }
        d
    }

    fn update_params(&mut self, lr: f32) {
        for (layer, slice) in self.layers.iter_mut().zip(&self.slices) {
            layer.apply_gradients(&mut self.store, lr, slice);
        }
    }

    fn num_workers(&self) -> usize { 1 }
}
use crate::layers::Layer;
use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::{LayerInfo, Model1D};
use std::sync::Arc;
use std::thread;

pub struct AutoModel1D {
    layers: Vec<Arc<dyn Layer + Send + Sync>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
    num_workers: usize,
}

impl AutoModel1D {
    pub fn new(
        layers: Vec<Arc<dyn Layer + Send + Sync>>,
        slices: Vec<ParamSlice>,
        store: ParamStore,
        num_threads: usize,
    ) -> Self {
        assert_eq!(layers.len(), slices.len());
        let num_workers = num_threads.max(1);
        AutoModel1D { layers, slices, store, num_workers }
    }
}

impl Model1D for AutoModel1D {
    fn forward(&mut self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let dim = input.len();
        let total_params = j_input.num_params;          // <-- исправлено

        let infos: Vec<LayerInfo> = self.layers.iter().map(|l| {
            let info = l.layer_info();
            LayerInfo {
                id: 0,
                layer_type: crate::dispatchers::common::LayerType::Linear,
                in_features: info.input_dim,
                out_features: info.output_dim,
                total_rows: dim,
            }
        }).collect();

        let num_layers = self.layers.len();
        let mut val_bufs = Vec::with_capacity(num_layers + 1);
        let mut jac_bufs = Vec::with_capacity(num_layers + 1);
        val_bufs.push(input.clone());
        jac_bufs.push(j_input.clone());

        for i in 0..num_layers {
            let out_dim = infos[i].out_features;
            val_bufs.push(Tensor1D::zeros(out_dim));
            jac_bufs.push(Jacobian::new(out_dim, total_params));
        }

        let params = self.store.all_params();

        for (layer_idx, (layer, slice)) in self.layers.iter().zip(self.slices.iter()).enumerate() {
            let out_dim = infos[layer_idx].out_features;

            let n_tasks = self.num_workers.min(dim);
            let chunk_size = (dim + n_tasks - 1) / n_tasks;

            let in_val = val_bufs[layer_idx].clone();
            let in_jac = jac_bufs[layer_idx].clone();
            let in_val_arc = Arc::new(in_val);
            let in_jac_arc = Arc::new(in_jac);
            let layer_arc = Arc::clone(layer);
            let params_arc = Arc::new(params.to_vec());
            let slice = *slice;

            let mut handles = Vec::new();
            for task_idx in 0..n_tasks {
                let start = task_idx * chunk_size;
                let end = (start + chunk_size).min(dim);
                if start >= end { continue; }
                let in_val = Arc::clone(&in_val_arc);
                let in_jac = Arc::clone(&in_jac_arc);
                let layer = Arc::clone(&layer_arc);
                let params = Arc::clone(&params_arc);
                let slice = slice;
                let handle = thread::spawn(move || {
                    let chunk_len = end - start;
                    let mut out_part = vec![0.0f32; chunk_len];
                    let mut jac_part_flat = vec![0.0f32; chunk_len * total_params];
                    layer.execute_range(
                        &in_val, &in_jac,
                        &mut out_part, &mut jac_part_flat,
                        start, end, total_params,
                        &params, &slice,
                    );
                    (start, out_part, jac_part_flat)
                });
                handles.push((start, handle));
            }

            let mut out_data = vec![0.0f32; out_dim];
            let mut jac_data = vec![vec![0.0f32; total_params]; out_dim];
            for (_s, handle) in handles {
                let (start, out_part, jac_flat) = handle.join().unwrap();
                let len = out_part.len();
                out_data[start..start + len].copy_from_slice(&out_part);
                for (i, global_idx) in (start..start + len).enumerate() {
                    for p in 0..total_params {
                        jac_data[global_idx][p] = jac_flat[i * total_params + p];
                    }
                }
            }
            val_bufs[layer_idx + 1] = Tensor1D::new(out_data);
            jac_bufs[layer_idx + 1] = Jacobian {
                out_features: out_dim,
                num_params: total_params,
                data: jac_data,
            };
        }

        let out_val = val_bufs.pop().unwrap();
        let out_jac = jac_bufs.pop().unwrap();
        (out_val, out_jac)
    }

    fn update_params(&mut self, lr: f32, grad: &[f32]) {
        self.store.apply_gradient(lr, grad);
    }

    fn num_workers(&self) -> usize { self.num_workers }
}
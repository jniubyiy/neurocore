// ============================================================
// Файл: src/dispatchers/auto/model/dim2d.rs (обновлён для forward_into)
// ============================================================
use crate::layers::{Layer2D, LayerContext};
use crate::tensor::Tensor2D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::model_trait::Model2D;
use std::sync::Arc;
use std::thread;

pub struct AutoModel2D {
    layers: Arc<Vec<Box<dyn Layer2D>>>,
    slices: Arc<Vec<ParamSlice>>,
    store: ParamStore,
    num_threads: usize,
}

impl AutoModel2D {
    pub fn new(layers: Vec<Box<dyn Layer2D>>, slices: Vec<ParamSlice>, store: ParamStore, num_threads: usize) -> Self {
        assert_eq!(layers.len(), slices.len());
        Self {
            layers: Arc::new(layers),
            slices: Arc::new(slices),
            store,
            num_threads: num_threads.max(1),
        }
    }

    fn chunk_ranges(batch: usize, chunks: usize) -> Vec<(usize, usize)> {
        let size = (batch + chunks - 1) / chunks;
        (0..chunks).map(|i| {
            let start = i * size;
            let end = (start + size).min(batch);
            (start, end)
        }).collect()
    }
}

impl Model2D for AutoModel2D {
    fn forward(&self, input: &Tensor2D) -> (Tensor2D, Vec<Vec<LayerContext>>) {
        let batch = input.rows;
        let chunks = self.num_threads.min(batch);
        let ranges = Self::chunk_ranges(batch, chunks);
        let params = self.store.all_params().to_vec();
        let layers = Arc::clone(&self.layers);
        let slices = Arc::clone(&self.slices);
        let input_arc = Arc::new(input.clone());

        let handles: Vec<_> = ranges.into_iter().map(|(start, end)| {
            let input = Arc::clone(&input_arc);
            let layers = Arc::clone(&layers);
            let slices = Arc::clone(&slices);
            let params = params.clone();
            thread::spawn(move || {
                let chunk_in = Tensor2D::new(input.data[start..end].to_vec());
                let mut current = chunk_in;
                let mut ctxs = Vec::new();
                for (layer, slice) in layers.iter().zip(slices.iter()) {
                    let rows = current.rows;
                    let cols = layer.out_features();
                    let mut buf = vec![vec![0.0; cols]; rows];
                    let ctx = layer.forward_into(&current, &params, slice, &mut buf);
                    current = Tensor2D::new(buf);
                    ctxs.push(ctx);
                }
                (current, ctxs)
            })
        }).collect();

        let mut outputs = Vec::with_capacity(batch);
        let mut per_layer_ctxs: Vec<Vec<LayerContext>> = vec![Vec::new(); self.layers.len()];

        for handle in handles {
            let (out, ctxs) = handle.join().unwrap();
            outputs.extend(out.data);
            for (layer_idx, ctx) in ctxs.into_iter().enumerate() {
                per_layer_ctxs[layer_idx].push(ctx);
            }
        }

        (Tensor2D::new(outputs), per_layer_ctxs)
    }

    fn backward(&self, contexts: &[Vec<LayerContext>], delta: &Tensor2D) -> (Tensor2D, Vec<Vec<f32>>) {
        let params = self.store.all_params().to_vec();
        let layers = &self.layers;
        let slices = &self.slices;

        let chunk_sizes: Vec<usize> = contexts[0].iter().map(|ctx| ctx.rows()).collect();
        let total_rows: usize = chunk_sizes.iter().sum();
        assert_eq!(total_rows, delta.rows);

        let mut d_chunks = Vec::new();
        let mut offset = 0;
        for &size in &chunk_sizes {
            d_chunks.push(Tensor2D::new(delta.data[offset..offset+size].to_vec()));
            offset += size;
        }

        let mut d_prev_chunks = d_chunks.clone();
        let mut all_grads: Vec<Vec<f32>> = vec![Vec::new(); layers.len()];

        for (layer_idx, layer) in layers.iter().rev().enumerate() {
            let layer_grads = &mut all_grads[layers.len() - 1 - layer_idx];
            *layer_grads = vec![0.0; layer.param_len()];
            let slice = &slices[layers.len() - 1 - layer_idx];
            let ctxs = &contexts[layers.len() - 1 - layer_idx];
            let mut new_d_prev_chunks = Vec::new();

            for (i, ctx) in ctxs.iter().enumerate() {
                let (d_prev, grads) = layer.backward(ctx, &d_prev_chunks[i], &params, slice);
                new_d_prev_chunks.push(d_prev);
                for (j, g) in grads.iter().enumerate() {
                    layer_grads[j] += g;
                }
            }
            d_prev_chunks = new_d_prev_chunks;
        }

        let mut d_prev_flat = Vec::new();
        for chunk in d_prev_chunks {
            d_prev_flat.extend(chunk.data);
        }
        let delta_prev = Tensor2D::new(d_prev_flat);

        (delta_prev, all_grads)
    }

    fn update_params(&mut self, lr: f32, all_grads: &[Vec<f32>]) {
        for (layer_idx, grads) in all_grads.iter().enumerate() {
            let slice = &self.slices[layer_idx];
            for (i, &g) in grads.iter().enumerate() {
                self.store.add_to_param(slice.start + i, -lr * g);
            }
        }
    }

    fn num_workers(&self) -> usize { self.num_threads }
}
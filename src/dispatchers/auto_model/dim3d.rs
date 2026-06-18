// src/dispatchers/auto_model/dim3d.rs

use std::sync::{Arc, Mutex};

use crate::layers::{Layer3D, LayerContext3D};
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::tensor::Tensor3D;

/// Процессор группы 3D слоёв. Использует общий ParamStore только для чтения параметров.
pub struct Dim3Processor {
    pub layers: Vec<Box<dyn Layer3D>>,
    pub slices: Vec<ParamSlice>,
    store: Arc<Mutex<ParamStore>>,
}

impl Dim3Processor {
    pub fn new(
        layers: Vec<Box<dyn Layer3D>>,
        slices: Vec<ParamSlice>,
        store: Arc<Mutex<ParamStore>>,
    ) -> Self {
        assert_eq!(layers.len(), slices.len());
        Self { layers, slices, store }
    }

    // ---- Одиночный образец (совместимость) ----

    /// Прямой проход одного образца.
    pub fn forward(&self, input: &Tensor3D, params: &[f32]) -> (Tensor3D, Vec<LayerContext3D>) {
        let (outputs, ctxs) = self.forward_batch(&[input.clone()], params);
        (outputs.into_iter().next().unwrap(), ctxs.into_iter().next().unwrap())
    }

    /// Обратный проход одного образца.
    pub fn backward(
        &self,
        ctxs: &[LayerContext3D],
        delta: &Tensor3D,
        params: &[f32],
    ) -> (Tensor3D, Vec<f32>) {
        let (mut in_deltas, grad) = self.backward_batch(&[ctxs.to_vec()], &[delta.clone()], params);
        (in_deltas.remove(0), grad)
    }

    // ---- Пакетная обработка ----

    /// Прямой проход батча. Возвращает выходы и контексты для каждого образца.
    pub fn forward_batch(
        &self,
        inputs: &[Tensor3D],
        params: &[f32],
    ) -> (Vec<Tensor3D>, Vec<Vec<LayerContext3D>>) {
        let batch = inputs.len();
        let mut outputs = Vec::with_capacity(batch);
        let mut all_ctxs = Vec::with_capacity(batch);

        for input in inputs {
            let (out, ctxs) = self.forward_one(input, params);
            outputs.push(out);
            all_ctxs.push(ctxs);
        }

        (outputs, all_ctxs)
    }

    /// Обратный проход батча. Принимает для каждого образца его контексты и дельту.
    /// Возвращает градиенты по входам для каждого образца и суммарный градиент параметров.
    pub fn backward_batch(
        &self,
        ctxs: &[Vec<LayerContext3D>],
        deltas: &[Tensor3D],
        params: &[f32],
    ) -> (Vec<Tensor3D>, Vec<f32>) {
        let batch = ctxs.len();
        assert_eq!(deltas.len(), batch);

        let param_len = self.store.lock().unwrap().len();
        let mut total_grad = vec![0.0f32; param_len];
        let mut in_grads = Vec::with_capacity(batch);

        for i in 0..batch {
            let (in_grad, grad) = self.backward_one(&ctxs[i], &deltas[i], params);
            in_grads.push(in_grad);
            for (j, &g) in grad.iter().enumerate() {
                total_grad[j] += g;
            }
        }

        (in_grads, total_grad)
    }

    // ---- Внутренние методы для одного образца ----

    fn forward_one(&self, input: &Tensor3D, params: &[f32]) -> (Tensor3D, Vec<LayerContext3D>) {
        let mut current = vec![input.clone()];
        let mut ctxs = Vec::with_capacity(self.layers.len());

        for (layer, slice) in self.layers.iter().zip(&self.slices) {
            let out_sizes = layer.output_dims();
            let dim1 = current[0].dim1;
            let dim2 = current[0].dim2;
            let mut bufs: Vec<Vec<Vec<Vec<f32>>>> = out_sizes
                .iter()
                .map(|&sz| vec![vec![vec![0.0; sz]; dim2]; dim1])
                .collect();
            let layer_ctxs = layer.forward_into(&current, params, slice, &mut bufs);
            ctxs.push(layer_ctxs.into_iter().next().unwrap());
            current = bufs.into_iter().map(Tensor3D::new).collect();
        }

        let output = current.into_iter().next().unwrap();
        (output, ctxs)
    }

    fn backward_one(
        &self,
        ctxs: &[LayerContext3D],
        delta: &Tensor3D,
        params: &[f32],
    ) -> (Tensor3D, Vec<f32>) {
        let n = self.layers.len();
        assert_eq!(ctxs.len(), n);

        let mut current_delta = delta.clone();
        let mut grad = vec![0.0f32; self.store.lock().unwrap().len()];

        for i in (0..n).rev() {
            let layer = &self.layers[i];
            let slice = &self.slices[i];
            let ctx = &ctxs[i];
            let (in_deltas, param_grad) = layer.backward(&[ctx.clone()], &[current_delta], params, slice);
            for (j, &g) in param_grad.iter().enumerate() {
                grad[slice.start + j] += g;
            }
            current_delta = in_deltas.into_iter().next().unwrap();
        }

        (current_delta, grad)
    }
}
// src/dispatchers/auto_model/mixed.rs

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::dispatchers::common::hardware::CPU_INFO;
use crate::dispatchers::common::{CostModel, LayerInfo, LayerType, Scheduler, WorkerPool};
use crate::layers::layers1d::{Layer as Layer1D, LayerContext1D};
use crate::layers::{Layer2D, Layer3D, Layer4D, Layer5D, LayerContext, LayerContext3D, LayerContext4D, LayerContext5D};
use crate::model_plan::blueprint::{LayerBlueprint, LayerKind};
use crate::model_plan::dim::Dim;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use super::dim_change::{self, DynamicTensor};
use super::dim1d::Dim1Processor;
use super::dim2d::Dim2Processor;
use super::dim3d::Dim3Processor;
use super::dim4d::Dim4Processor;
use super::dim5d::Dim5Processor;

// ============================================================================
// Динамические контейнеры для контекстов и батчей
// ============================================================================

#[derive(Clone)]
pub enum DynamicContext {
    Ctx1D(LayerContext1D),
    Ctx2D(LayerContext),
    Ctx3D(LayerContext3D),
    Ctx4D(LayerContext4D),
    Ctx5D(LayerContext5D),
}

pub enum DynamicBatchTensor {
    Dim1(Vec<Tensor1D>),
    Dim2(Vec<Tensor2D>),
    Dim3(Vec<Tensor3D>),
    Dim4(Vec<Tensor4D>),
    Dim5(Vec<Tensor5D>),
}

// ============================================================================
// Сегменты модели
// ============================================================================

enum Segment {
    Processor1D(Arc<Dim1Processor>),
    Processor2D(Arc<Dim2Processor>),
    Processor3D(Arc<Dim3Processor>),
    Processor4D(Arc<Dim4Processor>),
    Processor5D(Arc<Dim5Processor>),
    Unsqueeze(usize),
    ReduceMean(usize),
}

// ============================================================================
// Основная модель
// ============================================================================

pub struct MixedModel {
    segments: Vec<Segment>,
    store: Arc<Mutex<ParamStore>>,
    pool: WorkerPool,
    scheduler: Mutex<Scheduler>,
    layer_infos: Vec<Vec<LayerInfo>>,
}

impl MixedModel {
    pub fn from_plan(blueprints: Vec<LayerBlueprint>, num_threads: usize) -> Result<Self, String> {
        let store = Arc::new(Mutex::new(ParamStore::new()));
        let mut segments: Vec<Segment> = Vec::new();
        let mut layer_infos: Vec<Vec<LayerInfo>> = Vec::new();

        let mut current_layers_1d: Vec<Box<dyn Layer1D>> = Vec::new();
        let mut current_slices_1d: Vec<ParamSlice> = Vec::new();

        let mut current_layers_2d: Vec<Box<dyn Layer2D>> = Vec::new();
        let mut current_slices_2d: Vec<ParamSlice> = Vec::new();

        let mut current_layers_3d: Vec<Box<dyn Layer3D>> = Vec::new();
        let mut current_slices_3d: Vec<ParamSlice> = Vec::new();

        let mut current_layers_4d: Vec<Box<dyn Layer4D>> = Vec::new();
        let mut current_slices_4d: Vec<ParamSlice> = Vec::new();

        let mut current_layers_5d: Vec<Box<dyn Layer5D>> = Vec::new();
        let mut current_slices_5d: Vec<ParamSlice> = Vec::new();

        let mut current_dim: Option<Dim> = None;

        macro_rules! finalize_current {
            () => {
                if let Some(dim) = current_dim.take() {
                    let (proc_segment, infos) = match dim {
                        Dim::Dim1 => {
                            let proc = Dim1Processor::new(
                                std::mem::take(&mut current_layers_1d),
                                std::mem::take(&mut current_slices_1d),
                                Arc::clone(&store),
                            );
                            let infos = proc.layers.iter().enumerate().map(|(i, layer)| {
                                LayerInfo {
                                    id: i,
                                    layer_type: LayerType::Linear,
                                    in_features: layer.input_dim1s()[0],
                                    out_features: layer.output_dim1s()[0],
                                    total_rows: 1,
                                }
                            }).collect();
                            (Segment::Processor1D(Arc::new(proc)), infos)
                        },
                        Dim::Dim2 => {
                            let proc = Dim2Processor::new(
                                std::mem::take(&mut current_layers_2d),
                                std::mem::take(&mut current_slices_2d),
                                Arc::clone(&store),
                            );
                            let infos = proc.layers.iter().enumerate().map(|(i, layer)| {
                                LayerInfo {
                                    id: i,
                                    layer_type: LayerType::Linear,
                                    in_features: layer.input_dims()[0],
                                    out_features: layer.output_dims()[0],
                                    total_rows: 1,
                                }
                            }).collect();
                            (Segment::Processor2D(Arc::new(proc)), infos)
                        },
                        Dim::Dim3 => {
                            let proc = Dim3Processor::new(
                                std::mem::take(&mut current_layers_3d),
                                std::mem::take(&mut current_slices_3d),
                                Arc::clone(&store),
                            );
                            let infos = proc.layers.iter().enumerate().map(|(i, layer)| {
                                LayerInfo {
                                    id: i,
                                    layer_type: LayerType::Linear,
                                    in_features: layer.input_dims()[0],
                                    out_features: layer.output_dims()[0],
                                    total_rows: 1,
                                }
                            }).collect();
                            (Segment::Processor3D(Arc::new(proc)), infos)
                        },
                        Dim::Dim4 => {
                            let proc = Dim4Processor::new(
                                std::mem::take(&mut current_layers_4d),
                                std::mem::take(&mut current_slices_4d),
                                Arc::clone(&store),
                            );
                            let infos = proc.layers.iter().enumerate().map(|(i, layer)| {
                                LayerInfo {
                                    id: i,
                                    layer_type: LayerType::Linear,
                                    in_features: layer.input_dims()[0],
                                    out_features: layer.output_dims()[0],
                                    total_rows: 1,
                                }
                            }).collect();
                            (Segment::Processor4D(Arc::new(proc)), infos)
                        },
                        Dim::Dim5 => {
                            let proc = Dim5Processor::new(
                                std::mem::take(&mut current_layers_5d),
                                std::mem::take(&mut current_slices_5d),
                                Arc::clone(&store),
                            );
                            let infos = proc.layers.iter().enumerate().map(|(i, layer)| {
                                LayerInfo {
                                    id: i,
                                    layer_type: LayerType::Linear,
                                    in_features: layer.input_dims()[0],
                                    out_features: layer.output_dims()[0],
                                    total_rows: 1,
                                }
                            }).collect();
                            (Segment::Processor5D(Arc::new(proc)), infos)
                        },
                    };
                    segments.push(proc_segment);
                    layer_infos.push(infos);
                }
            };
        }

        for bp in &blueprints {
            match &bp.kind {
                LayerKind::SplitterConnector | LayerKind::CombinerConnector => {
                    // Эти слои требуют графового выполнения, которое пока не реализовано
                    return Err("Модели с разветвлениями (SplitterConnector/CombinerConnector) пока не поддерживаются в MixedModel".into());
                }
                LayerKind::Unsqueeze(axis) => {
                    finalize_current!();
                    segments.push(Segment::Unsqueeze(*axis));
                }
                LayerKind::ReduceMean(axis) => {
                    finalize_current!();
                    segments.push(Segment::ReduceMean(*axis));
                }
                _ => {
                    let dim = bp.input_dim;
                    if Some(dim) != current_dim {
                        finalize_current!();
                        current_dim = Some(dim);
                    }

                    let mut store_lock = store.lock().unwrap();
                    match dim {
                        Dim::Dim1 => {
                            let layer = bp.create_layer_1d();
                            let slice = store_lock.allocate(bp.param_len());
                            current_layers_1d.push(layer);
                            current_slices_1d.push(slice);
                        },
                        Dim::Dim2 => {
                            let layer = bp.create_layer_2d();
                            let slice = store_lock.allocate(bp.param_len());
                            current_layers_2d.push(layer);
                            current_slices_2d.push(slice);
                        },
                        Dim::Dim3 => {
                            let layer = bp.create_layer_3d();
                            let slice = store_lock.allocate(bp.param_len());
                            current_layers_3d.push(layer);
                            current_slices_3d.push(slice);
                        },
                        Dim::Dim4 => {
                            let layer = bp.create_layer_4d();
                            let slice = store_lock.allocate(bp.param_len());
                            current_layers_4d.push(layer);
                            current_slices_4d.push(slice);
                        },
                        Dim::Dim5 => {
                            let layer = bp.create_layer_5d();
                            let slice = store_lock.allocate(bp.param_len());
                            current_layers_5d.push(layer);
                            current_slices_5d.push(slice);
                        },
                    }
                    drop(store_lock);
                }
            }
        }
        finalize_current!();

        let cost = CostModel::calibrate();
        let mut scheduler = Scheduler::new(cost, CPU_INFO.clone());
        scheduler.set_num_workers(num_threads.max(1));
        let pool = WorkerPool::new(num_threads.max(1));

        Ok(MixedModel {
            segments,
            store,
            pool,
            scheduler: Mutex::new(scheduler),
            layer_infos,
        })
    }

    // ========================================================================
    // Прямой проход
    // ========================================================================

    pub fn forward_batch(&self, batch: DynamicBatchTensor) -> (DynamicBatchTensor, Vec<Vec<DynamicContext>>) {
        let params = self.store.lock().unwrap().all_params().to_vec();
        let mut current: Vec<DynamicTensor> = match batch {
            DynamicBatchTensor::Dim1(v) => v.into_iter().map(DynamicTensor::Dim1).collect(),
            DynamicBatchTensor::Dim2(v) => v.into_iter().map(DynamicTensor::Dim2).collect(),
            DynamicBatchTensor::Dim3(v) => v.into_iter().map(DynamicTensor::Dim3).collect(),
            DynamicBatchTensor::Dim4(v) => v.into_iter().map(DynamicTensor::Dim4).collect(),
            DynamicBatchTensor::Dim5(v) => v.into_iter().map(DynamicTensor::Dim5).collect(),
        };

        let batch_size = current.len();
        let mut all_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_size];

        for (seg_idx, seg) in self.segments.iter().enumerate() {
            match seg {
                Segment::Unsqueeze(axis) => {
                    for sample in current.iter_mut() {
                        *sample = dim_change::unsqueeze(sample.clone(), *axis);
                    }
                }
                Segment::ReduceMean(axis) => {
                    for sample in current.iter_mut() {
                        *sample = dim_change::reduce_mean(sample.clone(), *axis);
                    }
                }
                Segment::Processor1D(proc) => {
                    let inputs: Vec<Tensor1D> = current.iter().map(|t| match t {
                        DynamicTensor::Dim1(t) => t.clone(),
                        _ => unreachable!(),
                    }).collect();
                    let infos = &self.layer_infos[seg_idx];
                    let chunk_sizes = self.scheduler.lock().unwrap().plan_forward(infos);
                    let start = Instant::now();
                    let (outputs, ctxs) = self.parallel_forward_1d(proc, &inputs, &params, &chunk_sizes);
                    let elapsed = start.elapsed().as_secs_f32();
                    self.scheduler.lock().unwrap().record_forward_time(infos, &chunk_sizes, elapsed);
                    current = outputs.into_iter().map(DynamicTensor::Dim1).collect();
                    for (i, sample_ctxs) in ctxs.into_iter().enumerate() {
                        all_ctxs[i].extend(sample_ctxs.into_iter().map(DynamicContext::Ctx1D));
                    }
                }
                Segment::Processor2D(proc) => {
                    let inputs: Vec<Tensor2D> = current.iter().map(|t| match t {
                        DynamicTensor::Dim2(t) => t.clone(),
                        _ => unreachable!(),
                    }).collect();
                    let infos = &self.layer_infos[seg_idx];
                    let chunk_sizes = self.scheduler.lock().unwrap().plan_forward(infos);
                    let start = Instant::now();
                    let (outputs, ctxs) = self.parallel_forward_2d(proc, &inputs, &params, &chunk_sizes);
                    let elapsed = start.elapsed().as_secs_f32();
                    self.scheduler.lock().unwrap().record_forward_time(infos, &chunk_sizes, elapsed);
                    current = outputs.into_iter().map(DynamicTensor::Dim2).collect();
                    for (i, sample_ctxs) in ctxs.into_iter().enumerate() {
                        all_ctxs[i].extend(sample_ctxs.into_iter().map(DynamicContext::Ctx2D));
                    }
                }
                Segment::Processor3D(proc) => {
                    let inputs: Vec<Tensor3D> = current.iter().map(|t| match t {
                        DynamicTensor::Dim3(t) => t.clone(),
                        _ => unreachable!(),
                    }).collect();
                    let infos = &self.layer_infos[seg_idx];
                    let chunk_sizes = self.scheduler.lock().unwrap().plan_forward(infos);
                    let start = Instant::now();
                    let (outputs, ctxs) = self.parallel_forward_3d(proc, &inputs, &params, &chunk_sizes);
                    let elapsed = start.elapsed().as_secs_f32();
                    self.scheduler.lock().unwrap().record_forward_time(infos, &chunk_sizes, elapsed);
                    current = outputs.into_iter().map(DynamicTensor::Dim3).collect();
                    for (i, sample_ctxs) in ctxs.into_iter().enumerate() {
                        all_ctxs[i].extend(sample_ctxs.into_iter().map(DynamicContext::Ctx3D));
                    }
                }
                Segment::Processor4D(proc) => {
                    let inputs: Vec<Tensor4D> = current.iter().map(|t| match t {
                        DynamicTensor::Dim4(t) => t.clone(),
                        _ => unreachable!(),
                    }).collect();
                    let infos = &self.layer_infos[seg_idx];
                    let chunk_sizes = self.scheduler.lock().unwrap().plan_forward(infos);
                    let start = Instant::now();
                    let (outputs, ctxs) = self.parallel_forward_4d(proc, &inputs, &params, &chunk_sizes);
                    let elapsed = start.elapsed().as_secs_f32();
                    self.scheduler.lock().unwrap().record_forward_time(infos, &chunk_sizes, elapsed);
                    current = outputs.into_iter().map(DynamicTensor::Dim4).collect();
                    for (i, sample_ctxs) in ctxs.into_iter().enumerate() {
                        all_ctxs[i].extend(sample_ctxs.into_iter().map(DynamicContext::Ctx4D));
                    }
                }
                Segment::Processor5D(proc) => {
                    let inputs: Vec<Tensor5D> = current.iter().map(|t| match t {
                        DynamicTensor::Dim5(t) => t.clone(),
                        _ => unreachable!(),
                    }).collect();
                    let infos = &self.layer_infos[seg_idx];
                    let chunk_sizes = self.scheduler.lock().unwrap().plan_forward(infos);
                    let start = Instant::now();
                    let (outputs, ctxs) = self.parallel_forward_5d(proc, &inputs, &params, &chunk_sizes);
                    let elapsed = start.elapsed().as_secs_f32();
                    self.scheduler.lock().unwrap().record_forward_time(infos, &chunk_sizes, elapsed);
                    current = outputs.into_iter().map(DynamicTensor::Dim5).collect();
                    for (i, sample_ctxs) in ctxs.into_iter().enumerate() {
                        all_ctxs[i].extend(sample_ctxs.into_iter().map(DynamicContext::Ctx5D));
                    }
                }
            }
        }

        let out_batch = match current.first().unwrap() {
            DynamicTensor::Dim1(_) => DynamicBatchTensor::Dim1(current.into_iter().map(|t| match t { DynamicTensor::Dim1(t) => t, _ => unreachable!() }).collect()),
            DynamicTensor::Dim2(_) => DynamicBatchTensor::Dim2(current.into_iter().map(|t| match t { DynamicTensor::Dim2(t) => t, _ => unreachable!() }).collect()),
            DynamicTensor::Dim3(_) => DynamicBatchTensor::Dim3(current.into_iter().map(|t| match t { DynamicTensor::Dim3(t) => t, _ => unreachable!() }).collect()),
            DynamicTensor::Dim4(_) => DynamicBatchTensor::Dim4(current.into_iter().map(|t| match t { DynamicTensor::Dim4(t) => t, _ => unreachable!() }).collect()),
            DynamicTensor::Dim5(_) => DynamicBatchTensor::Dim5(current.into_iter().map(|t| match t { DynamicTensor::Dim5(t) => t, _ => unreachable!() }).collect()),
        };

        (out_batch, all_ctxs)
    }

    fn parallel_forward_1d(
        &self,
        proc: &Arc<Dim1Processor>,
        inputs: &[Tensor1D],
        params: &[f32],
        chunk_sizes: &[usize],
    ) -> (Vec<Tensor1D>, Vec<Vec<LayerContext1D>>) {
        let batch = inputs.len();
        let mut outputs = vec![Tensor1D::zeros(0); batch];
        let mut ctxs: Vec<Vec<LayerContext1D>> = vec![Vec::new(); batch];
        let mut offset = 0;
        let mut handles = Vec::new();
        for &size in chunk_sizes {
            if size == 0 { continue; }
            let chunk_inputs = inputs[offset..offset+size].to_vec();
            let proc = Arc::clone(proc);
            let params = params.to_vec();
            let handle = std::thread::spawn(move || {
                proc.forward_batch(&chunk_inputs, &params)
            });
            handles.push((offset, handle));
            offset += size;
        }
        for (start, handle) in handles {
            let (chunk_out, chunk_ctxs) = handle.join().unwrap();
            for i in 0..chunk_out.len() {
                outputs[start + i] = chunk_out[i].clone();
                ctxs[start + i] = chunk_ctxs[i].clone();
            }
        }
        (outputs, ctxs)
    }

    fn parallel_forward_2d(
        &self,
        proc: &Arc<Dim2Processor>,
        inputs: &[Tensor2D],
        params: &[f32],
        chunk_sizes: &[usize],
    ) -> (Vec<Tensor2D>, Vec<Vec<LayerContext>>) {
        let batch = inputs.len();
        let mut outputs = vec![Tensor2D::zeros(0, 0); batch];
        let mut ctxs: Vec<Vec<LayerContext>> = vec![Vec::new(); batch];
        let mut offset = 0;
        let mut handles = Vec::new();
        for &size in chunk_sizes {
            if size == 0 { continue; }
            let chunk_inputs = inputs[offset..offset+size].to_vec();
            let proc = Arc::clone(proc);
            let params = params.to_vec();
            let handle = std::thread::spawn(move || {
                proc.forward_batch(&chunk_inputs, &params)
            });
            handles.push((offset, handle));
            offset += size;
        }
        for (start, handle) in handles {
            let (chunk_out, chunk_ctxs) = handle.join().unwrap();
            for i in 0..chunk_out.len() {
                outputs[start + i] = chunk_out[i].clone();
                ctxs[start + i] = chunk_ctxs[i].clone();
            }
        }
        (outputs, ctxs)
    }

    fn parallel_forward_3d(
        &self,
        proc: &Arc<Dim3Processor>,
        inputs: &[Tensor3D],
        params: &[f32],
        chunk_sizes: &[usize],
    ) -> (Vec<Tensor3D>, Vec<Vec<LayerContext3D>>) {
        let batch = inputs.len();
        let mut outputs = vec![Tensor3D::zeros(0, 0, 0); batch];
        let mut ctxs: Vec<Vec<LayerContext3D>> = vec![Vec::new(); batch];
        let mut offset = 0;
        let mut handles = Vec::new();
        for &size in chunk_sizes {
            if size == 0 { continue; }
            let chunk_inputs = inputs[offset..offset+size].to_vec();
            let proc = Arc::clone(proc);
            let params = params.to_vec();
            let handle = std::thread::spawn(move || {
                proc.forward_batch(&chunk_inputs, &params)
            });
            handles.push((offset, handle));
            offset += size;
        }
        for (start, handle) in handles {
            let (chunk_out, chunk_ctxs) = handle.join().unwrap();
            for i in 0..chunk_out.len() {
                outputs[start + i] = chunk_out[i].clone();
                ctxs[start + i] = chunk_ctxs[i].clone();
            }
        }
        (outputs, ctxs)
    }

    fn parallel_forward_4d(
        &self,
        proc: &Arc<Dim4Processor>,
        inputs: &[Tensor4D],
        params: &[f32],
        chunk_sizes: &[usize],
    ) -> (Vec<Tensor4D>, Vec<Vec<LayerContext4D>>) {
        let batch = inputs.len();
        let mut outputs = vec![Tensor4D::zeros(0, 0, 0, 0); batch];
        let mut ctxs: Vec<Vec<LayerContext4D>> = vec![Vec::new(); batch];
        let mut offset = 0;
        let mut handles = Vec::new();
        for &size in chunk_sizes {
            if size == 0 { continue; }
            let chunk_inputs = inputs[offset..offset+size].to_vec();
            let proc = Arc::clone(proc);
            let params = params.to_vec();
            let handle = std::thread::spawn(move || {
                proc.forward_batch(&chunk_inputs, &params)
            });
            handles.push((offset, handle));
            offset += size;
        }
        for (start, handle) in handles {
            let (chunk_out, chunk_ctxs) = handle.join().unwrap();
            for i in 0..chunk_out.len() {
                outputs[start + i] = chunk_out[i].clone();
                ctxs[start + i] = chunk_ctxs[i].clone();
            }
        }
        (outputs, ctxs)
    }

    fn parallel_forward_5d(
        &self,
        proc: &Arc<Dim5Processor>,
        inputs: &[Tensor5D],
        params: &[f32],
        chunk_sizes: &[usize],
    ) -> (Vec<Tensor5D>, Vec<Vec<LayerContext5D>>) {
        let batch = inputs.len();
        let mut outputs = vec![Tensor5D::zeros(0, 0, 0, 0, 0); batch];
        let mut ctxs: Vec<Vec<LayerContext5D>> = vec![Vec::new(); batch];
        let mut offset = 0;
        let mut handles = Vec::new();
        for &size in chunk_sizes {
            if size == 0 { continue; }
            let chunk_inputs = inputs[offset..offset+size].to_vec();
            let proc = Arc::clone(proc);
            let params = params.to_vec();
            let handle = std::thread::spawn(move || {
                proc.forward_batch(&chunk_inputs, &params)
            });
            handles.push((offset, handle));
            offset += size;
        }
        for (start, handle) in handles {
            let (chunk_out, chunk_ctxs) = handle.join().unwrap();
            for i in 0..chunk_out.len() {
                outputs[start + i] = chunk_out[i].clone();
                ctxs[start + i] = chunk_ctxs[i].clone();
            }
        }
        (outputs, ctxs)
    }

    pub fn forward(&self, input: DynamicTensor) -> (DynamicTensor, Vec<Vec<DynamicContext>>) {
        let batch = match input {
            DynamicTensor::Dim1(t) => DynamicBatchTensor::Dim1(vec![t]),
            DynamicTensor::Dim2(t) => DynamicBatchTensor::Dim2(vec![t]),
            DynamicTensor::Dim3(t) => DynamicBatchTensor::Dim3(vec![t]),
            DynamicTensor::Dim4(t) => DynamicBatchTensor::Dim4(vec![t]),
            DynamicTensor::Dim5(t) => DynamicBatchTensor::Dim5(vec![t]),
        };
        let (out_batch, ctxs) = self.forward_batch(batch);
        let out = match out_batch {
            DynamicBatchTensor::Dim1(mut v) => DynamicTensor::Dim1(v.remove(0)),
            DynamicBatchTensor::Dim2(mut v) => DynamicTensor::Dim2(v.remove(0)),
            DynamicBatchTensor::Dim3(mut v) => DynamicTensor::Dim3(v.remove(0)),
            DynamicBatchTensor::Dim4(mut v) => DynamicTensor::Dim4(v.remove(0)),
            DynamicBatchTensor::Dim5(mut v) => DynamicTensor::Dim5(v.remove(0)),
        };
        (out, ctxs)
    }

    // ========================================================================
    // Обратный проход
    // ========================================================================

    pub fn backward_batch(
        &self,
        contexts: &[Vec<DynamicContext>],
        delta: DynamicBatchTensor,
    ) -> (DynamicBatchTensor, Vec<f32>) {
        let batch_size = match &delta {
            DynamicBatchTensor::Dim1(v) => v.len(),
            DynamicBatchTensor::Dim2(v) => v.len(),
            DynamicBatchTensor::Dim3(v) => v.len(),
            DynamicBatchTensor::Dim4(v) => v.len(),
            DynamicBatchTensor::Dim5(v) => v.len(),
        };
        let params = self.store.lock().unwrap().all_params().to_vec();
        let param_len = params.len();
        let mut total_grad = vec![0.0f32; param_len];

        let mut deltas: Vec<DynamicTensor> = match delta {
            DynamicBatchTensor::Dim1(v) => v.into_iter().map(DynamicTensor::Dim1).collect(),
            DynamicBatchTensor::Dim2(v) => v.into_iter().map(DynamicTensor::Dim2).collect(),
            DynamicBatchTensor::Dim3(v) => v.into_iter().map(DynamicTensor::Dim3).collect(),
            DynamicBatchTensor::Dim4(v) => v.into_iter().map(DynamicTensor::Dim4).collect(),
            DynamicBatchTensor::Dim5(v) => v.into_iter().map(DynamicTensor::Dim5).collect(),
        };

        let total_context_len = contexts[0].len();
        let mut ctx_pos = total_context_len;

        for seg in self.segments.iter().rev() {
            match seg {
                Segment::Unsqueeze(axis) => {
                    for d in deltas.iter_mut() {
                        *d = dim_change::unsqueeze_backward(d.clone(), *axis);
                    }
                }
                Segment::ReduceMean(axis) => {
                    for d in deltas.iter_mut() {
                        *d = dim_change::reduce_mean_backward(d.clone(), *axis);
                    }
                }
                Segment::Processor1D(proc) => {
                    let num_layers = proc.layers.len();
                    let start = ctx_pos - num_layers;
                    let mut new_deltas = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let sample_ctxs: Vec<LayerContext1D> = contexts[i][start..ctx_pos]
                            .iter()
                            .map(|dc| match dc {
                                DynamicContext::Ctx1D(c) => c.clone(),
                                _ => panic!("Expected Ctx1D in backward"),
                            })
                            .collect();
                        let delta_t = match &deltas[i] {
                            DynamicTensor::Dim1(t) => t.clone(),
                            _ => panic!("Expected Dim1 delta"),
                        };
                        let (in_delta, grad) = proc.backward(&sample_ctxs, &delta_t, &params);
                        new_deltas.push(DynamicTensor::Dim1(in_delta));
                        for (idx, &g) in grad.iter().enumerate() {
                            total_grad[idx] += g;
                        }
                    }
                    deltas = new_deltas;
                    ctx_pos = start;
                }
                Segment::Processor2D(proc) => {
                    let num_layers = proc.layers.len();
                    let start = ctx_pos - num_layers;
                    let mut new_deltas = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let sample_ctxs: Vec<LayerContext> = contexts[i][start..ctx_pos]
                            .iter()
                            .map(|dc| match dc {
                                DynamicContext::Ctx2D(c) => c.clone(),
                                _ => panic!("Expected Ctx2D"),
                            })
                            .collect();
                        let delta_t = match &deltas[i] {
                            DynamicTensor::Dim2(t) => t.clone(),
                            _ => panic!("Expected Dim2 delta"),
                        };
                        let (in_delta, grad) = proc.backward(&sample_ctxs, &delta_t, &params);
                        new_deltas.push(DynamicTensor::Dim2(in_delta));
                        for (idx, &g) in grad.iter().enumerate() {
                            total_grad[idx] += g;
                        }
                    }
                    deltas = new_deltas;
                    ctx_pos = start;
                }
                Segment::Processor3D(proc) => {
                    let num_layers = proc.layers.len();
                    let start = ctx_pos - num_layers;
                    let mut new_deltas = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let sample_ctxs: Vec<LayerContext3D> = contexts[i][start..ctx_pos]
                            .iter()
                            .map(|dc| match dc {
                                DynamicContext::Ctx3D(c) => c.clone(),
                                _ => panic!("Expected Ctx3D"),
                            })
                            .collect();
                        let delta_t = match &deltas[i] {
                            DynamicTensor::Dim3(t) => t.clone(),
                            _ => panic!("Expected Dim3 delta"),
                        };
                        let (in_delta, grad) = proc.backward(&sample_ctxs, &delta_t, &params);
                        new_deltas.push(DynamicTensor::Dim3(in_delta));
                        for (idx, &g) in grad.iter().enumerate() {
                            total_grad[idx] += g;
                        }
                    }
                    deltas = new_deltas;
                    ctx_pos = start;
                }
                Segment::Processor4D(proc) => {
                    let num_layers = proc.layers.len();
                    let start = ctx_pos - num_layers;
                    let mut new_deltas = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let sample_ctxs: Vec<LayerContext4D> = contexts[i][start..ctx_pos]
                            .iter()
                            .map(|dc| match dc {
                                DynamicContext::Ctx4D(c) => c.clone(),
                                _ => panic!("Expected Ctx4D"),
                            })
                            .collect();
                        let delta_t = match &deltas[i] {
                            DynamicTensor::Dim4(t) => t.clone(),
                            _ => panic!("Expected Dim4 delta"),
                        };
                        let (in_delta, grad) = proc.backward(&sample_ctxs, &delta_t, &params);
                        new_deltas.push(DynamicTensor::Dim4(in_delta));
                        for (idx, &g) in grad.iter().enumerate() {
                            total_grad[idx] += g;
                        }
                    }
                    deltas = new_deltas;
                    ctx_pos = start;
                }
                Segment::Processor5D(proc) => {
                    let num_layers = proc.layers.len();
                    let start = ctx_pos - num_layers;
                    let mut new_deltas = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let sample_ctxs: Vec<LayerContext5D> = contexts[i][start..ctx_pos]
                            .iter()
                            .map(|dc| match dc {
                                DynamicContext::Ctx5D(c) => c.clone(),
                                _ => panic!("Expected Ctx5D"),
                            })
                            .collect();
                        let delta_t = match &deltas[i] {
                            DynamicTensor::Dim5(t) => t.clone(),
                            _ => panic!("Expected Dim5 delta"),
                        };
                        let (in_delta, grad) = proc.backward(&sample_ctxs, &delta_t, &params);
                        new_deltas.push(DynamicTensor::Dim5(in_delta));
                        for (idx, &g) in grad.iter().enumerate() {
                            total_grad[idx] += g;
                        }
                    }
                    deltas = new_deltas;
                    ctx_pos = start;
                }
            }
        }

        let out_batch = match deltas.first().unwrap() {
            DynamicTensor::Dim1(_) => DynamicBatchTensor::Dim1(deltas.into_iter().map(|d| match d { DynamicTensor::Dim1(t) => t, _ => unreachable!() }).collect()),
            DynamicTensor::Dim2(_) => DynamicBatchTensor::Dim2(deltas.into_iter().map(|d| match d { DynamicTensor::Dim2(t) => t, _ => unreachable!() }).collect()),
            DynamicTensor::Dim3(_) => DynamicBatchTensor::Dim3(deltas.into_iter().map(|d| match d { DynamicTensor::Dim3(t) => t, _ => unreachable!() }).collect()),
            DynamicTensor::Dim4(_) => DynamicBatchTensor::Dim4(deltas.into_iter().map(|d| match d { DynamicTensor::Dim4(t) => t, _ => unreachable!() }).collect()),
            DynamicTensor::Dim5(_) => DynamicBatchTensor::Dim5(deltas.into_iter().map(|d| match d { DynamicTensor::Dim5(t) => t, _ => unreachable!() }).collect()),
        };

        (out_batch, total_grad)
    }

    pub fn backward(
        &self,
        contexts: &[Vec<DynamicContext>],
        delta: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<f32>>) {
        let batch = match delta {
            DynamicTensor::Dim1(t) => DynamicBatchTensor::Dim1(vec![t]),
            DynamicTensor::Dim2(t) => DynamicBatchTensor::Dim2(vec![t]),
            DynamicTensor::Dim3(t) => DynamicBatchTensor::Dim3(vec![t]),
            DynamicTensor::Dim4(t) => DynamicBatchTensor::Dim4(vec![t]),
            DynamicTensor::Dim5(t) => DynamicBatchTensor::Dim5(vec![t]),
        };
        let (in_batch, grad) = self.backward_batch(contexts, batch);
        let in_tensor = match in_batch {
            DynamicBatchTensor::Dim1(mut v) => DynamicTensor::Dim1(v.remove(0)),
            DynamicBatchTensor::Dim2(mut v) => DynamicTensor::Dim2(v.remove(0)),
            DynamicBatchTensor::Dim3(mut v) => DynamicTensor::Dim3(v.remove(0)),
            DynamicBatchTensor::Dim4(mut v) => DynamicTensor::Dim4(v.remove(0)),
            DynamicBatchTensor::Dim5(mut v) => DynamicTensor::Dim5(v.remove(0)),
        };
        (in_tensor, vec![grad])
    }

    pub fn update_params(&mut self, lr: f32, all_grads: &[Vec<f32>]) {
        if let Some(grad) = all_grads.first() {
            self.store.lock().unwrap().apply_gradient(lr, grad);
        }
    }

    pub fn num_workers(&self) -> usize {
        self.scheduler.lock().unwrap().num_workers()
    }
}
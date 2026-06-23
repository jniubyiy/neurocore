// src/compute_manager/graph/builder.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::cpu::{CostModel, Scheduler, WorkerPool};
use crate::compute_manager::cpu::hardware::CPU_INFO;
use crate::compute_manager::cpu::scheduler::LayerInfo;
use crate::layers::UniversalLayer;
use crate::model_plan::layer_desc::LayerDesc;
use crate::model_plan::blueprint::LayerKind;
use crate::model_plan::param_store::{ParamSlice, ParamStore};

use super::model::MixedModel;
use super::types::Segment;

impl MixedModel {
    /// Создаёт экземпляр `MixedModel` из плана слоёв.
    pub fn from_plan(layers: Vec<LayerDesc>, num_threads: usize) -> Result<Self, String> {
        let store = Arc::new(Mutex::new(ParamStore::new()));
        let mut segments: Vec<Segment> = Vec::new();
        let mut layer_infos: Vec<Vec<LayerInfo>> = Vec::new();

        let mut current_layers: Vec<Box<dyn UniversalLayer>> = Vec::new();
        let mut current_slices: Vec<ParamSlice> = Vec::new();

        let mut active_ports: Option<Vec<usize>> = None;
        let mut current_branch: Option<usize> = None;
        let mut current_stream_indices: Option<Vec<usize>> = None;

        macro_rules! finalize_universal {
            () => {
                if !current_layers.is_empty() {
                    let infos: Vec<LayerInfo> = current_layers
                        .iter()
                        .enumerate()
                        .map(|(i, layer)| LayerInfo {
                            id: i,
                            layer_type: crate::compute_manager::cpu::scheduler::LayerType::Linear,
                            in_features: layer.input_features(),
                            out_features: layer.output_features(),
                            total_rows: 0,
                        })
                        .collect();

                    segments.push(Segment::UniversalProcessor(
                        Arc::new(std::mem::take(&mut current_layers)),
                        std::mem::take(&mut current_slices),
                        current_stream_indices.take(),
                    ));
                    layer_infos.push(infos);
                }
            };
        }

        for desc in &layers {
            match &desc.kind {
                LayerKind::SplitterConnector => {
                    finalize_universal!();
                    active_ports = Some(desc.out_features.clone());
                    current_branch = Some(0);
                    segments.push(Segment::SplitterConnector {
                        input_dim: desc.in_features[0],
                        output_dims: desc.out_features.clone(),
                    });
                }
                LayerKind::CombinerConnector => {
                    finalize_universal!();
                    active_ports = None;
                    current_branch = None;
                    segments.push(Segment::CombinerConnector {
                        input_dims: desc.in_features.clone(),
                        output_dim: desc.out_features[0],
                    });
                }
                LayerKind::Splitter => {
                    finalize_universal!();
                    // 1D Splitter больше не поддерживается — модель не должна доходить до этого блока,
                    // но если план содержит Splitter, сборка аварийно завершится.
                    panic!("Splitter layer is no longer supported via 1D plan. Use UniversalLayer or dynamic tensor methods.");
                }
                LayerKind::Combiner => {
                    finalize_universal!();
                    panic!("Combiner layer is no longer supported via 1D plan. Use UniversalLayer or dynamic tensor methods.");
                }
                LayerKind::Unsqueeze(target_dims) => {
                    finalize_universal!();
                    segments.push(Segment::Unsqueeze(target_dims.clone()));
                }
                LayerKind::ReduceMean(target_dims) => {
                    finalize_universal!();
                    segments.push(Segment::ReduceMean(target_dims.clone()));
                }
                _ => {
                    if current_stream_indices.is_none() {
                        let indices = if let Some(ref ports) = active_ports {
                            if let Some(ref mut branch) = current_branch {
                                if let Some(pos) = ports.iter().position(|&p| p == desc.in_features[0]) {
                                    *branch = pos;
                                }
                            } else {
                                if let Some(pos) = ports.iter().position(|&p| p == desc.in_features[0]) {
                                    current_branch = Some(pos);
                                } else {
                                    current_branch = Some(0);
                                }
                            }
                            Some(vec![current_branch.unwrap()])
                        } else {
                            None
                        };
                        current_stream_indices = indices;
                    }

                    let mut store_lock = store.lock().unwrap();
                    let layer = desc.create_universal_layer();
                    let slice = store_lock.allocate(desc.param_len());
                    current_layers.push(layer);
                    current_slices.push(slice);
                    drop(store_lock);
                }
            }
        }
        finalize_universal!();

        let cost = CostModel::calibrate();
        let mut scheduler = Scheduler::new(cost, CPU_INFO.clone());
        scheduler.set_num_workers(num_threads.max(1));
        let pool = Arc::new(WorkerPool::new(num_threads.max(1)));

        Ok(MixedModel {
            segments,
            store,
            pool,
            scheduler: Mutex::new(scheduler),
            layer_infos,
            flat_pred_buf: Mutex::new(Vec::new()),
            flat_target_buf: Mutex::new(Vec::new()),
        })
    }
}
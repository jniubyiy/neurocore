// src/compute_manager/graph_v2/builder_v2.rs
//
// Построение сегментов графа v2 из `Vec<LayerDesc>`.
//
// Логика группировки (перенесена из старого
// `compute_manager::graph::builder`):
//
//   * подряд идущие обычные слои (Linear, ReLU, Sigmoid, ...) собираются
//     в один `Universal`-сегмент. Все их параметры аллоцируются одним
//     `ParamStore::allocate_segment`, slices совпадают порядком со слоями.
//
//   * `Unsqueeze` / `ReduceMean` становятся отдельными `DimOp`-сегментами.
//
//   * `Splitter` / `Combiner` становятся `Connector`-сегментами с
//     обучаемыми параметрами.
//
//   * `SplitterConnector` / `CombinerConnector` НЕ создают сегментов.
//     Это «маркеры потоков», как в старом пути: SplitterConnector
//     переключает активную ветку (`current_branch`), CombinerConnector
//     не делает ничего. Оба влияют только на заполнение `stream_indices`
//     у последующих Universal/DimOp-сегментов.
//
//   * `stream_indices` для Universal/DimOp определяются по трекерам:
//       - `active_ports`   — размеры активных потоков (None — один поток);
//       - `current_branch` — индекс активной ветки;
//       - `current_stream_indices` — накопленный массив индексов для
//         текущего Universal/DimOp.

use std::sync::{Arc, Mutex};

use crate::compute_manager::jobs_v2::{ConnectorOpKind, DimOpKind};
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
use crate::model_plan::param_store::ParamStore;
use crate::plans::model_plan::blueprint::LayerKind;
use crate::plans::model_plan::layer_desc::LayerDesc;

use super::types_v2::{SegmentKindV2, SegmentV2};

/// Строит сегменты из описания слоёв.
///
/// `param_store` используется, чтобы аллоцировать параметры Universal- и
/// Connector-сегментов. `param_memory_kind` — где физически должны лежать
/// параметры (HostRam для CPU-only сборки).
///
/// # Замечание про `unused_assignments`
///
/// Внутри используется макрос `finalize_universal!()`, который в конце
/// каждой сегментации делает `seg_idx += 1`. После последнего вызова
/// макроса в функции значение `seg_idx` уже никем не читается — компилятор
/// справедливо предупреждает. `#[allow(unused_assignments)]` снимает
/// предупреждение, не меняя логику.
#[allow(unused_assignments)]
pub fn build_segments(
    layers_desc: &[LayerDesc],
    param_store: &Arc<Mutex<ParamStore>>,
    param_memory_kind: MemoryDeviceKind,
) -> Result<Vec<SegmentV2>, String> {
    let mut segments: Vec<SegmentV2> = Vec::new();
    let mut seg_idx = 0usize;

    // --- Накопители для текущего Universal-сегмента ---
    let mut cur_layers: Vec<Box<dyn crate::layers::UniversalLayer>> = Vec::new();
    let mut cur_layer_sizes: Vec<usize> = Vec::new();
    let mut cur_input_shape: Vec<usize> = Vec::new();
    let mut cur_output_shape: Vec<usize> = Vec::new();

    // --- Трекеры потоков (аналогично старому builder) ---
    //
    // `active_ports` — размеры активных потоков.
    //     None          — один поток (канонический случай, «нет ветвления»);
    //     Some([p, q])  — две ветки с размерами p и q;
    //     Some([p])     — одна активная ветка размера p.
    //
    // `current_branch` — индекс активной ветки в `active_ports`.
    //     None          — ветка не задана (Splitter только что запущен).
    //     Some(i)       — работаем с потоком `active_ports[i]`.
    //
    // `current_stream_indices` — накопленный массив индексов потоков
    // для текущего Universal/DimOp. `None` — работать с streams[0].
    let mut active_ports: Option<Vec<usize>> = None;
    let mut current_branch: Option<usize> = None;
    let mut current_stream_indices: Option<Vec<usize>> = None;

    // Замыкание «завершить текущий Universal-сегмент, если он непустой».
    macro_rules! finalize_universal {
        () => {
            if !cur_layers.is_empty() {
                let slices = {
                    let mut ps = param_store.lock().unwrap();
                    ps.allocate_segment(&cur_layer_sizes, param_memory_kind)
                };

                if slices.len() != cur_layers.len() {
                    return Err(format!(
                        "GraphV2::build_segments: slices count ({}) != layers count ({})",
                        slices.len(),
                        cur_layers.len()
                    ));
                }

                segments.push(SegmentV2 {
                    index: seg_idx,
                    kind: SegmentKindV2::Universal {
                        layers: Arc::new(std::mem::take(&mut cur_layers)),
                        slices,
                    },
                    input_shape: std::mem::take(&mut cur_input_shape),
                    output_shape: std::mem::take(&mut cur_output_shape),
                    stream_count: 1,
                    stream_indices: current_stream_indices.take(),
                });
                seg_idx += 1;
                cur_layer_sizes.clear();
            }
        };
    }

    for desc in layers_desc {
        match &desc.kind {
            // ------------------------------------------------------------------
            // SplitterConnector — маркер начала ветки, НЕ создаёт сегмент.
            //
            // Переключает `active_ports` / `current_branch` так, чтобы
            // последующие Universal/DimOp получили корректный
            // `stream_indices`.
            // ------------------------------------------------------------------
            LayerKind::SplitterConnector => {
                finalize_universal!();

                let dims: Vec<usize> = if !desc.output_shape.streams.is_empty() {
                    desc.output_shape.streams.clone()
                } else {
                    desc.input_shape.streams.clone()
                };

                if dims.len() == 1 {
                    if let Some(ref ports) = active_ports {
                        // Уже есть активные потоки. Ищем текущий по размеру.
                        if let Some(pos) = ports.iter().position(|&p| p == dims[0]) {
                            current_branch = Some(pos);
                        } else {
                            current_branch = Some(0);
                        }
                    } else {
                        // Первый SplitterConnector — стартуем одну активную
                        // «потоковую ветку» без явного Splitter.
                        active_ports = Some(vec![dims[0]]);
                        current_branch = Some(0);
                    }
                } else {
                    // Множественный маркер (обычно — общий SplitterConnector
                    // после Splitter): все ветки становятся активны, но ни
                    // одна не выбрана.
                    active_ports = Some(dims);
                    current_branch = None;
                }
            }

            // ------------------------------------------------------------------
            // CombinerConnector — маркер конца ветки, НЕ создаёт сегмент
            // и не меняет трекеры. Следующий Universal/DimOp получит тот
            // же stream_indices, что и раньше.
            // ------------------------------------------------------------------
            LayerKind::CombinerConnector => {
                // No-op. Как и в старом builder.
            }

            // ------------------------------------------------------------------
            // Splitter — сегмент Connector::Splitter.
            // ------------------------------------------------------------------
            LayerKind::Splitter => {
                finalize_universal!();

                if desc.input_shape.streams.len() != 1 {
                    return Err(format!(
                        "GraphV2::build_segments: Splitter expects exactly 1 input stream, got {}",
                        desc.input_shape.streams.len()
                    ));
                }
                if desc.output_shape.streams.len() != 2 {
                    return Err(format!(
                        "GraphV2::build_segments: Splitter expects exactly 2 output streams, got {}",
                        desc.output_shape.streams.len()
                    ));
                }

                let input_dim = desc.input_shape.streams[0];
                let output_dims = desc.output_shape.streams.clone();
                let param_len = desc.param_len();
                let slice = {
                    let mut ps = param_store.lock().unwrap();
                    let slices = ps.allocate_segment(&[param_len], param_memory_kind);
                    slices.into_iter().next().ok_or_else(|| {
                        "GraphV2::build_segments: Splitter param alloc failed".to_string()
                    })?
                };

                // После Splitter активны два потока размера output_dims,
                // current_branch = 0 (стартуем с первого).
                active_ports = Some(output_dims.clone());
                current_branch = Some(0);

                segments.push(SegmentV2 {
                    index: seg_idx,
                    kind: SegmentKindV2::Connector {
                        kind: ConnectorOpKind::Splitter {
                            input_dim,
                            output_dims: output_dims.clone(),
                            slice,
                        },
                    },
                    input_shape: desc.input_shape.streams.clone(),
                    output_shape: output_dims,
                    stream_count: 2,
                    stream_indices: None,
                });
                seg_idx += 1;
            }

            // ------------------------------------------------------------------
            // Combiner — сегмент Connector::Combiner.
            // ------------------------------------------------------------------
            LayerKind::Combiner => {
                finalize_universal!();

                if desc.input_shape.streams.len() != 2 {
                    return Err(format!(
                        "GraphV2::build_segments: Combiner expects exactly 2 input streams, got {}",
                        desc.input_shape.streams.len()
                    ));
                }
                if desc.output_shape.streams.len() != 1 {
                    return Err(format!(
                        "GraphV2::build_segments: Combiner expects exactly 1 output stream, got {}",
                        desc.output_shape.streams.len()
                    ));
                }

                let input_dim = desc.input_shape.streams[0];
                let output_dim = desc.output_shape.streams[0];
                let param_len = desc.param_len();
                let slice = {
                    let mut ps = param_store.lock().unwrap();
                    let slices = ps.allocate_segment(&[param_len], param_memory_kind);
                    slices.into_iter().next().ok_or_else(|| {
                        "GraphV2::build_segments: Combiner param alloc failed".to_string()
                    })?
                };

                // После Combiner активен один поток размера output_dim.
                active_ports = Some(vec![output_dim]);
                current_branch = None;

                segments.push(SegmentV2 {
                    index: seg_idx,
                    kind: SegmentKindV2::Connector {
                        kind: ConnectorOpKind::Combiner {
                            input_dim,
                            output_dim,
                            slice,
                        },
                    },
                    input_shape: desc.input_shape.streams.clone(),
                    output_shape: desc.output_shape.streams.clone(),
                    stream_count: 1,
                    stream_indices: None,
                });
                seg_idx += 1;
            }

            // ------------------------------------------------------------------
            // Unsqueeze / ReduceMean — отдельные DimOp-сегменты.
            // stream_indices наследуется от текущего трекера (без take()).
            // ------------------------------------------------------------------
            LayerKind::Unsqueeze => {
                finalize_universal!();
                let target_dims = desc.output_shape.streams.clone();

                segments.push(SegmentV2 {
                    index: seg_idx,
                    kind: SegmentKindV2::DimOp {
                        kind: DimOpKind::Unsqueeze(target_dims),
                    },
                    input_shape: desc.input_shape.streams.clone(),
                    output_shape: desc.output_shape.streams.clone(),
                    stream_count: 1,
                    stream_indices: current_stream_indices.clone(),
                });
                seg_idx += 1;
            }
            LayerKind::ReduceMean => {
                finalize_universal!();
                let target_dims = desc.output_shape.streams.clone();

                segments.push(SegmentV2 {
                    index: seg_idx,
                    kind: SegmentKindV2::DimOp {
                        kind: DimOpKind::ReduceMean(target_dims),
                    },
                    input_shape: desc.input_shape.streams.clone(),
                    output_shape: desc.output_shape.streams.clone(),
                    stream_count: 1,
                    stream_indices: current_stream_indices.clone(),
                });
                seg_idx += 1;
            }

            // ------------------------------------------------------------------
            // Обычный слой — накапливается в текущий Universal.
            // stream_indices вычисляется только в начале сегмента.
            // ------------------------------------------------------------------
            _ => {
                if cur_layers.is_empty() {
                    cur_input_shape = desc.input_shape.streams.clone();

                    // Вычисляем stream_indices для нового Universal по трекерам.
                    let indices = if let Some(ref ports) = active_ports {
                        let in_dim = desc.input_shape.streams[0];

                        if let Some(ref mut branch) = current_branch {
                            // Ветка уже выбрана. Уточняем её по размеру
                            // входного потока, если он совпал с одним из
                            // активных портов (полезно при переключении
                            // между ветками без явного SplitterConnector).
                            if let Some(pos) =
                                ports.iter().position(|&p| p == in_dim)
                            {
                                *branch = pos;
                            }
                        } else if let Some(pos) =
                            ports.iter().position(|&p| p == in_dim)
                        {
                            current_branch = Some(pos);
                        } else {
                            current_branch = Some(0);
                        }

                        Some(vec![current_branch.unwrap()])
                    } else {
                        None
                    };

                    current_stream_indices = indices;
                }

                let layer = desc.create_universal_layer();
                cur_layer_sizes.push(desc.param_len());
                cur_layers.push(layer);
                cur_output_shape = desc.output_shape.streams.clone();
            }
        }
    }

    finalize_universal!();

    Ok(segments)
}
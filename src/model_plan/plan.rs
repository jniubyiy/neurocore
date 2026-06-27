// src/model_plan/plan.rs

use super::layer_desc::LayerDesc;
use super::blueprint::LayerKind;
use crate::compute_manager::device::Device;
use crate::compute_manager::graph::model::MixedModel;

#[derive(Debug, Clone)]
pub struct Plan {
    pub(crate) layers: Vec<LayerDesc>,
}

impl Plan {
    pub(crate) fn from_descs(descs: Vec<LayerDesc>) -> Result<Self, String> {
        if descs.is_empty() {
            return Err("План не может быть пустым".into());
        }

        for (i, desc) in descs.iter().enumerate() {
            match desc.kind {
                LayerKind::SplitterConnector | LayerKind::CombinerConnector
                | LayerKind::Splitter | LayerKind::Combiner
                | LayerKind::Unsqueeze(_) | LayerKind::ReduceMean(_) => {
                    // Эти слои могут иметь множественные in_features/out_features
                }
                _ => {
                    if desc.in_features.len() != 1 || desc.out_features.len() != 1 {
                        return Err(format!(
                            "Слой {} ({} → {}) имеет нестандартное число входов/выходов. \
                             Мультивариабельные слои пока не поддерживаются.",
                            i, desc.in_features.len(), desc.out_features.len()
                        ));
                    }
                    if desc.in_features[0] == 0 || desc.out_features[0] == 0 {
                        return Err(format!("Слой {}: размерности не могут быть нулевыми", i));
                    }
                }
            }
        }

        // Проверка совместимости размерностей
        for i in 1..descs.len() {
            let prev = &descs[i - 1];
            let curr = &descs[i];

            if matches!(prev.kind, LayerKind::Unsqueeze(_) | LayerKind::ReduceMean(_))
                || matches!(curr.kind, LayerKind::Unsqueeze(_) | LayerKind::ReduceMean(_))
            {
                continue;
            }

            let prev_is_splitter = matches!(prev.kind, LayerKind::SplitterConnector | LayerKind::Splitter);
            let curr_is_combiner = matches!(curr.kind, LayerKind::CombinerConnector | LayerKind::Combiner);

            if prev_is_splitter && curr_is_combiner {
                if prev.out_features != curr.in_features {
                    return Err(format!(
                        "Несовместимость размерностей между Splitter (слой {}) и Combiner (слой {}): выходы {:?} != входы {:?}",
                        i, i + 1, prev.out_features, curr.in_features
                    ));
                }
            } else if prev_is_splitter {
                if curr.in_features.len() == 1 {
                    if curr.in_features[0] != prev.out_features[0] {
                        return Err(format!(
                            "Несовместимость размеров между слоем {} (Splitter) и слоем {}: первый выход {} не совпадает с входом {}",
                            i, i + 1, prev.out_features[0], curr.in_features[0]
                        ));
                    }
                }
            } else if curr_is_combiner {
                if prev.out_features.len() == 1 {
                    if prev.out_features[0] != curr.in_features[0] {
                        return Err(format!(
                            "Несовместимость размеров между слоем {} и Combiner (слой {}): выход {} не совпадает с первым входом {}",
                            i, i + 1, prev.out_features[0], curr.in_features[0]
                        ));
                    }
                }
            } else {
                if prev.out_features.len() == 1 && curr.in_features.len() == 1 {
                    let prev_out = prev.out_features[0];
                    let curr_in = curr.in_features[0];
                    if prev_out != curr_in {
                        return Err(format!(
                            "Несовместимость размеров последней оси между слоем {} (выход {}) и слоем {} (вход {})",
                            i, prev_out, i + 1, curr_in
                        ));
                    }
                }
            }
        }

        for (i, desc) in descs.iter().enumerate() {
            if desc.kind == LayerKind::Softmax && i != descs.len() - 1 {
                return Err("Softmax допускается только на последнем слое".into());
            }
        }

        Ok(Plan { layers: descs })
    }

    pub fn from_layer_descs(descs: Vec<LayerDesc>) -> Result<Plan, String> {
        Plan::from_descs(descs)
    }

    /// Собрать модель с указанным количеством потоков CPU (обратная совместимость).
    pub fn build_with_threads(&self, threads: usize) -> MixedModel {
        MixedModel::from_plan(self.layers.clone(), threads)
            .expect("Plan уже проверен")
    }

    /// Собрать модель на CPU с одним потоком (по умолчанию).
    pub fn build(&self) -> MixedModel {
        self.build_with_device(Device::Cpu { threads: 1 })
    }

    /// Собрать модель, явно указав целевое устройство.
    /// Для `Device::Cpu` число потоков берётся из описания устройства.
    /// Для `Device::Gpu` используется GPU с заданным индексом.
    pub fn build_with_device(&self, device: Device) -> MixedModel {
        MixedModel::from_plan_with_device(self.layers.clone(), 1, device)
            .expect("Plan уже проверен")
    }

    pub fn input_dim1(&self) -> usize {
        self.layers.first().unwrap().in_features[0]
    }

    pub fn output_dim1(&self) -> usize {
        self.layers.last().unwrap().out_features[0]
    }
}
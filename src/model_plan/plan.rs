// src/model_plan/plan.rs

use super::layer_desc::LayerDesc;
use super::blueprint::LayerKind;
use crate::compute_manager::graph::model::MixedModel;

/// План модели, готовый к сборке.
#[derive(Debug, Clone)]
pub struct Plan {
    pub(crate) layers: Vec<LayerDesc>,
}

impl Plan {
    /// Внутренний конструктор, проверяет корректность архитектуры.
    pub(crate) fn from_descs(descs: Vec<LayerDesc>) -> Result<Self, String> {
        if descs.is_empty() {
            return Err("План не может быть пустым".into());
        }

        // Проверка базовых размерностей для каждого слоя
        for (i, desc) in descs.iter().enumerate() {
            match desc.kind {
                LayerKind::SplitterConnector | LayerKind::CombinerConnector
                | LayerKind::Splitter | LayerKind::Combiner => {
                    if desc.in_features.is_empty() || desc.out_features.is_empty() {
                        return Err(format!("Слой {}: размерности не могут быть пустыми", i));
                    }
                    if matches!(desc.kind, LayerKind::SplitterConnector) {
                        let input_size = desc.in_features.iter().sum::<usize>();
                        let output_sum: usize = desc.out_features.iter().sum();
                        if input_size != output_sum {
                            return Err(format!(
                                "Слой {}: SplitterConnector: сумма выходных размеров ({}) должна равняться входному ({})",
                                i, output_sum, input_size
                            ));
                        }
                    } else if matches!(desc.kind, LayerKind::CombinerConnector) {
                        let output_size = desc.out_features.iter().sum::<usize>();
                        let input_sum: usize = desc.in_features.iter().sum();
                        if output_size != input_sum {
                            return Err(format!(
                                "Слой {}: CombinerConnector: сумма входных размеров ({}) должна равняться выходному ({})",
                                i, input_sum, output_size
                            ));
                        }
                    } else if matches!(desc.kind, LayerKind::Splitter) {
                        if desc.in_features.len() != 1 || desc.out_features.len() < 2 {
                            return Err(format!(
                                "Слой {}: Splitter требует 1 вход и минимум 2 выхода", i
                            ));
                        }
                    } else if matches!(desc.kind, LayerKind::Combiner) {
                        if desc.in_features.len() < 2 || desc.out_features.len() != 1 {
                            return Err(format!(
                                "Слой {}: Combiner требует минимум 2 входа и 1 выход", i
                            ));
                        }
                    }
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

        // Проверка совместимости размерностей и размеров последней оси между соседними слоями
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

    /// Публичная обёртка для создания плана из описаний слоёв.
    pub fn from_layer_descs(descs: Vec<LayerDesc>) -> Result<Plan, String> {
        Plan::from_descs(descs)
    }

    /// Собирает модель с указанным количеством потоков.
    pub fn build_with_threads(&self, threads: usize) -> MixedModel {
        MixedModel::from_plan(self.layers.clone(), threads)
            .expect("Plan уже проверен")
    }

    /// Собирает модель с одним потоком (обратная совместимость).
    pub fn build(&self) -> MixedModel {
        self.build_with_threads(1)
    }

    /// Размерность входа (первый слой).
    pub fn input_dim1(&self) -> usize {
        self.layers.first().unwrap().in_features[0]
    }

    /// Размерность выхода (последний слой).
    pub fn output_dim1(&self) -> usize {
        self.layers.last().unwrap().out_features[0]
    }
}
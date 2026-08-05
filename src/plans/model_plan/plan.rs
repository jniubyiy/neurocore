// src/plans/model_plan/plan.rs

use super::layer_desc::LayerDesc;
use super::blueprint::LayerKind;
use crate::compute_manager::device::Device;
use crate::device_plan::DevicePlan;
use crate::compute_manager::graph::model::MixedModel;

#[derive(Debug, Clone)]
pub struct Plan {
    pub(crate) layers: Vec<LayerDesc>,
}

impl Plan {
    /// Создаёт план из списка описаний слоёв.
    pub fn from_layer_descs(descs: Vec<LayerDesc>) -> Result<Plan, String> {
        Plan::from_descs(descs)
    }

    /// Внутренний метод проверки и создания плана.
    pub(crate) fn from_descs(descs: Vec<LayerDesc>) -> Result<Self, String> {
        if descs.is_empty() {
            return Err("План не может быть пустым".into());
        }

        // 1. Проверка каждого слоя по отдельности
        for (i, desc) in descs.iter().enumerate() {
            match desc.kind {
                // Слои, допускающие множественные входы/выходы – без дополнительных ограничений
                LayerKind::SplitterConnector
                | LayerKind::CombinerConnector
                | LayerKind::Splitter
                | LayerKind::Combiner
                | LayerKind::Unsqueeze
                | LayerKind::ReduceMean => {}

                // Обычные слои: теперь разрешено произвольное количество осей (потоков),
                // они интерпретируются как размерности одного тензора.
                _ => {
                    // Проверяем только, что размеры не нулевые
                    if desc.input_shape.streams.is_empty() {
                        return Err(format!(
                            "Слой {}: входной тензор должен иметь хотя бы одну размерность", i
                        ));
                    }
                    if desc.output_shape.streams.is_empty() {
                        return Err(format!(
                            "Слой {}: выходной тензор должен иметь хотя бы одну размерность", i
                        ));
                    }
                    for (j, &dim) in desc.input_shape.streams.iter().enumerate() {
                        if dim == 0 {
                            return Err(format!(
                                "Слой {}: входная размерность {} равна нулю", i, j
                            ));
                        }
                    }
                    for (j, &dim) in desc.output_shape.streams.iter().enumerate() {
                        if dim == 0 {
                            return Err(format!(
                                "Слой {}: выходная размерность {} равна нулю", i, j
                            ));
                        }
                    }
                }
            }
        }

        // 2. Проверка совместимости между соседними слоями
        for i in 1..descs.len() {
            let prev = &descs[i - 1];
            let curr = &descs[i];

            // Unsqueeze и ReduceMean меняют размерность – их совместимость определяется сохранением числа элементов
            if matches!(prev.kind, LayerKind::Unsqueeze | LayerKind::ReduceMean)
                || matches!(curr.kind, LayerKind::Unsqueeze | LayerKind::ReduceMean)
            {
                continue;
            }

            let prev_out = &prev.output_shape.streams;
            let curr_in = &curr.input_shape.streams;

            // Splitter / SplitterConnector → ожидается несколько выходов
            let prev_is_splitter = matches!(
                prev.kind,
                LayerKind::SplitterConnector | LayerKind::Splitter
            );
            // Combiner / CombinerConnector → ожидается несколько входов
            let curr_is_combiner = matches!(
                curr.kind,
                LayerKind::CombinerConnector | LayerKind::Combiner
            );

            if prev_is_splitter && curr_is_combiner {
                // Количество потоков должно совпадать
                if prev_out.len() != curr_in.len() {
                    return Err(format!(
                        "Несовместимость числа потоков между Splitter (слой {}) и Combiner (слой {}): выходов {}, входов {}",
                        i, i + 1, prev_out.len(), curr_in.len()
                    ));
                }
                // Размеры признаков попарно
                for (p, (out_sz, in_sz)) in prev_out.iter().zip(curr_in.iter()).enumerate() {
                    if out_sz != in_sz {
                        return Err(format!(
                            "Размеры не совпадают между Splitter (слой {}) поток {} и Combiner (слой {}): {} vs {}",
                            i, p, i + 1, out_sz, in_sz
                        ));
                    }
                }
            } else if prev_is_splitter {
                // После Splitter'а может идти обычный слой – тогда берём первый выходной поток
                if curr_in.len() == 1 {
                    if prev_out[0] != curr_in[0] {
                        return Err(format!(
                            "Размеры не совпадают между Splitter (слой {}) первый выход {} и вход слоя {} ({} входной поток)",
                            i, prev_out[0], i + 1, curr_in[0]
                        ));
                    }
                } else {
                    // Если следующий слой ожидает несколько входов, но не Combiner – пока считаем ошибкой
                    return Err(format!(
                        "Слой {} (kind {:?}) не может следовать за Splitter'ом с несколькими выходами",
                        i + 1, curr.kind
                    ));
                }
            } else if curr_is_combiner {
                // Перед Combiner'ом может идти обычный слой – тогда его выход должен совпадать с первым входом Combiner'а
                if prev_out.len() == 1 {
                    if prev_out[0] != curr_in[0] {
                        return Err(format!(
                            "Размеры не совпадают между слоем {} (выход {}) и Combiner (слой {}) первый вход {}",
                            i, prev_out[0], i + 1, curr_in[0]
                        ));
                    }
                } else {
                    return Err(format!(
                        "Слой {} (kind {:?}) не может предшествовать Combiner'у с несколькими входами",
                        i, prev.kind
                    ));
                }
            } else {
                // Обычная пара слоёв: сравниваем общее количество признаков
                let prev_total = prev_out.iter().product::<usize>();
                let curr_total = curr_in.iter().product::<usize>();
                if prev_total != curr_total {
                    return Err(format!(
                        "Несовместимость общего числа признаков между слоем {} (выход {} элементов) и слоем {} (вход {} элементов)",
                        i, prev_total, i + 1, curr_total
                    ));
                }
            }
        }

        // 3. Softmax разрешён только на последнем слое
        for (i, desc) in descs.iter().enumerate() {
            if desc.kind == LayerKind::Softmax && i != descs.len() - 1 {
                return Err("Softmax допускается только на последнем слое".into());
            }
        }

        Ok(Plan { layers: descs })
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
    pub fn build_with_device(&self, device: Device) -> MixedModel {
        MixedModel::from_plan_with_device(self.layers.clone(), 1, device)
            .expect("Plan уже проверен")
    }

    /// Собрать модель с детализированным планом устройств (разделение Compute/Storage).
    pub fn build_with_device_plan(&self, device_plan: DevicePlan) -> MixedModel {
        MixedModel::from_plan_with_device_plan(self.layers.clone(), device_plan)
            .expect("Plan уже проверен")
    }

    /// Размерность входа (первый поток первого слоя).
    pub fn input_dim1(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.input_shape.streams[0])
            .unwrap_or(0)
    }

    /// Размерность выхода (первый поток последнего слоя).
    pub fn output_dim1(&self) -> usize {
        self.layers
            .last()
            .map(|l| l.output_shape.streams[0])
            .unwrap_or(0)
    }
}
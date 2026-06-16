use std::sync::{Arc, Barrier};
use super::cost::CostModel;
use super::hardware::CpuInfo;
use super::task::{LayerPlan, RangeTask};

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub id: usize,
    pub layer_type: LayerType,
    pub in_features: usize,
    pub out_features: usize,
    pub total_rows: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    Linear,
    Activation,
}

pub struct Scheduler {
    cost: CostModel,
    cpu_info: CpuInfo,
    num_workers: usize,
    /// Опциональный предиктор (для TrainedModel), возвращает желаемое число задач.
    pub predictor: Option<Box<dyn Fn(&LayerInfo) -> usize + Send + Sync>>,
}

// Добавляем ручную реализацию Clone, чтобы предиктор не клонировался
impl Clone for Scheduler {
    fn clone(&self) -> Self {
        Scheduler {
            cost: self.cost.clone(),
            cpu_info: self.cpu_info.clone(),
            num_workers: self.num_workers,
            predictor: None,
        }
    }
}

impl Scheduler {
    pub fn new(cost: CostModel, cpu_info: CpuInfo) -> Self {
        let num_workers = cost.num_cores;
        Scheduler { cost, cpu_info, num_workers, predictor: None }
    }

    pub fn set_num_workers(&mut self, n: usize) {
        self.num_workers = n;
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    pub fn build_plan(&self, layers: &[LayerInfo]) -> Vec<LayerPlan> {
        layers.iter().map(|l| self.plan_layer(l)).collect()
    }

    fn plan_layer(&self, layer: &LayerInfo) -> LayerPlan {
        let num_workers = self.num_workers;
        let total_rows = layer.total_rows;
        let out_features = layer.out_features;
        let in_features = layer.in_features;

        // Если есть обученный предиктор – используем его
        if let Some(ref pred) = self.predictor {
            let num_tasks = pred(layer).max(1);
            let rows_per = (total_rows as f64 / num_tasks as f64).ceil() as usize;
            let tasks: Vec<_> = split_range(0, total_rows, rows_per)
                .into_iter()
                .map(|(rs, re)| RangeTask {
                    row_start: rs, row_end: re,
                    col_start: 0, col_end: out_features,
                })
                .collect();
            return LayerPlan { tasks, barrier: Arc::new(Barrier::new(num_workers)) };
        }

        // Автоматический пороговый метод
        let work_macs = match layer.layer_type {
            LayerType::Linear => total_rows * out_features * in_features,
            LayerType::Activation => total_rows * out_features * 2,
        };
        let min_mac_per_task = self.cost.min_mac_per_task();
        let min_total_for_parallel = min_mac_per_task * (num_workers as f64);

        if (work_macs as f64) < min_total_for_parallel {
            let single = vec![RangeTask {
                row_start: 0, row_end: total_rows,
                col_start: 0, col_end: out_features,
            }];
            return LayerPlan { tasks: single, barrier: Arc::new(Barrier::new(num_workers)) };
        }

        if layer.layer_type == LayerType::Activation {
            let rows_per = Self::optimal_chunk(total_rows, work_macs, min_mac_per_task, num_workers);
            let tasks = split_range(0, total_rows, rows_per)
                .into_iter()
                .map(|(rs, re)| RangeTask { row_start: rs, row_end: re, col_start: 0, col_end: out_features })
                .collect();
            return LayerPlan { tasks, barrier: Arc::new(Barrier::new(num_workers)) };
        }

        let cache_limit = self.cpu_info.l3_cache_size.max(256 * 1024);
        let row_mem = in_features * 4;
        let col_mem = total_rows * 4;
        let max_rows_per_chunk = (cache_limit / (row_mem + col_mem).max(1)).max(1);
        let rows_per = total_rows.min(max_rows_per_chunk).max(1);

        let row_chunks = split_range(0, total_rows, rows_per);
        let col_chunks = if total_rows <= num_workers / 2 {
            let cols_per = (out_features as f64 / num_workers as f64).ceil() as usize;
            split_range(0, out_features, cols_per)
        } else {
            vec![(0, out_features)]
        };

        let mut tasks: Vec<RangeTask> = row_chunks.iter()
            .flat_map(|&(rs, re)| col_chunks.iter().map(move |&(cs, ce)| RangeTask { row_start: rs, row_end: re, col_start: cs, col_end: ce }))
            .filter(|t| ((t.row_end - t.row_start) * (t.col_end - t.col_start) * in_features) as f64 >= min_mac_per_task)
            .collect();

        if tasks.is_empty() {
            tasks = vec![RangeTask { row_start: 0, row_end: total_rows, col_start: 0, col_end: out_features }];
        }

        LayerPlan { tasks, barrier: Arc::new(Barrier::new(num_workers)) }
    }

    fn optimal_chunk(total: usize, work: usize, min_mac: f64, workers: usize) -> usize {
        let work_per = work / total.max(1);
        let min_rows = (min_mac / work_per.max(1) as f64).ceil() as usize;
        let chunks = total / min_rows.max(1);
        let chunks = chunks.max(1).min(workers);
        (total as f64 / chunks as f64).ceil() as usize
    }
}

fn split_range(start: usize, end: usize, size: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut cur = start;
    while cur < end {
        let next = (cur + size).min(end);
        v.push((cur, next));
        cur = next;
    }
    v
}
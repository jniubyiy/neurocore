// src/compute_manager/cpu/scheduler.rs

use std::path::PathBuf;

use crate::compute_manager::cpu::cost::CostModel;
use crate::compute_manager::cpu::hardware::CpuInfo;
use crate::compute_manager::cpu::profiler::HardwareProfile;
use crate::compute_manager::cpu::mini_model::ForwardTimePredictor;

const MAX_CORES_FOR_MODEL: usize = 16;
const MAX_CHUNKS_PER_WORKER: usize = 10;
const TRAINING_THRESHOLD: usize = 50;

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

/// Планировщик задач для CPU с персональными мини-моделями для каждого ядра.
pub struct Scheduler {
    num_workers: usize,
    predictors: Vec<ForwardTimePredictor>,
    predictors_paths: Vec<PathBuf>,
    profile: HardwareProfile,
    cpu_info: CpuInfo,
    cost: CostModel,
    training_data: Vec<Vec<(usize, f64)>>,
    training_threshold: usize,
    /// Наличие GPU в плане устройств. Может использоваться для
    /// корректировки стратегии планирования (например, уменьшения
    /// числа CPU-чанков, если часть работы перекладывается на GPU).
    has_gpu: bool,
}

impl Clone for Scheduler {
    fn clone(&self) -> Self {
        let num_cpus = self.predictors.len();
        let input_dim = 5 + MAX_CORES_FOR_MODEL;
        let mut new_predictors = Vec::with_capacity(num_cpus);
        for _ in 0..num_cpus {
            new_predictors.push(ForwardTimePredictor::new(input_dim, 8));
        }
        Scheduler {
            num_workers: self.num_workers,
            predictors: new_predictors,
            predictors_paths: self.predictors_paths.clone(),
            profile: self.profile.clone(),
            cpu_info: self.cpu_info.clone(),
            cost: self.cost.clone(),
            training_data: vec![Vec::new(); num_cpus],
            training_threshold: self.training_threshold,
            has_gpu: self.has_gpu,
        }
    }
}

fn get_data_dir() -> PathBuf {
    let env_dir = std::env::var("NEUROCORE_DATA_DIR").ok();
    if let Some(dir) = env_dir {
        let p = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&p);
        return p;
    }

    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).ok();
    if let Some(home) = home {
        let p = PathBuf::from(home).join(".local/share/neurocore");
        if std::fs::create_dir_all(&p).is_ok() {
            return p;
        }
    }

    let fallback = PathBuf::from("neurocore_data");
    if std::fs::create_dir_all(&fallback).is_err() {
        return std::env::temp_dir().join("neurocore_data");
    }
    fallback
}

impl Scheduler {
    /// Создаёт планировщик с указанием наличия GPU.
    pub fn new_with_cpus(
        cost: CostModel,
        cpu_info: CpuInfo,
        num_cpus: usize,
        has_gpu: bool,
    ) -> Self {
        let num_workers = cost.num_cores;
        let data_dir = get_data_dir();
        let profile_path = data_dir.join("hardware_profile.json");

        let profile = HardwareProfile::load(&profile_path)
            .unwrap_or_else(|| {
                let p = HardwareProfile::calibrate();
                p.save(&profile_path);
                p
            });

        let mut predictors = Vec::with_capacity(num_cpus);
        let mut predictors_paths = Vec::with_capacity(num_cpus);
        let input_dim = 5 + MAX_CORES_FOR_MODEL;
        for cpu_idx in 0..num_cpus {
            let model_path = data_dir.join(format!("chunk_model_cpu{}.json", cpu_idx));
            predictors_paths.push(model_path.clone());
            let model = ForwardTimePredictor::load(&model_path, input_dim, 8)
                .unwrap_or_else(|| ForwardTimePredictor::new(input_dim, 8));
            predictors.push(model);
        }

        Scheduler {
            num_workers,
            predictors,
            predictors_paths,
            profile,
            cpu_info,
            cost,
            training_data: vec![Vec::new(); num_cpus],
            training_threshold: TRAINING_THRESHOLD,
            has_gpu,
        }
    }

    /// Создаёт планировщик без GPU (обратная совместимость).
    pub fn new(cost: CostModel, cpu_info: CpuInfo) -> Self {
        Self::new_with_cpus(cost, cpu_info, 1, false)
    }

    pub fn set_num_workers(&mut self, n: usize) {
        self.num_workers = n;
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    pub fn report_execution_time(&mut self, cpu_idx: usize, task_size: usize, duration_ns: f64) {
        if cpu_idx >= self.training_data.len() {
            return;
        }

        self.training_data[cpu_idx].push((task_size, duration_ns));

        if self.training_data[cpu_idx].len() >= self.training_threshold {
            // Забираем данные для этого CPU
            let data = std::mem::take(&mut self.training_data[cpu_idx]);

            // Строим признаки для каждого элемента заранее, чтобы не держать mutable borrow на predictor
            // во время вызова self.build_time_features (который заимствует self неизменно)
            let mut features_list = Vec::with_capacity(data.len());
            let mut targets = Vec::with_capacity(data.len());
            for (size, time) in &data {
                let features = self.build_time_features(*size);
                let target = (*time / 1_000_000_000.0) as f32;
                features_list.push(features);
                targets.push(target);
            }

            // Теперь mutable borrow на predictor
            let predictor = &mut self.predictors[cpu_idx];
            for (features, target) in features_list.into_iter().zip(targets) {
                predictor.train(&features, target, 0.001);
            }

            // Сохраняем модель
            if cpu_idx < self.predictors_paths.len() {
                predictor.save(&self.predictors_paths[cpu_idx]);
            }
        }
    }

    pub fn predict_time(&self, cpu_idx: usize, task_size: usize) -> Option<f64> {
        if cpu_idx >= self.predictors.len() {
            return None;
        }
        let features = self.build_time_features(task_size);
        let pred = self.predictors[cpu_idx].predict(&features);
        if pred > 0.0 {
            Some(pred as f64)
        } else {
            None
        }
    }

    pub fn plan_chunks_assignment(&mut self, total_tasks: usize) -> Vec<Vec<(usize, usize, usize)>> {
        if total_tasks == 0 {
            return vec![Vec::new(); self.num_workers];
        }

        // Если есть GPU, можно уменьшить максимальное количество чанков,
        // так как часть работы уже выполняется на GPU, и CPU‑потоки не должны
        // создавать излишнюю конкуренцию. Для простоты ограничим максимальное
        // число чанков половиной от обычного.
        let max_chunks_per_worker = if self.has_gpu {
            MAX_CHUNKS_PER_WORKER / 2
        } else {
            MAX_CHUNKS_PER_WORKER
        };

        let max_chunks = total_tasks.min(self.num_workers * max_chunks_per_worker);

        let speeds = self.profile.core_relative_speeds.clone();

        let has_trained_model = self.predictors.iter().enumerate().any(|(idx, _)| {
            !self.training_data[idx].is_empty()
        });

        if has_trained_model {
            self.plan_with_predictions(total_tasks, max_chunks, &speeds)
        } else {
            self.plan_fallback(total_tasks, max_chunks, &speeds)
        }
    }

    fn plan_with_predictions(
        &mut self,
        total_tasks: usize,
        max_chunks: usize,
        speeds: &[f64],
    ) -> Vec<Vec<(usize, usize, usize)>> {
        let num_workers = self.num_workers;
        let mut best_assignment = vec![Vec::new(); num_workers];
        let mut best_max_time = f64::MAX;

        let max_c = max_chunks.min(total_tasks);
        for c in 1..=max_c {
            let chunks = split_into_chunks(total_tasks, c);
            let assignment = self.assign_chunks_with_predictions(&chunks);
            let mut max_time = 0.0;
            for cpu_idx in 0..num_workers {
                let mut total_time = 0.0;
                for &(_start, size, _end) in &assignment[cpu_idx] {
                    if let Some(time) = self.predict_time(cpu_idx, size) {
                        total_time += time;
                    } else {
                        let speed = if cpu_idx < speeds.len() { speeds[cpu_idx] } else { 1.0 };
                        total_time += size as f64 / speed;
                    }
                }
                if total_time > max_time {
                    max_time = total_time;
                }
            }
            if max_time < best_max_time {
                best_max_time = max_time;
                best_assignment = assignment;
            }
        }

        best_assignment
    }

    fn assign_chunks_with_predictions(&self, chunks: &[(usize, usize, usize)]) -> Vec<Vec<(usize, usize, usize)>> {
        let num_workers = self.num_workers;
        let mut assignment: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_workers];
        let mut load_times = vec![0.0; num_workers];

        let mut sorted_chunks: Vec<(usize, usize, usize)> = chunks.to_vec();
        sorted_chunks.sort_by(|a, b| b.1.cmp(&a.1)); // сортируем по size

        for &(start, size, end) in &sorted_chunks {
            let mut best_cpu = 0;
            let mut best_time = f64::MAX;
            for cpu_idx in 0..num_workers {
                let time = if let Some(pred) = self.predict_time(cpu_idx, size) {
                    load_times[cpu_idx] + pred
                } else {
                    let speed = if cpu_idx < self.profile.core_relative_speeds.len() {
                        self.profile.core_relative_speeds[cpu_idx]
                    } else {
                        1.0
                    };
                    load_times[cpu_idx] + size as f64 / speed
                };
                if time < best_time {
                    best_time = time;
                    best_cpu = cpu_idx;
                }
            }
            assignment[best_cpu].push((start, size, end));
            if let Some(pred) = self.predict_time(best_cpu, size) {
                load_times[best_cpu] += pred;
            } else {
                let speed = if best_cpu < self.profile.core_relative_speeds.len() {
                    self.profile.core_relative_speeds[best_cpu]
                } else {
                    1.0
                };
                load_times[best_cpu] += size as f64 / speed;
            }
        }

        assignment
    }

    fn plan_fallback(
        &mut self,
        total_tasks: usize,
        max_chunks: usize,
        speeds: &[f64],
    ) -> Vec<Vec<(usize, usize, usize)>> {
        let mut best_penalty = f32::MAX;
        let mut best_assignment = vec![Vec::new(); self.num_workers];

        for c in 1..=max_chunks {
            let chunks = split_into_chunks(total_tasks, c);
            let assignment = greedy_assign(&chunks, speeds);
            let loads: Vec<f64> = assignment
                .iter()
                .map(|assigned| assigned.iter().map(|(_, size, _)| *size as f64).sum())
                .collect();
            let penalty = calculate_imbalance(&loads, speeds);
            if penalty < best_penalty {
                best_penalty = penalty;
                best_assignment = assignment;
            }
        }

        best_assignment
    }

    fn build_time_features(&self, task_size: usize) -> Vec<f32> {
        let fmadd_ms = (self.cost.fmadd_ns as f32) / 1_000_000.0;
        let neuron_time_ms = (self.profile.neuron_time_ns as f32) / 1_000_000.0;

        let mut feats = vec![
            task_size as f32,
            fmadd_ms,
            neuron_time_ms,
            self.profile.cache_congestion_factor as f32,
            self.profile.memory_per_neuron_bytes as f32,
        ];

        let speeds = &self.profile.core_relative_speeds;
        for i in 0..MAX_CORES_FOR_MODEL {
            feats.push(if i < speeds.len() { speeds[i] as f32 } else { 0.0 });
        }
        feats
    }
}

fn split_into_chunks(total: usize, num_chunks: usize) -> Vec<(usize, usize, usize)> {
    let base = total / num_chunks;
    let rem = total % num_chunks;
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut start = 0;
    for i in 0..num_chunks {
        let size = if i < rem { base + 1 } else { base };
        let end = start + size;
        chunks.push((start, size, end));
        start = end;
    }
    chunks
}

fn greedy_assign(
    chunks: &[(usize, usize, usize)],
    speeds: &[f64],
) -> Vec<Vec<(usize, usize, usize)>> {
    let num_workers = speeds.len();
    let mut assignment: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); num_workers];
    let mut loads = vec![0.0; num_workers];

    let mut sorted_chunks: Vec<(usize, usize, usize)> = chunks.to_vec();
    sorted_chunks.sort_by(|a, b| b.1.cmp(&a.1)); // по size

    for chunk in &sorted_chunks {
        let mut best_worker = 0;
        let mut best_time = f64::MAX;
        for i in 0..num_workers {
            let speed = if i < speeds.len() { speeds[i] } else { 1.0 };
            let time = loads[i] / speed;
            if time < best_time {
                best_time = time;
                best_worker = i;
            }
        }
        assignment[best_worker].push(*chunk);
        loads[best_worker] += chunk.1 as f64;
    }
    assignment
}

fn calculate_imbalance(loads: &[f64], speeds: &[f64]) -> f32 {
    let n = loads.len();
    if n == 0 {
        return 0.0;
    }
    let weighted: Vec<f64> = loads
        .iter()
        .enumerate()
        .map(|(i, &load)| {
            let speed = if i < speeds.len() { speeds[i] } else { 1.0 };
            load / speed
        })
        .collect();
    let avg = weighted.iter().sum::<f64>() / n as f64;
    let penalty: f64 = weighted.iter().map(|w| (w - avg).abs()).sum();
    (penalty / n as f64) as f32
}
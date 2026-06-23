// src/compute_manager/cpu/scheduler.rs

use std::path::PathBuf;
use crate::compute_manager::cpu::cost::CostModel;
use crate::compute_manager::cpu::hardware::CpuInfo;
use crate::compute_manager::cpu::profiler::HardwareProfile;
use crate::compute_manager::cpu::mini_model::ForwardTimePredictor;

const MAX_CORES_FOR_MODEL: usize = 16;
const MAX_CHUNKS_PER_WORKER: usize = 10;

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
    num_workers: usize,
    chunk_model: ForwardTimePredictor,
    chunk_model_path: PathBuf,
    profile: HardwareProfile,
    cpu_info: CpuInfo,
    cost: CostModel,
}

impl Clone for Scheduler {
    fn clone(&self) -> Self {
        let input_dim = 5 + MAX_CORES_FOR_MODEL;
        let new_model = ForwardTimePredictor::new(input_dim, 8);
        Scheduler {
            num_workers: self.num_workers,
            chunk_model: new_model,
            chunk_model_path: self.chunk_model_path.clone(),
            profile: self.profile.clone(),
            cpu_info: self.cpu_info.clone(),
            cost: self.cost.clone(),
        }
    }
}

fn get_data_dir() -> PathBuf {
    let env_dir = std::env::var("NEUROCORE_DATA_DIR").ok();
    if let Some(dir) = env_dir {
        let p = PathBuf::from(dir);
        if let Err(_) = std::fs::create_dir_all(&p) { /* игнорируем */ }
        return p;
    }

    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).ok();
    if let Some(home) = home {
        let p = PathBuf::from(home).join(".local/share/neurocore");
        if let Err(_) = std::fs::create_dir_all(&p) { /* не удалось, fallback */ }
        else { return p; }
    }

    let fallback = PathBuf::from("neurocore_data");
    if let Err(_) = std::fs::create_dir_all(&fallback) {
        return std::env::temp_dir().join("neurocore_data");
    }
    fallback
}

impl Scheduler {
    pub fn new(cost: CostModel, cpu_info: CpuInfo) -> Self {
        let num_workers = cost.num_cores;
        let data_dir = get_data_dir();
        let profile_path = data_dir.join("hardware_profile.json");
        let chunk_model_path = data_dir.join("chunk_model.json");

        let profile = HardwareProfile::load(&profile_path)
            .unwrap_or_else(|| {
                let p = HardwareProfile::calibrate();
                p.save(&profile_path);
                p
            });

        let input_dim = 5 + MAX_CORES_FOR_MODEL;
        let chunk_model = ForwardTimePredictor::load(&chunk_model_path, input_dim)
            .unwrap_or_else(|| ForwardTimePredictor::new(input_dim, 8));

        Scheduler {
            num_workers,
            chunk_model,
            chunk_model_path,
            profile,
            cpu_info,
            cost,
        }
    }

    pub fn set_num_workers(&mut self, n: usize) {
        self.num_workers = n;
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Основной метод планирования: определяет оптимальное количество чанков,
    /// разбивает общее число задач на чанки и жадным алгоритмом назначает их ядрам.
    /// Возвращает для каждого ядра список диапазонов (start, size).
    pub fn plan_chunks_assignment(&mut self, total_tasks: usize) -> Vec<Vec<(usize, usize)>> {
        if total_tasks == 0 {
            return vec![Vec::new(); self.num_workers];
        }

        let speeds = &self.profile.core_relative_speeds;
        let max_chunks = total_tasks.min(self.num_workers * MAX_CHUNKS_PER_WORKER);

        let mut best_penalty = f32::MAX;
        let mut best_assignment = vec![Vec::new(); self.num_workers];
        let mut best_c = 1;

        // Перебор возможных количеств чанков
        for c in 1..=max_chunks {
            // Разбиваем total_tasks на c равных (с остатком) непрерывных чанков
            let chunks = split_into_chunks(total_tasks, c);
            // Жадно назначаем чанки ядрам
            let assignment = greedy_assign(&chunks, speeds);
            // Вычисляем нагрузки и штраф
            let loads: Vec<f64> = assignment
                .iter()
                .map(|assigned| assigned.iter().map(|(_, size)| *size as f64).sum())
                .collect();
            let penalty = calculate_imbalance(&loads, speeds);
            if penalty < best_penalty {
                best_penalty = penalty;
                best_assignment = assignment;
                best_c = c;
            }
        }

        // Обучаем модель, используя лучший c
        let features = self.build_features(total_tasks);
        let target = best_c as f32 / total_tasks as f32;
        self.chunk_model.train(&features, target, 0.001);
        self.chunk_model.save(&self.chunk_model_path);

        best_assignment
    }

    fn build_features(&self, total_tasks: usize) -> Vec<f32> {
        let fmadd_ms = (self.cost.fmadd_ns as f32) / 1_000_000.0;
        let neuron_time_ms = (self.profile.neuron_time_ns as f32) / 1_000_000.0;

        let mut feats = vec![
            total_tasks as f32,
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

// Разбивает total_tasks на num_chunks непрерывных чанков с равномерным распределением остатка.
fn split_into_chunks(total: usize, num_chunks: usize) -> Vec<(usize, usize)> {
    let base = total / num_chunks;
    let rem = total % num_chunks;
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut start = 0;
    for i in 0..num_chunks {
        let size = if i < rem { base + 1 } else { base };
        chunks.push((start, size));
        start += size;
    }
    chunks
}

// Жадное назначение чанков ядрам с учётом относительных скоростей.
fn greedy_assign(
    chunks: &[(usize, usize)],
    speeds: &[f64],
) -> Vec<Vec<(usize, usize)>> {
    let num_workers = speeds.len();
    let mut assignment: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_workers];
    let mut loads = vec![0.0; num_workers];

    // Сортируем чанки по убыванию размера
    let mut sorted_chunks: Vec<(usize, usize)> = chunks.to_vec();
    sorted_chunks.sort_by(|a, b| b.1.cmp(&a.1));

    for chunk in &sorted_chunks {
        // Ищем ядро с минимальным ожидаемым временем завершения (load / speed)
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

// Вычисляет среднее абсолютное отклонение нормализованных нагрузок (load/speed).
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
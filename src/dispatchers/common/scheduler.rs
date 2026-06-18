// src/dispatchers/common/scheduler.rs

use std::path::PathBuf;
use super::cost::CostModel;
use super::hardware::CpuInfo;
use super::profiler::HardwareProfile;
use super::mini_model::ForwardTimePredictor;

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
    mini_model: Option<ForwardTimePredictor>,
    training_data: Vec<(Vec<f32>, usize, f32)>,
    model_path: PathBuf,
    profile: HardwareProfile,
}

impl Clone for Scheduler {
    fn clone(&self) -> Self {
        Scheduler {
            cost: self.cost.clone(),
            cpu_info: self.cpu_info.clone(),
            num_workers: self.num_workers,
            mini_model: None,
            training_data: Vec::new(),
            model_path: self.model_path.clone(),
            profile: self.profile.clone(),
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
        let predictor_path = data_dir.join("forward_time_predictor.json");
        let profile_path = data_dir.join("hardware_profile.json");

        // Ожидаемая размерность признаков: 12 (11 базовых + num_tasks)
        let expected_input_dim = 12;
        let mini_model = ForwardTimePredictor::load(&predictor_path, expected_input_dim);
        let profile = HardwareProfile::load(&profile_path)
            .unwrap_or_else(|| {
                let p = HardwareProfile::calibrate();
                p.save(&profile_path);
                p
            });

        Scheduler {
            cost,
            cpu_info,
            num_workers,
            mini_model,
            training_data: Vec::new(),
            model_path: predictor_path,
            profile,
        }
    }

    pub fn set_num_workers(&mut self, n: usize) {
        self.num_workers = n;
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    pub fn plan_forward(&self, layers: &[LayerInfo]) -> Vec<usize> {
        let total_rows = layers.first().map(|l| l.total_rows).unwrap_or(0);
        if total_rows == 0 {
            return vec![0];
        }

        let max_in = layers.iter().map(|l| l.in_features).max().unwrap_or(0);
        let max_out = layers.iter().map(|l| l.out_features).max().unwrap_or(0);
        let sum_linear_work: usize = layers
            .iter()
            .filter(|l| l.layer_type == LayerType::Linear)
            .map(|l| l.in_features * l.out_features)
            .sum();
        let num_linear = layers.iter().filter(|l| l.layer_type == LayerType::Linear).count();
        let num_activ = layers.iter().filter(|l| l.layer_type == LayerType::Activation).count();

        let features = vec![
            max_in as f32,
            max_out as f32,
            total_rows as f32,
            sum_linear_work as f32,
            num_linear as f32,
            num_activ as f32,
            self.profile.neuron_time_ns as f32,
            self.profile.linear_layer_time_ns as f32,
            self.profile.cache_congestion_factor as f32,
            self.profile.memory_per_neuron_bytes as f32,
            self.profile.core_relative_speeds.iter().sum::<f64>() as f32 / self.profile.core_relative_speeds.len() as f32,
        ];

        if let Some(model) = &self.mini_model {
            let mut best_tasks = 1;
            let mut best_time = f32::MAX;
            for n in 1..=self.num_workers {
                let mut feats = features.clone();
                feats.push(n as f32); // добавляем num_tasks
                let pred_time = model.predict(&feats);
                if pred_time < best_time {
                    best_time = pred_time;
                    best_tasks = n;
                }
            }
            self.distribute_rows(total_rows, best_tasks)
        } else {
            let tasks_count = self.num_workers.min(total_rows).max(1);
            self.distribute_rows(total_rows, tasks_count)
        }
    }

    fn distribute_rows(&self, total: usize, num_tasks: usize) -> Vec<usize> {
        let speeds = &self.profile.core_relative_speeds;
        if speeds.len() >= num_tasks {
            let total_speed: f64 = speeds.iter().take(num_tasks).sum();
            let mut remaining = total;
            let mut sizes = Vec::with_capacity(num_tasks);
            for i in 0..num_tasks - 1 {
                let ratio = speeds[i] / total_speed;
                let rows = (total as f64 * ratio).round() as usize;
                let rows = rows.min(remaining);
                sizes.push(rows);
                remaining -= rows;
            }
            sizes.push(remaining);
            sizes
        } else {
            let rows_per = total / num_tasks;
            let mut sizes = vec![rows_per; num_tasks];
            let remainder = total % num_tasks;
            for i in 0..remainder {
                sizes[i] += 1;
            }
            sizes
        }
    }

    pub fn record_forward_time(&mut self, layers: &[LayerInfo], chunk_sizes: &[usize], elapsed_secs: f32) {
        let total_rows: usize = chunk_sizes.iter().sum();
        if total_rows == 0 { return; }

        let max_in = layers.iter().map(|l| l.in_features).max().unwrap_or(0);
        let max_out = layers.iter().map(|l| l.out_features).max().unwrap_or(0);
        let sum_linear_work: usize = layers
            .iter()
            .filter(|l| l.layer_type == LayerType::Linear)
            .map(|l| l.in_features * l.out_features)
            .sum();
        let num_linear = layers.iter().filter(|l| l.layer_type == LayerType::Linear).count();
        let num_activ = layers.iter().filter(|l| l.layer_type == LayerType::Activation).count();

        let num_tasks = chunk_sizes.len();
        let mut features = vec![
            max_in as f32,
            max_out as f32,
            total_rows as f32,
            sum_linear_work as f32,
            num_linear as f32,
            num_activ as f32,
            self.profile.neuron_time_ns as f32,
            self.profile.linear_layer_time_ns as f32,
            self.profile.cache_congestion_factor as f32,
            self.profile.memory_per_neuron_bytes as f32,
            self.profile.core_relative_speeds.iter().sum::<f64>() as f32 / self.profile.core_relative_speeds.len() as f32,
        ];
        features.push(num_tasks as f32); // <-- добавлено для согласования с plan_forward

        self.training_data.push((features, num_tasks, elapsed_secs));

        if self.training_data.len() >= 50 {
            self.train_model();
        }
    }

    fn train_model(&mut self) {
        if self.training_data.is_empty() { return; }
        if self.mini_model.is_none() {
            let input_dim = self.training_data[0].0.len(); // теперь будет 12
            self.mini_model = Some(ForwardTimePredictor::new(input_dim, 8));
        }
        let model = self.mini_model.as_mut().unwrap();
        let lr = 0.001;
        for (features, _, time) in &self.training_data {
            model.train(features, *time, lr);
        }
        model.save(&self.model_path);
    }
}
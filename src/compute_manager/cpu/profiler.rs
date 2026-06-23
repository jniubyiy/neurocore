// src/compute_manager/cpu/profiler.rs

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use std::hint::black_box;

/// Результаты профилирования железа
#[derive(Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub neuron_time_ns: f64,
    pub linear_layer_time_ns: f64,
    pub core_relative_speeds: Vec<f64>,
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub cache_congestion_factor: f64,
    pub memory_per_neuron_bytes: usize,
}

impl HardwareProfile {
    /// Запускает калибровку и возвращает профиль
    pub fn calibrate() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        // 1. Время нейрона (ReLU)
        let neuron_time_ns = {
            let mut x = vec![0.0f32; 1024];
            for (i, v) in x.iter_mut().enumerate() { *v = (i as f32) * 0.001; }
            let start = Instant::now();
            let n_iters = 10_000;
            for _ in 0..n_iters {
                for val in &mut x {
                    *val = val.max(0.0);
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1e9;
            elapsed / (n_iters * x.len()) as f64
        };

        // 2. Время линейного слоя (эталон)
        let linear_layer_time_ns = {
            // Эмуляция небольшого линейного слоя
            let rows = 64;
            let in_dim = 128;
            let out_dim = 64;
            let data = vec![vec![1.0f32; in_dim]; rows];
            let input = crate::tensor::Tensor2D::new(data);
            let w = vec![0.1f32; in_dim * out_dim];
            let start = Instant::now();
            let n_iters = 500;
            for _ in 0..n_iters {
                let m = crate::linalg::tensor2d_to_faer(&input);
                let _ = &m * &faer::Mat::from_fn(in_dim, out_dim, |r, c| w[r * out_dim + c]);
            }
            let elapsed = start.elapsed().as_secs_f64() * 1e9;
            elapsed / n_iters as f64
        };

        // 3. Относительная производительность ядер
        let core_relative_speeds = vec![1.0; logical_cores];

        // 4. Размеры кэша
        let l1 = 32 * 1024;
        let l2 = 256 * 1024;
        let l3 = 8 * 1024 * 1024;

        // 5. Коэффициент запылённости кэша
        let cache_congestion_factor = {
            let array_size = l3 / std::mem::size_of::<f64>();
            let arr = vec![0.0f64; array_size];
            let start = Instant::now();
            for _ in 0..10 {
                for val in &arr {
                    black_box(*val);
                }
            }
            let time_l3 = start.elapsed().as_secs_f64();
            let array_size_l2 = l2 / std::mem::size_of::<f64>();
            let arr2 = vec![0.0f64; array_size_l2];
            let start2 = Instant::now();
            for _ in 0..10 {
                for val in &arr2 {
                    black_box(*val);
                }
            }
            let time_l2 = start2.elapsed().as_secs_f64();
            let expected_ratio = array_size as f64 / array_size_l2 as f64;
            let actual_ratio = time_l3 / time_l2;
            if actual_ratio > expected_ratio * 1.5 { 0.7 } else { 1.0 }
        };

        // 6. Память на нейрон
        let memory_per_neuron_bytes = std::mem::size_of::<f32>() * 2;

        HardwareProfile {
            neuron_time_ns,
            linear_layer_time_ns,
            core_relative_speeds,
            l1_cache_size: l1,
            l2_cache_size: l2,
            l3_cache_size: l3,
            cache_congestion_factor,
            memory_per_neuron_bytes,
        }
    }

    /// Пытается сохранить профиль в файл. Ошибки логируются, но не прерывают работу.
    pub fn save(&self, path: &PathBuf) {
        match serde_json::to_string(self) {
            Ok(data) => {
                if let Err(e) = fs::write(path, data) {
                    eprintln!("[neurocore] Не удалось сохранить профиль железа: {}", e);
                }
            }
            Err(e) => eprintln!("[neurocore] Ошибка сериализации профиля: {}", e),
        }
    }

    /// Загружает профиль из файла, либо возвращает None при ошибке.
    pub fn load(path: &PathBuf) -> Option<Self> {
        if path.exists() {
            match fs::read_to_string(path) {
                Ok(data) => match serde_json::from_str(&data) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        eprintln!("[neurocore] Ошибка парсинга профиля: {}", e);
                        None
                    }
                },
                Err(e) => {
                    eprintln!("[neurocore] Не удалось прочитать профиль: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }
}
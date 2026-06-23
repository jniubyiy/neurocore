// src/compute_manager/cpu/mini_model.rs

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Миниатюрная нейросеть с одним выходом.
/// Используется для предсказания оптимального количества чанков (нормированного значения).
#[derive(Serialize, Deserialize)]
pub struct ForwardTimePredictor {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,   // размер: hidden * 1
    b2: f32,        // размер: 1
    input_dim: usize,
    hidden: usize,
}

impl ForwardTimePredictor {
    /// Создаёт модель с заданной размерностью входа и скрытого слоя.
    /// Веса инициализируются случайно малыми значениями.
    pub fn new(input_dim: usize, hidden: usize) -> Self {
        let mut rng = RandGenerator::new();
        let w1 = (0..input_dim * hidden)
            .map(|_| rng.randn() * 0.1)
            .collect();
        let b1 = vec![0.0; hidden];
        let w2 = (0..hidden)
            .map(|_| rng.randn() * 0.1)
            .collect();
        let b2 = 0.0;
        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden,
        }
    }

    /// Предсказывает единственное значение по вектору признаков.
    pub fn predict(&self, features: &[f32]) -> f32 {
        assert_eq!(features.len(), self.input_dim);
        let mut hidden_vals = vec![0.0; self.hidden];
        for i in 0..self.hidden {
            let mut sum = self.b1[i];
            for j in 0..self.input_dim {
                sum += self.w1[j * self.hidden + i] * features[j];
            }
            hidden_vals[i] = sum.max(0.0); // ReLU
        }
        let out = self.b2
            + hidden_vals
                .iter()
                .enumerate()
                .map(|(h, &val)| self.w2[h] * val)
                .sum::<f32>();
        out
    }

    /// Обучает модель на одном примере, используя градиентный спуск.
    pub fn train(&mut self, features: &[f32], target: f32, lr: f32) {
        // Прямой проход
        let mut hidden_vals = vec![0.0; self.hidden];
        for i in 0..self.hidden {
            let mut sum = self.b1[i];
            for j in 0..self.input_dim {
                sum += self.w1[j * self.hidden + i] * features[j];
            }
            hidden_vals[i] = sum.max(0.0);
        }
        let pred: f32 = self.b2
            + hidden_vals
                .iter()
                .enumerate()
                .map(|(h, &val)| self.w2[h] * val)
                .sum::<f32>();
        let error = pred - target;

        // Градиенты выходного нейрона
        self.b2 -= lr * error;
        for h in 0..self.hidden {
            self.w2[h] -= lr * error * hidden_vals[h];
        }

        // Градиенты скрытого слоя
        let mut d_hidden = vec![0.0; self.hidden];
        for h in 0..self.hidden {
            d_hidden[h] = error * self.w2[h];
        }
        // ReLU derivative и обновление w1, b1
        for h in 0..self.hidden {
            if hidden_vals[h] > 0.0 {
                for j in 0..self.input_dim {
                    self.w1[j * self.hidden + h] -= lr * d_hidden[h] * features[j];
                }
                self.b1[h] -= lr * d_hidden[h];
            }
        }
    }

    /// Сохраняет модель в файл.
    pub fn save(&self, path: &PathBuf) {
        match serde_json::to_string(self) {
            Ok(data) => {
                if let Err(e) = fs::write(path, data) {
                    eprintln!("[neurocore] Не удалось сохранить мини-модель: {}", e);
                }
            }
            Err(e) => eprintln!("[neurocore] Ошибка сериализации мини-модели: {}", e),
        }
    }

    /// Загружает модель из файла, только если её `input_dim` совпадает с ожидаемым.
    /// Проверки на `hidden` не делается, но структура должна быть совместима.
    pub fn load(path: &PathBuf, expected_input_dim: usize) -> Option<Self> {
        if path.exists() {
            match fs::read_to_string(path) {
                Ok(data) => match serde_json::from_str::<Self>(&data) {
                    Ok(m) => {
                        if m.input_dim == expected_input_dim {
                            Some(m)
                        } else {
                            eprintln!(
                                "[neurocore] Мини-модель имеет несовпадающую размерность (in {} вместо {}), будет создана новая.",
                                m.input_dim, expected_input_dim
                            );
                            None
                        }
                    }
                    Err(e) => {
                        eprintln!("[neurocore] Ошибка парсинга мини-модели: {}", e);
                        None
                    }
                },
                Err(e) => {
                    eprintln!("[neurocore] Не удалось прочитать мини-модель: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }
}

/// Простейший генератор псевдослучайных чисел (нормальное распределение).
struct RandGenerator {
    state: u64,
}

impl RandGenerator {
    fn new() -> Self {
        Self {
            state: 123456789,
        }
    }

    /// Возвращает случайное число со стандартным нормальным распределением (≈ N(0,1)).
    fn randn(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (self.state >> 12) as f64 / (u64::MAX as f64);
        let u2 = (self.state as f64 % 10000.0) / 10000.0;
        let mag = (-2.0 * x.ln()).sqrt();
        (mag * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}
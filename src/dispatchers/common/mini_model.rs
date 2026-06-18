// src/dispatchers/common/mini_model.rs

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Небольшая нейросеть (один скрытый слой) для предсказания времени
/// выполнения прямого прохода по признакам слоя и числу потоков.
#[derive(Serialize, Deserialize)]
pub struct ForwardTimePredictor {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
    input_dim: usize,
    hidden: usize,
}

impl ForwardTimePredictor {
    /// Создаёт модель с заданной размерностью входа и числом скрытых нейронов.
    /// Веса инициализируются случайно малыми значениями.
    pub fn new(input_dim: usize, hidden: usize) -> Self {
        let mut rng = RandGenerator::new();
        let w1 = (0..input_dim * hidden)
            .map(|_| rng.randn() * 0.1)
            .collect();
        let b1 = vec![0.0; hidden];
        let w2 = (0..hidden).map(|_| rng.randn() * 0.1).collect();
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

    /// Предсказывает время (в секундах) по вектору признаков.
    pub fn predict(&self, features: &[f32]) -> f32 {
        assert_eq!(
            features.len(),
            self.input_dim,
            "Размерность признаков ({}) не совпадает с input_dim модели ({})",
            features.len(),
            self.input_dim
        );
        let mut hidden = vec![0.0; self.hidden];
        for i in 0..self.hidden {
            let mut sum = self.b1[i];
            for j in 0..self.input_dim {
                sum += self.w1[j * self.hidden + i] * features[j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }
        let mut out = self.b2;
        for i in 0..self.hidden {
            out += self.w2[i] * hidden[i];
        }
        out
    }

    /// Обучает модель на одном примере (стохастический градиентный спуск).
    pub fn train(&mut self, features: &[f32], target: f32, lr: f32) {
        assert_eq!(features.len(), self.input_dim);
        // Прямой проход
        let mut hidden = vec![0.0; self.hidden];
        for i in 0..self.hidden {
            let mut sum = self.b1[i];
            for j in 0..self.input_dim {
                sum += self.w1[j * self.hidden + i] * features[j];
            }
            hidden[i] = sum.max(0.0);
        }
        let mut out = self.b2;
        for i in 0..self.hidden {
            out += self.w2[i] * hidden[i];
        }
        let loss = out - target;

        // Обратный проход
        let d_out = 1.0;
        let d_loss = loss;

        // Градиенты по w2 и b2
        let d_w2: Vec<f32> = hidden.iter().map(|&h| d_out * h).collect();
        let d_b2 = d_out;

        // Градиенты по скрытому слою
        let mut d_hidden = vec![0.0; self.hidden];
        for i in 0..self.hidden {
            d_hidden[i] = d_out * self.w2[i];
        }

        // Производная ReLU
        let d_relu: Vec<f32> = hidden
            .iter()
            .zip(d_hidden.iter())
            .map(|(&h, &d)| if h > 0.0 { d } else { 0.0 })
            .collect();

        // Градиенты по w1 и b1
        let mut d_w1 = vec![0.0; self.input_dim * self.hidden];
        for j in 0..self.input_dim {
            for i in 0..self.hidden {
                d_w1[j * self.hidden + i] = d_relu[i] * features[j];
            }
        }
        let d_b1 = d_relu.clone();

        // Обновление параметров
        for i in 0..self.hidden {
            self.w2[i] -= lr * d_w2[i] * d_loss;
        }
        self.b2 -= lr * d_b2 * d_loss;
        for i in 0..self.input_dim * self.hidden {
            self.w1[i] -= lr * d_w1[i] * d_loss;
        }
        for i in 0..self.hidden {
            self.b1[i] -= lr * d_b1[i] * d_loss;
        }
    }

    /// Сохраняет модель в файл (ошибки логируются, работа продолжается).
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

    /// Загружает модель из файла, **только если её `input_dim` совпадает с `expected_input_dim`**.
    /// Иначе возвращает `None`, чтобы модель можно было пересоздать.
    pub fn load(path: &PathBuf, expected_input_dim: usize) -> Option<Self> {
        if path.exists() {
            match fs::read_to_string(path) {
                Ok(data) => match serde_json::from_str::<Self>(&data) {
                    Ok(m) => {
                        if m.input_dim == expected_input_dim {
                            Some(m)
                        } else {
                            eprintln!(
                                "[neurocore] Мини-модель имеет несовпадающую размерность ({} вместо {}), будет создана новая.",
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

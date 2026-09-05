// src/layers/batch_renorm/batch_renorm.rs

use std::sync::Mutex;
use crate::layers::UniversalLayer;

/// Слой BatchRenorm1d — улучшенный BatchNorm с обучаемыми поправками r и d.
///
/// Формула:
/// y = (x - μ_B) / σ_B * r * γ + (d * γ + β),
/// где μ_B, σ_B — статистики текущего батча (в режиме обучения) или скользящие средние (в режиме инференса),
/// r и d — обучаемые параметры коррекции (инициализируются 1 и 0),
/// γ и β — обычные параметры BatchNorm.
pub struct BatchRenorm1d {
    /// Количество признаков (столбцов матрицы).
    pub features: usize,
    /// Моментум для обновления скользящих статистик.
    pub momentum: f32,
    /// Эпсилон для численной стабильности.
    pub eps: f32,
    /// Текущий режим: true — обучение (используются батч-статистики), false — инференс (используются running статистики).
    pub(crate) training: bool,
    /// Состояние скользящих статистик.
    pub(crate) state: Mutex<BatchRenormState>,
}

/// Внутреннее состояние для хранения скользящих средних и дисперсий.
pub(crate) struct BatchRenormState {
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
}

impl BatchRenorm1d {
    /// Создаёт новый слой с заданным числом признаков.
    ///
    /// # Аргументы
    /// * `features` – количество признаков.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "BatchRenorm1d: features must be positive");
        Self {
            features,
            momentum: 0.1,
            eps: 1e-5,
            training: true,
            state: Mutex::new(BatchRenormState {
                running_mean: vec![0.0; features],
                running_var: vec![1.0; features],
            }),
        }
    }

    /// Создаёт слой с заданным числом признаков и параметрами моментума/эпсилон.
    pub fn with_params(features: usize, momentum: f32, eps: f32) -> Self {
        assert!(features > 0, "BatchRenorm1d: features must be positive");
        assert!(momentum >= 0.0 && momentum <= 1.0, "BatchRenorm1d: momentum must be in [0,1]");
        assert!(eps > 0.0, "BatchRenorm1d: eps must be positive");
        Self {
            features,
            momentum,
            eps,
            training: true,
            state: Mutex::new(BatchRenormState {
                running_mean: vec![0.0; features],
                running_var: vec![1.0; features],
            }),
        }
    }

    /// Устанавливает режим обучения.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Возвращает текущий режим обучения.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Сбрасывает скользящие статистики к начальным значениям.
    pub fn reset_running_stats(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.running_mean = vec![0.0; self.features];
            state.running_var = vec![1.0; self.features];
        }
    }
}

impl UniversalLayer for BatchRenorm1d {
    fn as_batch_renorm(&self) -> Option<&BatchRenorm1d> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        4 * self.features // γ, β, r, d
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}
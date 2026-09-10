// src/layers/batch_renorm/batch_renorm.rs

use std::sync::RwLock;
use crate::layers::UniversalLayer;

/// Внутреннее состояние слоя BatchRenorm1d.
///
/// Содержит скользящие статистики (`running_mean`, `running_var`),
/// обновляемые во время обучения, и флаг режима `training`.
/// Режим хранится внутри состояния, чтобы его чтение и запись
/// выполнялись под одним замком вместе со статистиками.
pub(crate) struct BatchRenormState {
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub training: bool,
}

/// Слой BatchRenorm1d — улучшенный BatchNorm с обучаемыми поправками r и d.
///
/// Формула:
///   y = (x - μ_B) / σ_B * r * γ + (d * γ + β),
/// где μ_B, σ_B — статистики текущего батча (в режиме обучения) или
/// скользящие средние (в режиме инференса),
/// r и d — обучаемые параметры коррекции (инициализируются 1 и 0),
/// γ и β — обычные параметры BatchNorm.
///
/// Параметры слоя (в порядке в общем буфере):
///   γ (features), β (features), r (features), d (features).
pub struct BatchRenorm1d {
    /// Количество признаков (столбцов матрицы).
    pub features: usize,
    /// Моментум для обновления скользящих статистик.
    pub momentum: f32,
    /// Эпсилон для численной стабильности.
    pub eps: f32,
    /// Состояние слоя (running stats + режим).
    pub(crate) state: RwLock<BatchRenormState>,
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
        Self::with_params(features, 0.1, 1e-5)
    }

    /// Создаёт слой с заданным числом признаков и параметрами моментума/эпсилон.
    ///
    /// # Паника
    /// Паникует, если `features == 0`, `momentum` вне `[0, 1]` или `eps <= 0`.
    pub fn with_params(features: usize, momentum: f32, eps: f32) -> Self {
        assert!(features > 0, "BatchRenorm1d: features must be positive");
        assert!(
            momentum >= 0.0 && momentum <= 1.0,
            "BatchRenorm1d: momentum must be in [0,1]"
        );
        assert!(eps > 0.0, "BatchRenorm1d: eps must be positive");
        Self {
            features,
            momentum,
            eps,
            state: RwLock::new(BatchRenormState {
                running_mean: vec![0.0; features],
                running_var: vec![1.0; features],
                training: true,
            }),
        }
    }

    /// Устанавливает режим обучения.
    pub fn set_training(&self, training: bool) {
        let mut guard = self.state.write().unwrap();
        guard.training = training;
    }

    /// Возвращает текущий режим обучения.
    pub fn is_training(&self) -> bool {
        self.state.read().unwrap().training
    }

    /// Сбрасывает скользящие статистики к начальным значениям.
    /// Флаг `training` не изменяется.
    pub fn reset_running_stats(&self) {
        let mut guard = self.state.write().unwrap();
        guard.running_mean = vec![0.0; self.features];
        guard.running_var = vec![1.0; self.features];
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
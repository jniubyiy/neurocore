// src/plans/loss_plan/desc.rs

use std::sync::Arc;

use super::chain::ElementChain;
use super::expr::{Aggregation, LossExpr};

/// Описание (план) функции потерь.
///
/// Хранит цепочку элементарных кубиков, способ агрегации и размерности данных.
/// Параметры `total_tasks`, `pred_features`, `target_features` соответствуют
/// новому векторному представлению:
/// - `total_tasks` – размер батча (количество сэмплов),
/// - `pred_features` – число признаков предсказания на один сэмпл,
/// - `target_features` – число признаков цели на один сэмпл (обычно равно `pred_features`).
///
/// Первый кубик цепочки должен принимать `pred_features + target_features` столбцов
/// (например, `Sub::new(pred_features)`).
#[derive(Debug, Clone)]
pub struct LossDesc {
    pub chain: Arc<ElementChain>,
    pub aggregation: Aggregation,
    pub total_tasks: usize,
    pub pred_features: usize,
    pub target_features: usize,
}

impl LossDesc {
    /// Создаёт описание на основе готовой цепочки кубиков и параметров данных.
    ///
    /// # Аргументы
    /// * `chain` – цепочка элементарных кубиков. Первый кубик должен быть способен
    ///   принять матрицу с числом столбцов, равным `pred_features + target_features`.
    /// * `aggregation` – способ агрегирования (сумма или среднее).
    /// * `total_tasks` – размер батча (количество сэмплов).
    /// * `pred_features` – количество признаков предсказания на один сэмпл.
    /// * `target_features` – количество признаков целевой переменной на один сэмпл.
    pub fn from_chain(
        chain: ElementChain,
        aggregation: Aggregation,
        total_tasks: usize,
        pred_features: usize,
        target_features: usize,
    ) -> Self {
        Self {
            chain: Arc::new(chain),
            aggregation,
            total_tasks,
            pred_features,
            target_features,
        }
    }

    /// Собирает готовое выражение потерь, обёрнутое в `Arc` для безопасного разделения между потоками.
    pub fn build(self) -> Arc<LossExpr> {
        Arc::new(LossExpr::new(
            self.chain,
            self.aggregation,
            self.total_tasks,
            self.pred_features,
            self.target_features,
        ))
    }
}
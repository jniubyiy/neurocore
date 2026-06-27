// src/loss_plan/desc.rs

use std::sync::Arc;

use super::chain::ElementChain;
use super::expr::{Aggregation, LossExpr};

/// Описание (план) функции потерь.
///
/// Позволяет сконструировать готовое выражение [`LossExpr`] через метод [`build`].
pub struct LossDesc {
    pub chain: ElementChain,
    pub aggregation: Aggregation,
    pub total_tasks: usize,
    pub pred_features: usize,
    pub target_features: usize,
}

impl LossDesc {
    /// Создаёт описание на основе готовой цепочки кубиков и параметров агрегации.
    ///
    /// * `chain` — цепочка элементарных кубиков.
    /// * `aggregation` — способ агрегирования (сумма или среднее).
    /// * `total_tasks` — общее количество задач (например, элементов батча).
    /// * `pred_features` — количество признаков предсказания на одну задачу.
    /// * `target_features` — количество признаков целевой переменной на одну задачу.
    pub fn from_chain(
        chain: ElementChain,
        aggregation: Aggregation,
        total_tasks: usize,
        pred_features: usize,
        target_features: usize,
    ) -> Self {
        Self {
            chain,
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
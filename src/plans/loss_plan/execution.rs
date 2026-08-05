// src/plans/loss_plan/execution.rs

use std::sync::Arc;
use faer::Mat;
use crate::compute_manager::cpu::{Scheduler, WorkerPool};
use super::expr::LossExpr;

/// Вычисляет значение функции потерь и градиент по предсказанию на CPU.
///
/// # Аргументы
/// * `expr` – выражение потерь (цепочка кубиков + агрегация).
/// * `pred` – матрица предсказаний `(batch, pred_features)`.
/// * `target` – матрица целей `(batch, target_features)`.
/// * `_scheduler` – планировщик (зарезервирован для будущей параллелизации по батчам).
/// * `_pool` – пул потоков (зарезервирован).
///
/// # Возвращает
/// * `loss` – агрегированное значение потерь (скаляр).
/// * `grad_pred` – матрица градиентов по pred той же размерности, что и `pred`.
pub fn compute_loss_mat(
    expr: &Arc<LossExpr>,
    pred: &Mat<f32>,
    target: &Mat<f32>,
    _scheduler: &mut Scheduler,
    _pool: &WorkerPool,
) -> (f32, Mat<f32>) {
    let pred_feat = expr.pred_features();
    let target_feat = expr.target_features();
    let batch = pred.nrows();
    assert_eq!(batch, target.nrows(), "Pred and target must have the same batch size");
    assert_eq!(pred.ncols(), pred_feat, "Pred cols mismatch");
    assert_eq!(target.ncols(), target_feat, "Target cols mismatch");

    let in_features = pred_feat + target_feat;

    // Формируем полную матрицу [pred | target]
    let mut full_input = Mat::zeros(batch, in_features);
    for i in 0..batch {
        for j in 0..pred_feat {
            full_input[(i, j)] = pred[(i, j)];
        }
        for j in 0..target_feat {
            full_input[(i, pred_feat + j)] = target[(i, j)];
        }
    }

    let (loss_vec, intermediates) = expr.forward_chunk(&full_input);
    let loss = expr.aggregate_loss(&loss_vec);

    let grad_loss = vec![1.0f32; batch];
    let grad_full = expr.backward_chunk(&intermediates, &grad_loss);

    // Извлекаем градиент только по pred (первые pred_feat столбцов)
    let mut grad_pred = Mat::zeros(batch, pred_feat);
    for i in 0..batch {
        for j in 0..pred_feat {
            grad_pred[(i, j)] = grad_full[(i, j)];
        }
    }

    (loss, grad_pred)
}
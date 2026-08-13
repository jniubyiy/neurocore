// src/plans/loss_plan/execution.rs

use std::sync::Arc;
use faer::Mat;
use crate::compute_manager::cpu::{Scheduler, WorkerPool};
use crate::compute_manager::matrix_buffer::{MatrixBuffer, TempMatrixPool};
use super::expr::LossExpr;

/// Вычисляет значение функции потерь и градиент по предсказанию на CPU (матричная версия).
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

/// Вычисляет значение функции потерь и градиент по предсказанию на CPU с использованием
/// управляемых буферов `MatrixBuffer` и пула `TempMatrixPool`.
///
/// # Аргументы
/// * `expr` – выражение потерь.
/// * `pred` – буфер предсказаний `(batch, pred_features)` (CPU).
/// * `target` – буфер целей `(batch, target_features)` (CPU).
/// * `pool` – пул временных матриц для выделения промежуточных буферов.
///
/// # Возвращает
/// * `loss` – агрегированное значение потерь.
/// * `grad_pred` – буфер градиентов по pred размерности `(batch, pred_features)`.
///
/// # Паника
/// Паникует, если `pred` или `target` являются GPU-буферами.
pub fn compute_loss_mat_buffered(
    expr: &Arc<LossExpr>,
    pred: &MatrixBuffer,
    target: &MatrixBuffer,
    pool: &mut TempMatrixPool,
) -> (f32, MatrixBuffer) {
    assert!(!pred.is_gpu() && !target.is_gpu(),
        "compute_loss_mat_buffered supports only CPU buffers");

    let pred_feat = expr.pred_features();
    let target_feat = expr.target_features();
    let batch = pred.rows();
    assert_eq!(batch, target.rows(), "Pred and target must have the same batch size");
    assert_eq!(pred.cols(), pred_feat, "Pred cols mismatch");
    assert_eq!(target.cols(), target_feat, "Target cols mismatch");

    let in_features = pred_feat + target_feat;

    // Формируем полный вход [pred | target]
    let mut full_input = pool.acquire(batch, in_features);
    {
        let src_pred = pred.as_slice();
        let src_target = target.as_slice();
        let dst_full = full_input.as_slice_mut();

        // Копируем pred в первые столбцы, target в следующие
        for c in 0..pred_feat {
            for r in 0..batch {
                dst_full[c * batch + r] = src_pred[c * batch + r];
            }
        }
        for c in 0..target_feat {
            for r in 0..batch {
                dst_full[(c + pred_feat) * batch + r] = src_target[c * batch + r];
            }
        }
    }

    // Прямой проход
    let (loss_vec, intermediates) = expr.forward_chunk_buffered(&full_input, pool);
    let loss = expr.aggregate_loss(&loss_vec);

    // full_input больше не нужен
    pool.release(full_input);

    // Градиент по агрегированному loss (вектор единиц)
    let grad_loss = vec![1.0f32; batch];
    let grad_full = expr.backward_chunk_buffered(&intermediates, &grad_loss, pool);

    // Извлекаем градиент только по pred (первые pred_feat столбцов)
    let mut grad_pred = pool.acquire(batch, pred_feat);
    {
        let src_grad = grad_full.as_slice();
        let dst_grad = grad_pred.as_slice_mut();

        for c in 0..pred_feat {
            for r in 0..batch {
                dst_grad[c * batch + r] = src_grad[c * batch + r];
            }
        }
    }

    // Освобождаем grad_full и промежуточные буферы
    pool.release(grad_full);
    for (inp, outp) in intermediates {
        pool.release(inp);
        pool.release(outp);
    }

    (loss, grad_pred)
}
// src/plans/loss_plan/execution.rs

use std::sync::Arc;

use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use super::expr::LossExpr;

/// Вычисляет значение функции потерь и градиент по предсказанию на CPU с использованием
/// управляемых буферов `MatrixBufferHandle` и пула `TempMatrixPool`.
///
/// # Аргументы
/// * `expr` – выражение потерь.
/// * `pred` – дескриптор предсказаний `(batch, pred_features)` (CPU).
/// * `target` – дескриптор целей `(batch, target_features)` (CPU).
/// * `pool` – пул временных матриц для выделения промежуточных буферов.
///
/// # Возвращает
/// * `loss` – агрегированное значение потерь.
/// * `grad_pred` – дескриптор градиентов по pred размерности `(batch, pred_features)`.
///
/// # Паника
/// Паникует, если `pred` или `target` являются GPU-буферами.
pub fn compute_loss_mat_buffered(
    expr: &Arc<LossExpr>,
    pred: &MatrixBufferHandle,
    target: &MatrixBufferHandle,
    pool: &mut TempMatrixPool,
) -> (f32, MatrixBufferHandle) {
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
    let full_input = pool.acquire(batch, in_features);
    {
        let src_pred_guard = pred.read();
        let src_pred = src_pred_guard.as_slice().expect("Pred must be CPU");
        let src_target_guard = target.read();
        let src_target = src_target_guard.as_slice().expect("Target must be CPU");

        let mut dst_guard = full_input.write();
        let dst_full = dst_guard.as_slice_mut().expect("Full input must be CPU");

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
    let grad_pred = pool.acquire(batch, pred_feat);
    {
        let src_grad_guard = grad_full.read();
        let src_grad = src_grad_guard.as_slice().expect("Grad full must be CPU");
        let mut dst_guard = grad_pred.write();
        let dst_grad = dst_guard.as_slice_mut().expect("Grad pred must be CPU");

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